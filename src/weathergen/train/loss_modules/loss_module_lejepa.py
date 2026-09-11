# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
LeJEPA (Balestriero & LeCun, 2025) for the WeatherGenerator.

`LossLatentSSLLeJEPA` inherits the whole JEPA/deep-SSL machinery from
`LossLatentSSLStudentTeacher` and only *adds* a SIGReg term on the student's latent
representation, so a LeJEPA run and a temporal-JEPA run differ by exactly one loss term.

The student latent has shape ``[V, C, D]`` -- V weather states on this rank, C HEALPix
cells, D embedding dimensions. Which axis plays the role of the *statistical sample* is
the central design decision, and it is what `aggregation` selects:

    pooled  reshape to [V*C, D]; every (state, cell) pair is a sample. One test.
            This is what upstream/shmh40/dev/lejepa-per-cell does. It asks "is the
            spatial distribution of cell embeddings Gaussian?", which an encoder that
            emits a fixed climatology field -- identical for every date -- satisfies
            perfectly. The collapse we care about is invisible to it. Baseline only.

    cell    one test per cell, sample axis = weather states (V x ranks). Asks "does this
            location's embedding vary across weather states?". Correct notion of
            collapse, but only V*world_size samples per test.

    parent  average the 4^k children of each HEALPix level-(L-k) cell first, then one
            test per parent cell, sample axis = weather states. Interpolates between
            `cell` (k=0) and near-`global`. Spatial averaging cuts the per-test noise
            while keeping some locality. Valid because the model uses NESTED ordering,
            where a parent's children are contiguous (parent index = child index // 4^k).

    global  average over *all* cells to get one vector per weather state, then a single
            test with sample axis = weather states. This is the exact analogue of the
            image setting: one weather state <-> one image, cells <-> patch tokens, and
            just as a ViT pools patch tokens into one embedding per image before SIGReg
            is applied across the batch, we pool cells into one embedding per state.
            Note this constrains only the global summary, not any individual cell.

The empirical characteristic function is reduced across DDP ranks with the
*differentiable* ``torch.distributed.nn.all_reduce``. Because the ECF is a mean of
per-sample terms, summing the per-rank numerators and counts before the nonlinearity
gives exactly the statistic of the concatenated global batch -- which is why DDP scales
the effective sample count whereas gradient accumulation does not.
"""

import logging

import torch
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossValues
from weathergen.train.loss_modules.loss_module_ssl import LossLatentSSLStudentTeacher
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)

_AGGREGATIONS = ("cell", "parent", "global", "pooled", "class_token")


def _gaussian_quadrature(knots: int, t_max: float, device, dtype):
    """Trapezoidal weights on [0, t_max] windowed by the standard-Gaussian CF.

    Matches the official LeJEPA implementation, which integrates on [0, t_max] and
    doubles the interior weights instead of integrating over [-t_max, t_max].
    """
    t = torch.linspace(0.0, t_max, knots, device=device, dtype=dtype)
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=device, dtype=dtype)
    weights[0] = dt
    weights[-1] = dt
    phi = torch.exp(-t.square() / 2.0)
    return t, weights * phi, phi


def _dist_active() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Differentiable cross-rank sum.

    ``torch.distributed.nn.all_reduce`` backpropagates through the collective;
    ``torch.distributed.all_reduce`` does not, and would silently drop every other
    rank's contribution to the SIGReg gradient.
    """
    if not _dist_active():
        return x
    import torch.distributed.nn as dist_nn

    return dist_nn.all_reduce(x, op=torch.distributed.ReduceOp.SUM)


def sigreg_from_projections(
    proj: torch.Tensor,
    sample_dim: int,
    knots: int,
    t_max: float,
    cross_rank: bool,
) -> torch.Tensor:
    """Epps-Pulley SIGReg statistic from pre-computed 1-D projections.

    Args:
        proj: projected representations. ``sample_dim`` indexes independent samples;
            every other axis is an independent test, averaged at the end.
        sample_dim: axis of ``proj`` holding the statistical samples.
        knots: quadrature nodes for the characteristic-function integral.
        t_max: upper limit of the frequency integral.
        cross_rank: pool the empirical CF across DDP ranks before comparing against the
            Gaussian CF. Essential whenever the per-rank sample count is small.

    Returns:
        Scalar statistic; ~1 for a standard Gaussian at any sample count, growing
        linearly in the sample count as the distribution degenerates.
    """
    t, weights, phi = _gaussian_quadrature(knots, t_max, proj.device, proj.dtype)

    x_t = proj.unsqueeze(-1) * t
    cos_sum = x_t.cos().sum(dim=sample_dim)
    sin_sum = x_t.sin().sum(dim=sample_dim)
    n_total = torch.tensor(
        float(proj.shape[sample_dim]), device=proj.device, dtype=proj.dtype
    )

    if cross_rank:
        cos_sum = _all_reduce_sum(cos_sum)
        sin_sum = _all_reduce_sum(sin_sum)
        n_total = _all_reduce_sum(n_total)

    err = (cos_sum / n_total - phi).square() + (sin_sum / n_total).square()
    return ((err @ weights) * n_total).mean()


class LossLatentSSLLeJEPA(LossLatentSSLStudentTeacher):
    """Temporal JEPA + SIGReg.

    Config, under the loss term's ``mode_cfg`` (i.e. ``training_config``)::

        sigreg:
          weight: 0.2
          aggregation: cell        # cell | parent | global | pooled | class_token
          parent_levels_up: 1      # only for aggregation=parent
          source: latent_state     # latent_state | sigreg_proj | <loss_fcts head name>
          num_projections: 256
          knots: 17
          t_max: 3.0
          num_cells: 256           # cells/groups subsampled per step (0 = all)
          min_samples: 4
          warmup_steps: 0
          projector:
            mode: none             # none | random | learned
            out_dim: 128           # random and learned
            hidden_dims: [2048]    # learned only
            seed: 1234             # random only

    ``projector.mode: learned`` builds an MLP on the *model* (see model.py) and exposes
    its output as ``sigreg_proj``; set ``source: sigreg_proj`` to use it. ``random`` uses
    a fixed, non-learnable sketch computed here, which needs no model change.
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        super().__init__(cf, mode_cfg, stage, device, **losses)
        self.name = "LossLatentSSLLeJEPA"

        sr = mode_cfg.get("sigreg", None) or {}
        self.sigreg_weight = float(sr.get("weight", 0.2))
        self.aggregation = str(sr.get("aggregation", "cell"))
        self.parent_levels_up = int(sr.get("parent_levels_up", 1))
        self.source = str(sr.get("source", "latent_state"))
        self.num_projections = int(sr.get("num_projections", 256))
        self.knots = int(sr.get("knots", 17))
        self.t_max = float(sr.get("t_max", 3.0))
        self.num_cells = int(sr.get("num_cells", 256))
        self.min_samples = int(sr.get("min_samples", 4))
        self.warmup_steps = int(sr.get("warmup_steps", 0))

        proj = sr.get("projector", None) or {}
        self.projector_mode = str(proj.get("mode", "none"))
        self.projector_out_dim = int(proj.get("out_dim", 128))
        self.projector_seed = int(proj.get("seed", 1234))
        self._random_projector: torch.Tensor | None = None

        assert self.aggregation in _AGGREGATIONS, (
            f"Unknown sigreg.aggregation {self.aggregation!r}; expected one of {_AGGREGATIONS}."
        )
        assert self.projector_mode in ("none", "random", "learned"), (
            f"Unknown sigreg.projector.mode {self.projector_mode!r}."
        )
        if self.projector_mode == "learned" and self.source != "sigreg_proj":
            logger.warning(
                "sigreg.projector.mode='learned' but source=%r; the learned projector "
                "output is published as 'sigreg_proj' and will be ignored.",
                self.source,
            )

        # Incremented once per compute_loss call. All ranks call compute_loss equally
        # often, so this stays in lockstep and can seed identical projection directions
        # and cell subsets everywhere -- required for the cross-rank reduction to be
        # combining like with like.
        self._step = 0

    # ---------------------------------------------------------------- helpers

    def _select_representation(self, preds) -> torch.Tensor | None:
        latent = preds.latent[0]
        if self.source == "latent_state":
            state = latent.get("latent_state", None)
            if state is None:
                return None
            if self.aggregation == "class_token":
                # The faithful analogue of the paper's setting: LeJEPA applies SIGReg to
                # whatever `forward()` returns, which for a ViT is the class token -- a
                # learned query that has attended to every patch. `global` mean-pools the
                # cells instead, which is fixed and unweighted and cannot learn what to
                # attend to. The class token is produced by the encoder's auxiliary
                # queries (encoder.q_aux) and is only non-empty when the config sets
                # num_class_tokens >= 1.
                ct = getattr(state, "class_token", None)
                if ct is None or ct.numel() == 0 or ct.shape[1] == 0:
                    raise ValueError(
                        "sigreg.aggregation='class_token' needs num_class_tokens >= 1 in "
                        "the model config; the encoder currently emits no class token."
                    )
                return ct
            return state.patch_tokens
        return latent.get(self.source, None)

    def _apply_random_projector(self, z: torch.Tensor) -> torch.Tensor:
        """Fixed (non-learnable) Gaussian sketch D -> out_dim, constant across steps."""
        d = z.shape[-1]
        if self._random_projector is None or self._random_projector.shape[0] != d:
            gen = torch.Generator(device="cpu").manual_seed(self.projector_seed)
            m = torch.randn(d, self.projector_out_dim, generator=gen) / (d**0.5)
            self._random_projector = m.to(device=z.device, dtype=z.dtype)
        return z @ self._random_projector

    def _reduce_cells(self, z: torch.Tensor) -> torch.Tensor:
        """Collapse the cell axis according to `aggregation`. Returns [V, G, D]."""
        v, c, d = z.shape
        if self.aggregation == "cell":
            return z
        if self.aggregation == "class_token":
            # Already one vector per weather state per class token: the middle axis is the
            # class tokens, not cells, so there is nothing to collapse. With the usual
            # num_class_tokens=1 this is a single Gaussianity test over the batch, which is
            # exactly the paper's setting.
            return z
        if self.aggregation == "global":
            return z.mean(dim=1, keepdim=True)
        # parent: average the 4^k contiguous children of each level-(L-k) cell.
        group = 4**self.parent_levels_up
        if c % group != 0:
            logger.warning(
                "Cell count %d is not divisible by 4^%d; falling back to aggregation='cell'.",
                c,
                self.parent_levels_up,
            )
            return z
        return z.reshape(v, c // group, group, d).mean(dim=2)

    def _draw_projections(self, dim: int, device, dtype) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(0x5157 * 1_000_003 + self._step)
        a = torch.randn(dim, self.num_projections, generator=gen, dtype=torch.float32)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        return a.to(device=device, dtype=dtype)

    # ---------------------------------------------------------------- SIGReg

    def _compute_sigreg(self, z: torch.Tensor) -> torch.Tensor | None:
        if z.dim() != 3:
            logger.warning(
                "SIGReg expects [views, cells, dim]; got %s. Skipping.", tuple(z.shape)
            )
            return None

        z = z.float()
        if self.projector_mode == "random":
            z = self._apply_random_projector(z)

        world_size = torch.distributed.get_world_size() if _dist_active() else 1

        if self.aggregation == "pooled":
            # Every (state, cell) pair is a sample: [1, V*C, D] with sample axis 1.
            proj = (z.reshape(-1, z.shape[-1]) @ self._draw_projections(
                z.shape[-1], z.device, z.dtype
            )).unsqueeze(0)
            return sigreg_from_projections(
                proj, sample_dim=1, knots=self.knots, t_max=self.t_max, cross_rank=True
            )

        z = self._reduce_cells(z)
        views, groups, dim = z.shape

        n_eff = views * world_size
        if n_eff < self.min_samples:
            logger.warning(
                "SIGReg aggregation=%r has only %d samples (views=%d x world_size=%d), "
                "below min_samples=%d. Raise training_config.model_input num_samples, use "
                "more GPUs, or switch to aggregation='pooled'.",
                self.aggregation,
                n_eff,
                views,
                world_size,
                self.min_samples,
            )
            return None

        if 0 < self.num_cells < groups:
            # Identical subset on every rank, so the reduced ECF stays per-group coherent.
            gen = torch.Generator(device="cpu").manual_seed(0x5157 * 7_919 + self._step)
            idx = torch.randperm(groups, generator=gen)[: self.num_cells]
            z = z.index_select(1, idx.to(z.device))

        proj = z @ self._draw_projections(dim, z.device, z.dtype)  # [V, G', P]
        return sigreg_from_projections(
            proj, sample_dim=0, knots=self.knots, t_max=self.t_max, cross_rank=True
        )

    # ---------------------------------------------------------------- entry point

    def compute_loss(self, preds, targets, metadata, istep=0, **kwargs) -> LossValues:
        base = super().compute_loss(preds, targets, metadata, istep=istep, **kwargs)
        loss, losses_all = base.loss, dict(base.losses_all)

        warmup = (
            min(1.0, self._step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        )
        self._step += 1

        z = self._select_representation(preds)
        if z is None:
            logger.warning(
                "SIGReg source %r not found in latent predictions (%s); skipping.",
                self.source,
                sorted(preds.latent[0].keys()),
            )
            return LossValues(loss=loss, losses_all=losses_all, stddev_all=base.stddev_all)

        sigreg = self._compute_sigreg(z)
        if sigreg is None:
            return LossValues(loss=loss, losses_all=losses_all, stddev_all=base.stddev_all)

        weight = self.sigreg_weight * warmup
        loss = loss + weight * sigreg
        losses_all["sigreg"] = sigreg.item()
        losses_all["sigreg_weight"] = weight

        return LossValues(loss=loss, losses_all=losses_all, stddev_all=base.stddev_all)
