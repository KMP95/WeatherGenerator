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
from omegaconf import DictConfig, OmegaConf

from weathergen.train.loss_modules.loss_module_base import LossValues
from weathergen.train.loss_modules.loss_module_ssl import LossLatentSSLStudentTeacher
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)

_AGGREGATIONS = ("cell", "state", "parent", "global", "pooled", "class_token")


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
          aggregation: cell        # cell | state | parent | global | pooled | class_token,
                                   # or a list, e.g. [cell, state] -- one test per axis
          term_weights: {cell: 0.2, state: 0.2}   # optional per-term weights (default: weight)
          target_branch: false     # false | true (pool both branches into one test) |
                                   # separate (one test per branch, averaged) -- E3 / E3c
          parent_levels_up: 1      # only for aggregation=parent
          source: latent_state     # latent_state (after the forecast engine) | encoder
                                   # (before it -- the tensor the targets are made of) |
                                   # sigreg_proj | <loss_fcts head name>
          num_projections: 256
          knots: 17
          t_max: 3.0
          num_cells: 0             # cell test: cells/groups tested per step (0 = all)
          state_num_cells: 256     # state/pooled tests: sparse cell subset per step (0 = all;
                                   # not recommended, neighbouring cells are correlated)
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
        # One aggregation (str) or several (list): each is an independent test along its
        # own axis and enters the loss with its own weight. ``cell`` (sample axis = weather
        # states) and ``state`` (sample axis = cells) are the two marginals of the
        # (state, cell) array; either alone is blind to the collapse along the other axis.
        agg = sr.get("aggregation", "cell")
        if OmegaConf.is_list(agg) or isinstance(agg, (list, tuple)):
            self.aggregations = [str(a) for a in agg]
        else:
            self.aggregations = [str(agg)]
        self.aggregation = self.aggregations[0]  # kept for messages / single-term paths
        tw = sr.get("term_weights", None) or {}
        self.term_weights = {a: float(tw.get(a, self.sigreg_weight)) for a in self.aggregations}
        # ``target_branch``: also regularise the target branch's encoder tokens
        # (``targets.latent["encoder_tokens"]``, published by SharedEncoderTarget) by pooling
        # them into the sample axis with the context tokens -- LeWorldModel applies SIGReg to
        # every time step's embedding. Doubles N for the cell test.
        # ``target_branch``: false | true (pool both branches' states into one test, as
        # LeWorldModel pools all time steps) | "separate" (test each branch on its own and
        # average). Pooling the two views of one trajectory into a single test lets the
        # network lower the statistic by making the pair *anti-correlated* (antithetic
        # samples estimate a characteristic function with less variance than independent
        # ones): E3 showed sigreg_cell sinking to 0.86 while the temporal contrast rose to
        # 3.2 (rho ~ -0.5 between E(x_t) and E(x_{t+6h})). "separate" removes that incentive.
        tb = sr.get("target_branch", False)
        self.target_branch = str(tb).lower() if isinstance(tb, str) else bool(tb)
        if self.target_branch not in (False, True, "pool", "separate"):
            raise ValueError(f"sigreg.target_branch must be false|true|pool|separate, got {tb!r}")
        if self.target_branch == "pool":
            self.target_branch = True
        self.parent_levels_up = int(sr.get("parent_levels_up", 1))
        self.source = str(sr.get("source", "latent_state"))
        self.num_projections = int(sr.get("num_projections", 256))
        self.knots = int(sr.get("knots", 17))
        self.t_max = float(sr.get("t_max", 3.0))
        self.num_cells = int(sr.get("num_cells", 256))
        # Cells are not independent samples of one field: neighbouring cells are
        # correlated, so a test whose sample axis is the cells (`state`, `pooled`) sees far
        # fewer independent draws than C and its null value is ~1.05 * C / N_eff instead of
        # 1.05 (E2d, pooled over all cells, sat at 57-73 and could only go down by
        # decorrelating the field). Testing a sparse random subset of cells, redrawn every
        # step, keeps the samples far enough apart to be roughly independent: the null
        # returns to ~1.05 and the term penalises a genuine global code (m identical
        # values -> ~m), not spatial smoothness. 0 = all cells (not recommended).
        self.state_num_cells = int(sr.get("state_num_cells", 256))
        self.min_samples = int(sr.get("min_samples", 4))
        self.warmup_steps = int(sr.get("warmup_steps", 0))

        proj = sr.get("projector", None) or {}
        self.projector_mode = str(proj.get("mode", "none"))
        self.projector_out_dim = int(proj.get("out_dim", 128))
        self.projector_seed = int(proj.get("seed", 1234))
        self._random_projector: torch.Tensor | None = None

        for a in self.aggregations:
            assert a in _AGGREGATIONS, (
                f"Unknown sigreg.aggregation {a!r}; expected one of {_AGGREGATIONS}."
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
        if self.source == "encoder":
            # Encoder output before the forecast engine (published by Model.forward). This
            # is the tensor the JEPA targets are made of; regularising it, rather than the
            # post-forecast-engine latent state, closes the route where the encoder keeps
            # the weather at a tiny amplitude and the forecast engine re-inflates it.
            return latent.get("encoder_tokens", None)
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

    def _subsample_cells(self, z: torch.Tensor, m: int) -> torch.Tensor:
        """Random subset of m cells (same on every rank, redrawn each step); m <= 0 = all."""
        c = z.shape[1]
        if m <= 0 or m >= c:
            return z
        gen = torch.Generator(device="cpu").manual_seed(0x5157 * 104_729 + self._step)
        idx = torch.randperm(c, generator=gen)[:m]
        return z.index_select(1, idx.to(z.device))

    def _draw_projections(self, dim: int, device, dtype) -> torch.Tensor:
        gen = torch.Generator(device="cpu").manual_seed(0x5157 * 1_000_003 + self._step)
        a = torch.randn(dim, self.num_projections, generator=gen, dtype=torch.float32)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        return a.to(device=device, dtype=dtype)

    # ---------------------------------------------------------------- SIGReg

    def _compute_sigreg_terms(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """One statistic per configured aggregation, sharing the projection matrix."""
        out: dict[str, torch.Tensor] = {}
        for agg in self.aggregations:
            self.aggregation = agg
            stat = self._compute_sigreg(z)
            if stat is not None:
                out[agg] = stat
        self.aggregation = self.aggregations[0]
        return out

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
        if self.aggregation == "state":
            # One test per weather state with its C cells as the samples: the transpose of
            # `cell`. Every state's cells live on one rank, so no cross-rank pooling; the
            # statistic is averaged over the local states (and the directions).
            zs = self._subsample_cells(z, self.state_num_cells)
            proj = zs @ self._draw_projections(zs.shape[-1], zs.device, zs.dtype)  # [V, m, P]
            return sigreg_from_projections(
                proj.transpose(0, 1).contiguous(),
                sample_dim=0,
                knots=self.knots,
                t_max=self.t_max,
                cross_rank=False,
            )
        if self.aggregation == "pooled":
            # Every (state, cell) pair is a sample: [1, V*m, D] with sample axis 1, on the
            # same sparse cell subset as `state` (see state_num_cells).
            zs = self._subsample_cells(z, self.state_num_cells)
            proj = (zs.reshape(-1, zs.shape[-1]) @ self._draw_projections(
                zs.shape[-1], zs.device, zs.dtype
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

        zt = None
        if self.target_branch:
            if isinstance(getattr(targets, "latent", None), dict):
                zt = targets.latent.get("encoder_tokens", None)
            if zt is None:
                logger.warning(
                    "sigreg.target_branch is set but targets.latent has no 'encoder_tokens' "
                    "(needs SharedEncoderTarget with publish_encoder_tokens); using the "
                    "context branch only."
                )
            elif zt.shape != z.shape:
                logger.warning(
                    "target-branch tokens %s do not match context tokens %s; skipping them.",
                    tuple(zt.shape),
                    tuple(z.shape),
                )
                zt = None

        if zt is not None and self.target_branch is True:
            # Pooled: both branches' states become samples of the same test. Draws from
            # one trajectory are correlated, and the network can exploit that (see the
            # constructor comment); kept as an option for the comparison with LeWorldModel.
            terms = self._compute_sigreg_terms(torch.cat([z, zt.to(z.dtype)], dim=0))
        elif zt is not None and self.target_branch == "separate":
            # Each branch is its own test with N samples; the statistics are averaged.
            # No sample ever sees its own trajectory partner inside a test, so there is
            # nothing to gain from anti-correlating the two encodings.
            t_ctx = self._compute_sigreg_terms(z)
            self._step -= 1  # same projections / cell subset for the target branch
            t_tgt = self._compute_sigreg_terms(zt.to(z.dtype))
            self._step += 1
            terms = {k: 0.5 * (t_ctx[k] + t_tgt[k]) for k in t_ctx if k in t_tgt}
            for k in t_ctx:
                if k in t_tgt:
                    losses_all[f"sigreg_{k}_ctx"] = t_ctx[k].item()
                    losses_all[f"sigreg_{k}_tgt"] = t_tgt[k].item()
        else:
            terms = self._compute_sigreg_terms(z)
        if not terms:
            return LossValues(loss=loss, losses_all=losses_all, stddev_all=base.stddev_all)

        total = 0.0
        for agg, stat in terms.items():
            w = self.term_weights[agg] * warmup
            loss = loss + w * stat
            total = total + stat
            if len(terms) > 1:
                losses_all[f"sigreg_{agg}"] = stat.item()
                losses_all[f"sigreg_{agg}_weight"] = w
        # `sigreg` stays the headline number: the single statistic, or the mean of the
        # terms when several axes are regularised (each is ~1.05 under the null).
        sigreg = total / len(terms)
        losses_all["sigreg"] = sigreg.item()
        losses_all["sigreg_weight"] = self.sigreg_weight * warmup
        # Peak GPU memory of this rank so far (GB): the number that decides how many
        # weather states fit per GPU, i.e. the SIGReg sample count per node.
        if torch.cuda.is_available():
            losses_all["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

        return LossValues(loss=loss, losses_all=losses_all, stddev_all=base.stddev_all)
