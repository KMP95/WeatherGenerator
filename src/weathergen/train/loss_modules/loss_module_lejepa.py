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
representation. Everything else -- masks, teacher, deep-SSL taps, context loss --
behaves exactly as in the temporal-JEPA baseline, so a LeJEPA run and a TJEPA run
differ by one loss term and are directly comparable.

Relation to `upstream/shmh40/dev/lejepa-per-cell`
-------------------------------------------------
That branch's `LossLeJEPA` is a standalone `LossModuleBase` subclass (its author
flagged this as "not good") and computes SIGReg with the *cells* flattened into the
sample axis::

    sigreg_loss(student_raw.reshape(-1, D), ...)      # N = views * cells

That asks "is the spatial distribution of cell embeddings Gaussian?", which a single
frozen climatology field satisfies perfectly -- the collapse mode it is meant to
prevent is invisible to it. Set ``aggregation: pooled`` to reproduce that behaviour
as a baseline.

The default here, ``aggregation: cell``, instead treats the *weather states*
(views x DDP ranks) as the sample axis and applies SIGReg per HEALPix cell. That asks
"does a given cell's embedding vary like a Gaussian across weather states?", which is
the property that actually rules out collapse. Because the per-rank sample count is
tiny, the empirical characteristic function is reduced across ranks with the
*differentiable* ``torch.distributed.nn.all_reduce`` -- plain ``dist.all_reduce``
would silently drop the other ranks' gradient contributions.
"""

import logging

import torch
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossValues
from weathergen.train.loss_modules.loss_module_ssl import LossLatentSSLStudentTeacher
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)


def _gaussian_quadrature(knots: int, t_max: float, device, dtype):
    """Trapezoidal weights on [0, t_max] windowed by the standard-Gaussian CF.

    Returns (t, weights, phi) where phi = exp(-t^2/2) is the characteristic function
    of N(0, 1) evaluated at the quadrature nodes.
    """
    t = torch.linspace(0.0, t_max, knots, device=device, dtype=dtype)
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2 * dt, device=device, dtype=dtype)
    weights[0] = dt
    weights[-1] = dt
    phi = torch.exp(-t.square() / 2.0)
    return t, weights * phi, phi


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Differentiable cross-rank sum.

    ``torch.distributed.nn.all_reduce`` backpropagates through the collective;
    ``torch.distributed.all_reduce`` does not and would corrupt the SIGReg gradient.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return x
    if torch.distributed.get_world_size() == 1:
        return x
    import torch.distributed.nn as dist_nn

    return dist_nn.all_reduce(x, op=torch.distributed.ReduceOp.SUM)


def sigreg_from_projections(
    proj: torch.Tensor,
    sample_dims: tuple[int, ...],
    knots: int,
    t_max: float,
    cross_rank: bool,
) -> torch.Tensor:
    """SIGReg statistic from pre-computed 1-D projections.

    Args:
        proj: projected representations; the axes in ``sample_dims`` are the
            statistical sample axis, all remaining axes are averaged over at the end.
        sample_dims: which axes of ``proj`` index independent samples.
        knots: number of quadrature nodes for the characteristic-function integral.
        t_max: upper limit of the frequency integral.
        cross_rank: reduce the empirical CF across DDP ranks before comparing to the
            Gaussian CF. Required when the per-rank sample count is small.

    Returns:
        Scalar SIGReg statistic (higher = further from N(0, 1) along the projections).
    """
    t, weights, phi = _gaussian_quadrature(knots, t_max, proj.device, proj.dtype)

    # [*proj.shape, knots]
    x_t = proj.unsqueeze(-1) * t

    n_local = 1
    for d in sample_dims:
        n_local *= proj.shape[d]

    cos_sum = x_t.cos().sum(dim=sample_dims)
    sin_sum = x_t.sin().sum(dim=sample_dims)
    n_total = torch.tensor(float(n_local), device=proj.device, dtype=proj.dtype)

    if cross_rank:
        cos_sum = _all_reduce_sum(cos_sum)
        sin_sum = _all_reduce_sum(sin_sum)
        n_total = _all_reduce_sum(n_total)

    ecf_cos = cos_sum / n_total
    ecf_sin = sin_sum / n_total

    err = (ecf_cos - phi).square() + ecf_sin.square()
    statistic = (err @ weights) * n_total
    return statistic.mean()


class LossLatentSSLLeJEPA(LossLatentSSLStudentTeacher):
    """Temporal JEPA + SIGReg.

    Config (under the loss term, alongside ``loss_fcts``)::

        sigreg:
          weight: 0.2               # lambda; total = JEPA + weight * SIGReg
          aggregation: cell         # "cell" (default) or "pooled" (upstream behaviour)
          source: latent_state      # "latent_state" (raw encoder) or a loss_fcts head name
          num_projections: 256
          knots: 17
          t_max: 3.0
          num_cells: 256            # cells subsampled per step when aggregation=cell
          max_dims: 0               # 0 = use all latent dims; else subsample this many
          min_samples: 4            # skip SIGReg if the sample axis is smaller than this
          warmup_steps: 0           # linear ramp of the weight over this many steps
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **losses):
        super().__init__(cf, mode_cfg, stage, device, **losses)
        self.name = "LossLatentSSLLeJEPA"

        sr = mode_cfg.get("sigreg", None) or {}
        self.sigreg_weight = float(sr.get("weight", 0.2))
        self.sigreg_aggregation = str(sr.get("aggregation", "cell"))
        self.sigreg_source = str(sr.get("source", "latent_state"))
        self.sigreg_num_projections = int(sr.get("num_projections", 256))
        self.sigreg_knots = int(sr.get("knots", 17))
        self.sigreg_t_max = float(sr.get("t_max", 3.0))
        self.sigreg_num_cells = int(sr.get("num_cells", 256))
        self.sigreg_max_dims = int(sr.get("max_dims", 0))
        self.sigreg_min_samples = int(sr.get("min_samples", 4))
        self.sigreg_warmup_steps = int(sr.get("warmup_steps", 0))

        assert self.sigreg_aggregation in ("cell", "pooled"), (
            f"Unknown sigreg.aggregation {self.sigreg_aggregation!r}; use 'cell' or 'pooled'."
        )

        # Incremented once per compute_loss call. Every rank calls compute_loss the same
        # number of times, so this stays in lockstep and can seed identical projection
        # directions on all ranks -- required for the cross-rank ECF reduction to be
        # comparing like with like.
        self._sigreg_step = 0

    def _select_representation(self, preds) -> torch.Tensor | None:
        """Return the student tensor to regularize, shaped [views, cells, dim]."""
        latent = preds.latent[0]
        if self.sigreg_source == "latent_state":
            state = latent.get("latent_state", None)
            return None if state is None else state.patch_tokens
        return latent.get(self.sigreg_source, None)

    def _draw_projections(self, dim: int, device, dtype) -> torch.Tensor:
        """Rank-synchronised random unit directions, [dim, num_projections]."""
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0x5157 * 1_000_003 + self._sigreg_step)
        a = torch.randn(dim, self.sigreg_num_projections, generator=gen, dtype=torch.float32)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        return a.to(device=device, dtype=dtype)

    def _compute_sigreg(self, z: torch.Tensor) -> torch.Tensor | None:
        """SIGReg on student latents ``z`` of shape [views, cells, dim]."""
        if z.dim() != 3:
            logger.warning(
                f"SIGReg expects a [views, cells, dim] tensor, got shape {tuple(z.shape)}; "
                "skipping the regularizer this step."
            )
            return None

        z = z.float()
        views, cells, dim = z.shape

        # Optionally subsample latent dimensions to bound cost.
        if 0 < self.sigreg_max_dims < dim:
            idx = torch.randperm(dim, device=z.device)[: self.sigreg_max_dims]
            z = z.index_select(-1, idx)
            dim = self.sigreg_max_dims

        a = self._draw_projections(dim, z.device, z.dtype)

        if self.sigreg_aggregation == "pooled":
            # Upstream behaviour: every (view, cell) pair is a sample.
            proj = (z.reshape(-1, dim) @ a).unsqueeze(0)  # [1, N, P]
            n_local = proj.shape[1]
            if n_local < self.sigreg_min_samples:
                return None
            # Already many samples per rank; cross-rank reduction still sharpens the test.
            return sigreg_from_projections(
                proj.squeeze(0).unsqueeze(0),
                sample_dims=(1,),
                knots=self.sigreg_knots,
                t_max=self.sigreg_t_max,
                cross_rank=True,
            )

        # aggregation == "cell": weather states are the sample axis, one test per cell.
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        if views * world_size < self.sigreg_min_samples:
            logger.warning(
                f"SIGReg aggregation='cell' has only {views * world_size} samples "
                f"(views={views} x world_size={world_size}), below min_samples="
                f"{self.sigreg_min_samples}. Increase training_config.model_input "
                "num_samples or use aggregation='pooled'."
            )
            return None

        if 0 < self.sigreg_num_cells < cells:
            # Same cells on every rank, so the reduced ECF is per-cell coherent.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(0x5157 * 7_919 + self._sigreg_step)
            cell_idx = torch.randperm(cells, generator=gen)[: self.sigreg_num_cells]
            z = z.index_select(1, cell_idx.to(z.device))

        proj = z @ a  # [views, cells_sub, P]
        return sigreg_from_projections(
            proj,
            sample_dims=(0,),  # views are the sample axis; cells/projections averaged later
            knots=self.sigreg_knots,
            t_max=self.sigreg_t_max,
            cross_rank=True,
        )

    def compute_loss(self, preds, targets, metadata, istep=0, **kwargs) -> LossValues:
        base = super().compute_loss(preds, targets, metadata, istep=istep, **kwargs)

        loss = base.loss
        losses_all = dict(base.losses_all)

        warmup = (
            min(1.0, self._sigreg_step / self.sigreg_warmup_steps)
            if self.sigreg_warmup_steps > 0
            else 1.0
        )
        self._sigreg_step += 1

        z = self._select_representation(preds)
        if z is None:
            logger.warning(
                f"SIGReg source {self.sigreg_source!r} not found in latent predictions "
                f"({sorted(preds.latent[0].keys())}); skipping the regularizer."
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
