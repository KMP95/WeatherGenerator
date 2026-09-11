# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Teacher-free targets for LeJEPA: one encoder, gradients through both branches.

WHY THIS EXISTS
---------------
JEPA prevents representational collapse with an *architectural* asymmetry. The target
branch is an EMA copy of the student, run under ``torch.no_grad()``
(:meth:`EncoderTeacher.compute`), so no gradient can reach it. Without that asymmetry the
objective ``predictor(f(t)) ~= f(t+6h)`` has a trivial optimum -- ``f`` emits a constant,
the predictor learns the identity, the loss is zero and the representation is worthless.

LeJEPA makes a different bet: SIGReg constrains the embedding distribution towards an
isotropic Gaussian, and a constant output cannot be isotropic Gaussian, so collapse is
excluded by the *objective* rather than by the architecture. The paper states this
directly -- "the absence of prototypes, stop-gradients, and teacher-student networks makes
LeJEPA appealing as it only contains one hyperparameter" -- and its reference
implementation exposes it as ``teacher_student=false``.

This module is the WeatherGenerator version of that claim, and it is the experiment the
whole campaign builds towards: **remove the teacher, keep everything else, and see whether
SIGReg alone holds the representation up.**

WHAT IT DOES DIFFERENTLY FROM ``EMATeacher``
--------------------------------------------
1. **The live model produces the target.** No EMA copy, no second set of weights. This
   matches how the model is meant to work: one encoder that maps a weather state at *any*
   time to a representation, with the predictor carrying it forward.
2. **Gradients flow through the target branch.** ``EncoderTeacher.compute`` wraps its body
   in ``torch.no_grad()``, so :meth:`compute` is overridden here rather than just
   :meth:`forward_teacher`. Both branches are differentiated; nothing is detached.
3. **No EMA bookkeeping.** ``update_state_post_opt_step`` and ``reset`` are no-ops, and
   ``get_current_beta`` returns NaN -- there is no teacher momentum to report, so the
   ``collapse.ema_beta`` metric is meaningless for these runs.

WHAT STAYS THE SAME
-------------------
The prediction loss, the predictor head, the views, the masks. The only change is where
the target comes from and whether it carries gradient. That is deliberate: it keeps this a
single-variable comparison against the EMA arms.

Note this is NOT the paper's prediction loss. LeJEPA's is an *invariance* term -- every
view predicts the mean of the global views' embeddings -- which is right when views are
augmentations of one image. Our views are ``t`` and ``t+6h``: different times, not
augmentations, and forcing them to be invariant would erase the evolution a forecast
depends on. The asymmetric prediction loss is kept for exactly that reason.

WHAT TO WATCH
-------------
With no stop-gradient, the target branch can be dragged towards whatever is easiest to
predict. ``collapse.*.batch_to_cell_var_ratio`` is the instrument for it: it goes to zero
precisely when every cell becomes constant across weather states while the globe-shaped
pattern survives -- the failure mode the stock ``dimension_variance`` cannot see.

Usage::

    training_config:
      losses:
        student-teacher:
          type: LossLatentSSLLeJEPA
          target_and_aux_calc: { SharedEncoderTarget: {} }
      sigreg:
        weight: 0.2          # must be > 0: it is the only thing preventing collapse
        aggregation: cell
"""

from __future__ import annotations

import logging

import torch

from weathergen.train.target_and_aux_module_base import TargetAuxOutput
from weathergen.train.target_and_aux_ssl_teacher import EncoderTeacher

logger = logging.getLogger(__name__)


class SharedEncoderTarget(EncoderTeacher):
    """Targets from the live model, differentiable, with no teacher of any kind."""

    def __init__(self, model, training_cfg, sigreg_cfg=None, **kwargs):
        super().__init__(model, training_cfg, **kwargs)

        # SIGReg is the only thing standing between this configuration and a constant
        # representation, so refuse to start if it is switched off. Failing here costs a
        # few seconds; failing silently costs the length of the run.
        weight = None
        if sigreg_cfg is not None:
            weight = sigreg_cfg.get("weight", None)
        if weight is not None and float(weight) <= 0.0:
            raise ValueError(
                "SharedEncoderTarget removes the EMA teacher, so the prediction loss alone "
                f"has a trivial constant optimum. sigreg.weight is {weight}; it must be > 0."
            )

        logger.info(
            "SharedEncoderTarget: teacher-free targets from the live model, gradients "
            "enabled on both branches. Collapse is prevented by SIGReg alone."
        )

    def forward_teacher(self, model_params, batch):
        # self.teacher_model IS the student. Left in train mode on purpose: dropout and
        # any other stochastic layer should behave identically on both branches, otherwise
        # the two differ by more than the time offset.
        return self.teacher_model(model_params, batch)

    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        """Same body as :meth:`EncoderTeacher.compute`, without ``torch.no_grad()``.

        The base class cannot be reused here: its ``no_grad`` context is what this whole
        module exists to remove, and it wraps the call to ``forward_teacher`` rather than
        sitting inside it.
        """
        teacher_output = self.forward_teacher(model_params, batch)
        outputs = teacher_output.get_latent_prediction(0)

        targets = {}
        for loss_name, target_module in self.postprocess_targets.items():
            targets[loss_name] = target_module(outputs[loss_name])

        # deep SSL per-level targets, when the model produced them
        deep_targets = None
        if teacher_output.latent_deep is not None:
            deep_outputs = teacher_output.latent_deep[0]  # fstep 0
            deep_targets = {}
            for loss_name, target_module in self.postprocess_targets.items():
                if loss_name in deep_outputs:
                    deep_targets[loss_name] = [
                        target_module(level_pred, level_idx=i)
                        for i, level_pred in enumerate(deep_outputs[loss_name])
                    ]

        aux_outputs = [list(sample.meta_info.values())[0] for sample in batch.get_samples()]

        targets_out = TargetAuxOutput(batch.get_output_len(), batch.get_output_idxs())
        targets_out.latent = targets
        targets_out.latent_deep = deep_targets
        targets_out.aux_outputs = aux_outputs
        return targets_out

    def reset(self, batch_size=None) -> None:
        return

    def update_state_post_opt_step(self, istep, batch, model, **kwargs) -> None:
        return

    def get_current_beta(self, cur_step: int) -> float:
        # No teacher momentum exists. NaN rather than 0.0 so the logged series is visibly
        # absent instead of looking like a teacher that never moves.
        return float("nan")
