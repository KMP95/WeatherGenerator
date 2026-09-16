# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Teacher-free JEPA targets: the live encoder, differentiable, no EMA, no stop-gradient.

WHY THIS EXISTS
---------------
JEPA prevents representational collapse with an *architectural* asymmetry: the target is
produced by an EMA copy of the student under ``torch.no_grad()``. LeJEPA / LeWorldModel
make a different bet -- SIGReg constrains the embedding distribution towards an isotropic
Gaussian, a constant cannot be isotropic Gaussian, so collapse is excluded by the
*objective* and the teacher network, the EMA and the stop-gradient can all go
(LeWorldModel, Sec. 3.1: "We do not employ stop-gradient, exponential moving averages, or
additional stabilization heuristics. Gradients are propagated through all components").

This module is the WeatherGenerator version of that recipe, and it is the experiment the
campaign builds towards: remove the teacher, keep everything else, and see whether SIGReg
alone holds the representation up.

WHAT THE TARGET IS -- AND THE BUG THE FIRST VERSION HAD
--------------------------------------------------------
The EMA teacher is *stripped* by ``prepare_encoder_teacher``: no forecast engine, no
deep-SSL fusion, no per-level projections, and ``latent_heads[JEPA] = Identity``. Its
targets are therefore the RAW per-level encoder tokens of the t+6h state.

The first version of this class ran the *unstripped* live model instead. Its "targets" went
through the student path -- deep-SSL fusion, the 6-block transformer predictor, the
per-level projections -- so the loss became

    L1( proj_i(P(F(x_t))),  proj_i(P(F(x_{t+6h}))) )

with the same trainable predictor P on both sides and SIGReg constraining only F. P learned
to discard its input, all four depth taps went to ~1e-8 together while the encoder stayed
perfectly isotropic, and the run taught nothing about forecasting (run ``kmhysfor``,
cancelled 2026-09-16 after 21 h). That is not the collapse LeJEPA is about; it is the
predictor being on both sides of the equation.

This version rebuilds the stripped-teacher path on the live weights, with gradients:

    target_i = F_level_i(x_{t+6h})          raw encoder tokens, per deep-SSL level
    student_i = proj_i(P(fuse(F(x_t))))      unchanged, upstream code

so the prediction task is exactly the EMA teacher's task, minus the EMA and the no_grad.

WHAT STAYS THE SAME
-------------------
Prediction loss, predictor head, views, masks. The only change is where the target comes
from and whether it carries gradient -- a single-variable comparison against the EMA arms.

Note this is NOT the paper's prediction loss. LeJEPA's is an *invariance* term (every view
predicts the mean of the global views), right when views are augmentations of one image.
Our views are t and t+6h -- different times, not augmentations -- and forcing them to be
invariant would erase the evolution a forecast depends on. LeWorldModel faces the same
situation (temporal views, a predictor) and keeps the asymmetric next-embedding prediction
loss, as here.

WHAT TO WATCH
-------------
With no stop-gradient the target branch can be dragged towards whatever is easiest to
predict. Two instruments:
  * ``collapse.*.batch_to_cell_var_ratio`` -> 0: every cell constant across weather states.
  * the encoder ignoring the dynamic fields and encoding only what t and t+6h share
    (orography, land-sea mask, date features). That passes SIGReg and zeroes the
    prediction loss with a non-collapsed batch. A temporal-contrast monitor
    ||F(x_t) - F(x_{t+6h})|| relative to the across-batch spread is the check for it.

Usage::

    training_config:
      losses:
        student-teacher:
          type: LossLatentSSLLeJEPA
          target_and_aux_calc: { SharedEncoderTarget: { target_eval_mode: false } }
      sigreg:
        weight: 0.2          # must be > 0: it is the only thing preventing collapse
"""

from __future__ import annotations

import logging

import torch

from weathergen.model.model import ModelOutput
from weathergen.train.target_and_aux_module_base import TargetAuxOutput
from weathergen.train.target_and_aux_ssl_teacher import EncoderTeacher

logger = logging.getLogger(__name__)


class SharedEncoderTarget(EncoderTeacher):
    """Targets from the live encoder, differentiable, with no teacher of any kind."""

    def __init__(self, model, training_cfg, sigreg_cfg=None, target_eval_mode=False, **kwargs):
        super().__init__(model, training_cfg, **kwargs)
        self.target_eval_mode = bool(target_eval_mode)

        # SIGReg is the only thing standing between this configuration and a constant
        # representation, so refuse to start if it is switched off. Failing here costs a
        # few seconds; failing silently costs the length of the run.
        weight = None if sigreg_cfg is None else sigreg_cfg.get("weight", None)
        if weight is not None and float(weight) <= 0.0:
            raise ValueError(
                "SharedEncoderTarget removes the EMA teacher, so the prediction loss alone "
                f"has a trivial constant optimum. sigreg.weight is {weight}; it must be > 0."
            )

        for name in self._unwrapped().latent_heads:
            if name != "JEPA":
                raise NotImplementedError(
                    f"SharedEncoderTarget supports the JEPA head only (identity target); "
                    f"got {name!r}. iBOT/DINO need real teacher heads."
                )

        logger.info(
            "SharedEncoderTarget: teacher-free targets = raw per-level encoder tokens of the "
            "live model, gradients enabled on both branches, no forecast engine / fusion / "
            "predictor on the target side. Collapse is prevented by SIGReg alone."
        )

    # ------------------------------------------------------------------ helpers

    def _unwrapped(self) -> torch.nn.Module:
        # `model` arrives DDP-wrapped (model_interface.init_model_and_shard). Calling
        # sub-modules directly needs the bare module; DDP's gradient hooks are registered
        # on the parameters and still fire, and every parameter used here is also used by
        # the student forward, so nothing is left "unused" for find_unused_parameters.
        m = self.teacher_model
        return m.module if hasattr(m, "module") else m

    def forward_teacher(self, model_params, batch) -> ModelOutput:
        """Encoder-only forward mirroring what ``prepare_encoder_teacher`` leaves standing.

        Reproduces ``Model.forward`` up to the encoder, then the *teacher* branch of
        ``Model.predict_latent`` (identity head, independent per-level targets). Deliberately
        skipped, because the stripped EMA teacher does not have them: the forecast engine,
        ``deep_ssl_fusion``, ``deep_ssl_level_projections``, the predictor head, decoders.
        """
        m = self._unwrapped()
        was_training = m.training
        if self.target_eval_mode and was_training:
            m.eval()
        try:
            output = ModelOutput(batch.get_output_len())

            tokens, _posteriors, intermediates = m.encoder(model_params, batch)

            # recover batch dimension and collapse the input-step axis, as Model.forward does
            shape = (len(batch), batch.get_num_steps(), *tokens.shape[1:])
            tokens = tokens.reshape(shape).sum(axis=1)
            intermediates = [inter.reshape(shape).sum(axis=1) for inter in intermediates]

            step = 0  # SSL targets are consumed at forecast step 0 only
            tokens_post_norm = m.latent_pre_norm(tokens)
            latent_state = m.tokens_to_latent_state(tokens_post_norm, tokens)
            output.add_latent_prediction(step, "latent_state", latent_state)
            for name in m.latent_heads:
                # identity head: the target is the raw patch tokens
                output.add_latent_prediction(step, name, latent_state.patch_tokens)

            if intermediates:
                # teacher path of predict_latent: one raw target per deep-SSL level
                all_levels = intermediates + [tokens]
                per_head: dict[str, list[torch.Tensor]] = {name: [] for name in m.latent_heads}
                for level_tokens in all_levels:
                    level_post_norm = m.latent_pre_norm(level_tokens)
                    level_state = m.tokens_to_latent_state(level_post_norm, level_tokens)
                    for name in m.latent_heads:
                        per_head[name].append(level_state.patch_tokens)
                for name, preds in per_head.items():
                    output.add_deep_latent_prediction(step, name, preds)
            return output
        finally:
            if self.target_eval_mode and was_training:
                m.train()

    def compute(self, bidx, batch, model_params, model) -> TargetAuxOutput:
        """Same body as :meth:`EncoderTeacher.compute`, without ``torch.no_grad()``.

        The base class cannot be reused: its ``no_grad`` context is the thing this module
        exists to remove, and it wraps the call to ``forward_teacher`` rather than sitting
        inside it.
        """
        teacher_output = self.forward_teacher(model_params, batch)
        outputs = teacher_output.get_latent_prediction(0)

        targets = {}
        for loss_name, target_module in self.postprocess_targets.items():
            targets[loss_name] = target_module(outputs[loss_name])

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
