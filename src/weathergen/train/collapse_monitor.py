# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Collapse monitoring metrics for SSL training (JEPA/DINO).

This module implements metrics to detect representation collapse during self-supervised learning:
- Effective Rank (RankMe): Entropy of normalized singular values
- Singular Value Spectrum: Top-k singular values and concentration ratio
- Per-Dimension Variance: Min/mean/max variance across embedding dimensions
- Per-Cell Across-Batch Variance: variance over weather states, held per HEALPix cell
- Prototype Entropy: Normalized entropy of DINO prototype assignments
- EMA Beta: Current teacher momentum value

For forecasting, the monitor supports sequences of latents (one per time step) and computes:
- Per-step metrics (e.g., effective_rank.step_0, step_1, ...)
- Aggregate metrics (mean, min across steps)
- Degradation ratio (final step / initial step)

References:
- RankMe (ICML 2023): https://arxiv.org/abs/2210.02885
- C-JEPA (NeurIPS 2024): https://arxiv.org/abs/2410.19560
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch

from weathergen.model.engines import LatentState
from weathergen.train.target_and_aux_ssl_teacher import EMATeacher

logger = logging.getLogger(__name__)

# Valid values for tensor_source config option
VALID_TENSOR_SOURCES = frozenset({"student", "teacher", "both"})

# Valid values for forecast_aggregation config option
VALID_FORECAST_AGGREGATIONS = frozenset({"all", "aggregate_only", "per_step_only"})


class CollapseMonitor:
    """
    Monitor for detecting representation collapse during SSL training.

    Computes and caches various collapse indicators that can be logged
    at configurable intervals to minimize computational overhead.

    Supports both single latent tensors and sequences of latents for forecasting.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        """
        Initialize the collapse monitor.

        Args:
            config: Configuration dictionary with collapse_monitoring settings.
            device: Device to use for computations (currently unused, tensors
                    are processed on their original device).

        Raises:
            ValueError: If config contains invalid values.
        """
        self.enabled = config.get("enabled", False)
        self.compute_frequency = config.get("compute_frequency", 100)
        self.log_frequency = config.get("log_frequency", 100)

        # Validate frequencies
        if self.compute_frequency <= 0:
            raise ValueError(f"compute_frequency must be positive, got {self.compute_frequency}")
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be positive, got {self.log_frequency}")

        # Metric configurations
        metrics_config = config.get("metrics", {})

        self.effective_rank_config = metrics_config.get("effective_rank", {})
        self.singular_values_config = metrics_config.get("singular_values", {})
        self.dimension_variance_config = metrics_config.get("dimension_variance", {})
        self.cell_batch_variance_config = metrics_config.get("cell_batch_variance", {})
        # Temporal contrast between the two branches' encoder outputs (needs
        # preds.latent[0]["encoder_tokens"] and a target-branch tensor). Enabled by default.
        self.temporal_contrast_config = metrics_config.get("temporal_contrast", {})
        # Per-level (deep-SSL tap) target metrics. Off by default: one gather per level.
        self.deep_levels_config = metrics_config.get("deep_levels", {"enabled": False})
        self.prototype_entropy_config = metrics_config.get("prototype_entropy", {})
        self.ema_beta_config = metrics_config.get("ema_beta", {})

        # Validate tensor_source values
        self._validate_tensor_source(self.effective_rank_config, "effective_rank")
        self._validate_tensor_source(self.singular_values_config, "singular_values")
        self._validate_tensor_source(self.dimension_variance_config, "dimension_variance")
        self._validate_tensor_source(self.cell_batch_variance_config, "cell_batch_variance")

        # Validate forecast_aggregation values
        self._validate_forecast_aggregation(self.effective_rank_config, "effective_rank")
        self._validate_forecast_aggregation(self.singular_values_config, "singular_values")
        self._validate_forecast_aggregation(self.dimension_variance_config, "dimension_variance")
        self._validate_forecast_aggregation(self.cell_batch_variance_config, "cell_batch_variance")

        # Cache for accumulating metrics between log intervals
        self._metrics_cache: defaultdict[str, list[float]] = defaultdict(list)

    def _validate_tensor_source(self, metric_config: dict[str, Any], metric_name: str) -> None:
        """Validate tensor_source config value."""
        source = metric_config.get("tensor_source", "both")
        if source not in VALID_TENSOR_SOURCES:
            raise ValueError(
                f"Invalid tensor_source '{source}' for {metric_name}. "
                f"Must be one of: {sorted(VALID_TENSOR_SOURCES)}"
            )

    def _validate_forecast_aggregation(
        self, metric_config: dict[str, Any], metric_name: str
    ) -> None:
        """Validate forecast_aggregation config value."""
        aggregation = metric_config.get("forecast_aggregation", "all")
        if aggregation not in VALID_FORECAST_AGGREGATIONS:
            raise ValueError(
                f"Invalid forecast_aggregation '{aggregation}' for {metric_name}. "
                f"Must be one of: {sorted(VALID_FORECAST_AGGREGATIONS)}"
            )

    def should_compute(self, step: int) -> bool:
        """Check if metrics should be computed at this step."""
        return self.enabled and step % self.compute_frequency == 0

    def should_log(self, step: int) -> bool:
        """Check if metrics should be logged at this step."""
        return self.enabled and step % self.log_frequency == 0

    def _get_tensors_to_monitor(
        self,
        student_latent: torch.Tensor | list[torch.Tensor] | None,
        teacher_latent: torch.Tensor | list[torch.Tensor] | None,
        metric_config: dict[str, Any],
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        """
        Get tensors to monitor based on metric config's tensor_source.

        Args:
            student_latent: Student latent(s).
            teacher_latent: Teacher latent(s).
            metric_config: Config dict with tensor_source key.

        Returns:
            Dict mapping "student"/"teacher" to their tensors (if requested).
        """
        source = metric_config.get("tensor_source", "both")
        result: dict[str, torch.Tensor | list[torch.Tensor] | None] = {}

        if source in ("student", "both"):
            result["student"] = student_latent
        if source in ("teacher", "both"):
            result["teacher"] = teacher_latent

        return result

    def _compute_sequence_metrics(
        self,
        latents: list[torch.Tensor],
        compute_fn: Callable[..., float],
        metric_name: str,
        aggregation: str,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute metrics for a sequence of latents (forecasting).

        Args:
            latents: List of latent tensors, one per time step.
            compute_fn: Function to compute metric for a single tensor.
            metric_name: Base name for the metric (e.g., "effective_rank").
            aggregation: One of "all", "aggregate_only", "per_step_only".
            **kwargs: Additional arguments passed to compute_fn.

        Returns:
            Dictionary of metrics with per-step and/or aggregate values.
        """
        metrics: dict[str, float] = {}

        if not latents:
            return metrics

        # Compute per-step metrics
        step_values: list[float] = []
        for step_idx, latent in enumerate(latents):
            value = compute_fn(latent, **kwargs)
            step_values.append(value)

            if aggregation in ("all", "per_step_only"):
                metrics[f"{metric_name}.step_{step_idx}"] = value

        # Compute aggregate metrics
        if aggregation in ("all", "aggregate_only") and step_values:
            # Filter out invalid values (0.0 indicates computation failure)
            valid_values = [v for v in step_values if v > 0]

            if valid_values:
                metrics[f"{metric_name}.mean"] = sum(valid_values) / len(valid_values)
                metrics[f"{metric_name}.min"] = min(valid_values)
                metrics[f"{metric_name}.max"] = max(valid_values)

                # Degradation: ratio of last step to first step
                # Values > 1 mean rank increased, < 1 means degradation
                if step_values[0] > 0 and step_values[-1] > 0:
                    metrics[f"{metric_name}.degradation"] = step_values[-1] / step_values[0]

        return metrics

    def _compute_sequence_dict_metrics(
        self,
        latents: list[torch.Tensor],
        compute_fn: Callable[..., dict[str, float]],
        base_prefix: str,
        aggregation: str,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Compute dict-returning metrics for a sequence of latents.

        For metrics like singular_values that return multiple values per tensor.

        Args:
            latents: List of latent tensors.
            compute_fn: Function returning dict of metrics for single tensor.
            base_prefix: Prefix for metric names (e.g., "collapse.student").
            aggregation: One of "all", "aggregate_only", "per_step_only".
            **kwargs: Additional arguments passed to compute_fn.

        Returns:
            Dictionary of metrics.
        """
        metrics: dict[str, float] = {}

        if not latents:
            return metrics

        # Collect per-step values for each sub-metric
        step_metrics: dict[str, list[float]] = defaultdict(list)

        for step_idx, latent in enumerate(latents):
            step_result = compute_fn(latent, **kwargs)

            for key, value in step_result.items():
                step_metrics[key].append(value)

                if aggregation in ("all", "per_step_only"):
                    metrics[f"{base_prefix}.{key}.step_{step_idx}"] = value

        # Compute aggregates for each sub-metric
        if aggregation in ("all", "aggregate_only"):
            for key, values in step_metrics.items():
                valid_values = [v for v in values if v > 0 or key.startswith("var_")]
                if valid_values:
                    metrics[f"{base_prefix}.{key}.mean"] = sum(valid_values) / len(valid_values)
                    metrics[f"{base_prefix}.{key}.min"] = min(valid_values)
                    metrics[f"{base_prefix}.{key}.max"] = max(valid_values)

        return metrics

    def compute_metrics(
        self,
        student_latent: torch.Tensor | list[torch.Tensor] | None = None,
        teacher_latent: torch.Tensor | list[torch.Tensor] | None = None,
        prototype_probs: torch.Tensor | None = None,
        ema_beta: float | None = None,
        loss_type: str | None = None,
        encoder_tokens_student: torch.Tensor | None = None,
        encoder_tokens_target: torch.Tensor | None = None,
        deep_targets: list[torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """
        Compute all enabled collapse monitoring metrics.

        Supports both single tensors and sequences of tensors (for forecasting).
        For sequences, computes per-step metrics and aggregates based on config.

        Args:
            student_latent: Student model latent representations.
                Single tensor [B, N, D] or [B, D], or list of tensors for forecasting.
            teacher_latent: Teacher model latent representations.
                Single tensor [B, N, D] or [B, D], or list of tensors for forecasting.
            prototype_probs: Post-softmax prototype assignment probabilities [B, K] (DINO only).
            ema_beta: Current EMA momentum value.
            loss_type: Type of SSL loss ("JEPA" or "DINO").

        Returns:
            Dictionary of computed metrics.
        """
        if not self.enabled:
            return {}

        metrics: dict[str, float] = {}

        # Compute effective rank
        if self.effective_rank_config.get("enabled", True):
            sample_size = self.effective_rank_config.get("sample_size", 2048)
            aggregation = self.effective_rank_config.get("forecast_aggregation", "all")
            tensors = self._get_tensors_to_monitor(
                student_latent, teacher_latent, self.effective_rank_config
            )

            for name, tensor in tensors.items():
                if tensor is None:
                    continue

                if isinstance(tensor, list):
                    seq_metrics = self._compute_sequence_metrics(
                        tensor,
                        self._compute_effective_rank,
                        f"collapse.{name}.effective_rank",
                        aggregation,
                        sample_size=sample_size,
                    )
                    metrics.update(seq_metrics)
                else:
                    eff_rank = self._compute_effective_rank(tensor, sample_size)
                    metrics[f"collapse.{name}.effective_rank"] = eff_rank

        # Compute singular value spectrum
        if self.singular_values_config.get("enabled", True):
            sample_size = self.singular_values_config.get("sample_size", 2048)
            aggregation = self.singular_values_config.get("forecast_aggregation", "all")
            tensors = self._get_tensors_to_monitor(
                student_latent, teacher_latent, self.singular_values_config
            )

            for name, tensor in tensors.items():
                if tensor is None:
                    continue

                if isinstance(tensor, list):
                    seq_metrics = self._compute_sequence_dict_metrics(
                        tensor,
                        self._compute_singular_values,
                        f"collapse.{name}",
                        aggregation,
                        sample_size=sample_size,
                    )
                    metrics.update(seq_metrics)
                else:
                    sv_metrics = self._compute_singular_values(tensor, sample_size)
                    for key, value in sv_metrics.items():
                        metrics[f"collapse.{name}.{key}"] = value

        # Compute per-dimension variance
        if self.dimension_variance_config.get("enabled", True):
            aggregation = self.dimension_variance_config.get("forecast_aggregation", "all")
            tensors = self._get_tensors_to_monitor(
                student_latent, teacher_latent, self.dimension_variance_config
            )

            for name, tensor in tensors.items():
                if tensor is None:
                    continue

                if isinstance(tensor, list):
                    seq_metrics = self._compute_sequence_dict_metrics(
                        tensor,
                        self._compute_dimension_variance,
                        f"collapse.{name}",
                        aggregation,
                    )
                    metrics.update(seq_metrics)
                else:
                    var_metrics = self._compute_dimension_variance(tensor)
                    for key, value in var_metrics.items():
                        metrics[f"collapse.{name}.{key}"] = value

        # Compute per-cell across-batch variance
        if self.cell_batch_variance_config.get("enabled", True):
            aggregation = self.cell_batch_variance_config.get("forecast_aggregation", "all")
            tensors = self._get_tensors_to_monitor(
                student_latent, teacher_latent, self.cell_batch_variance_config
            )

            for name, tensor in tensors.items():
                if tensor is None:
                    continue

                if isinstance(tensor, list):
                    seq_metrics = self._compute_sequence_dict_metrics(
                        tensor,
                        self._compute_cell_batch_variance,
                        f"collapse.{name}",
                        aggregation,
                    )
                    metrics.update(seq_metrics)
                else:
                    cbv_metrics = self._compute_cell_batch_variance(tensor)
                    for key, value in cbv_metrics.items():
                        metrics[f"collapse.{name}.{key}"] = value

        # Compute prototype entropy (DINO only)
        if (
            self.prototype_entropy_config.get("enabled", True)
            and prototype_probs is not None
            and loss_type == "DINO"
        ):
            entropy = self._compute_prototype_entropy(prototype_probs)
            metrics["collapse.dino.prototype_entropy"] = entropy

        # Temporal contrast between the context and target encodings of one trajectory
        if (
            self.temporal_contrast_config.get("enabled", True)
            and encoder_tokens_student is not None
            and encoder_tokens_target is not None
        ):
            for key, value in self._compute_temporal_contrast(
                encoder_tokens_student, encoder_tokens_target
            ).items():
                metrics[f"collapse.{key}"] = value

        # Per-level target metrics (deep-SSL taps): the final-level target is what
        # `teacher` reports; the intermediate levels are only visible here.
        if self.deep_levels_config.get("enabled", False) and deep_targets:
            for i, level in enumerate(deep_targets):
                if not isinstance(level, torch.Tensor):
                    continue
                for key, value in self._compute_cell_batch_variance(level).items():
                    metrics[f"collapse.teacher_L{i}.{key}"] = value
                for key, value in self._compute_dimension_variance(level).items():
                    metrics[f"collapse.teacher_L{i}.{key}"] = value

        # Log EMA beta
        if self.ema_beta_config.get("enabled", True) and ema_beta is not None:
            metrics["collapse.ema_beta"] = ema_beta

        # Cache metrics for averaging
        for key, value in metrics.items():
            self._metrics_cache[key].append(value)

        return metrics

    def get_cached_metrics(self) -> dict[str, float]:
        """
        Get averaged cached metrics and clear the cache.

        Returns:
            Dictionary of averaged metrics since last call.
        """
        averaged_metrics: dict[str, float] = {}
        for key, values in self._metrics_cache.items():
            if values:
                averaged_metrics[key] = sum(values) / len(values)

        self._metrics_cache.clear()
        return averaged_metrics

    def _flatten_to_samples(self, z: torch.Tensor) -> torch.Tensor:
        """
        Flatten patch dimension into sample dimension.

        Treats [B, N, D] as [B*N, D] where each patch is an independent sample.
        This is consistent with C-JEPA/VICReg approach.

        Args:
            z: Tensor of shape [B, N, D] or [B, D].

        Returns:
            Tensor of shape [B*N, D] or [B, D].
        """
        # Convert to float32 for SVD compatibility (bfloat16/float16 can fail)
        if z.dtype in (torch.bfloat16, torch.float16):
            z = z.float()

        if z.ndim == 3:
            return z.reshape(-1, z.shape[-1])
        return z

    def _sample_rows(self, z: torch.Tensor, sample_size: int) -> torch.Tensor:
        """
        Randomly sample rows to reduce SVD computation cost.

        Args:
            z: Tensor of shape [N, D].
            sample_size: Maximum number of samples (0 = no sampling).

        Returns:
            Sampled tensor of shape [min(N, sample_size), D].
        """
        if sample_size <= 0 or z.shape[0] <= sample_size:
            return z

        indices = torch.randperm(z.shape[0], device=z.device)[:sample_size]
        return z[indices]

    def _compute_effective_rank(self, z: torch.Tensor, sample_size: int = 2048) -> float:
        """
        Compute effective rank via entropy of normalized singular values (RankMe).

        The effective rank measures how many dimensions are actually being used
        in the representation. A low effective rank indicates collapse.

        Args:
            z: Latent representations [B, N, D] or [B, D].
            sample_size: Maximum samples for SVD computation.

        Returns:
            Effective rank (exp of entropy of normalized singular values).
        """
        z = self._flatten_to_samples(z.detach())
        z = self._sample_rows(z, sample_size)

        # Validate tensor before SVD
        if z.numel() == 0:
            logger.warning("Empty tensor in effective rank computation")
            return 0.0
        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.warning("NaN/Inf values in tensor for effective rank computation")
            return 0.0
        if z.shape[0] < 2 or z.shape[1] < 2:
            logger.warning(f"Tensor too small for SVD: shape={z.shape}")
            return 0.0

        # Center the data
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
        except RuntimeError as e:
            # SVD can fail on degenerate matrices
            logger.warning(f"SVD failed in effective rank computation: {e}, shape={z.shape}")
            return 0.0

        # Normalize singular values to get a probability distribution
        s_normalized = s / (s.sum() + 1e-8)

        # Compute entropy
        entropy = -torch.sum(s_normalized * torch.log(s_normalized + 1e-8))

        # Effective rank is exp(entropy)
        effective_rank = torch.exp(entropy)

        return effective_rank.item()

    def _compute_singular_values(
        self, z: torch.Tensor, sample_size: int = 2048
    ) -> dict[str, float]:
        """
        Compute singular value statistics and concentration ratio.

        The concentration ratio (top SV / sum of all SVs) indicates how much
        variance is captured by the largest singular value. High concentration
        suggests dimensional collapse.

        Args:
            z: Latent representations [B, N, D] or [B, D].
            sample_size: Maximum samples for SVD computation.

        Returns:
            Dictionary with sv_min, sv_max, sv_mean, and sv_concentration.
        """
        z = self._flatten_to_samples(z.detach())
        z = self._sample_rows(z, sample_size)

        # Validate tensor before SVD
        if z.numel() == 0:
            logger.warning("Empty tensor in singular value computation")
            return {}
        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.warning("NaN/Inf values in tensor for singular value computation")
            return {}
        if z.shape[0] < 2 or z.shape[1] < 2:
            logger.warning(f"Tensor too small for SVD: shape={z.shape}")
            return {}

        # Center the data
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            _, s, _ = torch.linalg.svd(z_centered, full_matrices=False)
        except RuntimeError as e:
            logger.warning(f"SVD failed in singular value computation: {e}, shape={z.shape}")
            return {}

        metrics: dict[str, float] = {}

        # Singular value statistics
        metrics["sv_min"] = s.min().item()
        metrics["sv_max"] = s.max().item()
        metrics["sv_mean"] = s.mean().item()

        # Concentration ratio (top SV / sum)
        s_sum = s.sum() + 1e-8
        metrics["sv_concentration"] = (s[0] / s_sum).item()

        return metrics

    def _compute_dimension_variance(self, z: torch.Tensor) -> dict[str, float]:
        """
        Compute per-dimension variance statistics.

        Low minimum variance indicates "dead" dimensions that are not being used.
        Large variance ratio (max/min) suggests imbalanced dimension usage.

        Args:
            z: Latent representations [B, N, D] or [B, D].

        Returns:
            Dictionary with var_min, var_mean, var_max. Empty dict if tensor is invalid.
        """
        z = self._flatten_to_samples(z.detach())

        # Validate tensor
        if z.numel() == 0:
            logger.warning("Empty tensor in dimension variance computation")
            return {}
        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.warning("NaN/Inf values in tensor for dimension variance computation")
            return {}
        if z.shape[0] < 2:
            logger.warning(f"Need at least 2 samples to compute variance: shape={z.shape}")
            return {}

        # Compute variance along sample dimension
        var_per_dim = z.var(dim=0)

        return {
            "var_min": var_per_dim.min().item(),
            "var_mean": var_per_dim.mean().item(),
            "var_max": var_per_dim.max().item(),
        }

    def _gather_batch_across_ranks(self, z: torch.Tensor) -> torch.Tensor:
        """
        Concatenate the batch dimension of `z` across all ranks.

        Necessary because the batch is one weather state per GPU in the shipped configs
        (`model_input.*.num_samples: 1`), and a variance over a single sample is
        undefined. After gathering, the sample count is world_size * num_samples.

        Every rank runs this: `_compute_collapse_metrics` is called inside the training
        loop with no is_root guard, and `should_compute` keys off the synchronised istep,
        so the collective cannot deadlock. Gradients are irrelevant here, so the plain
        (non-differentiable) all_gather is the right one.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return z
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return z
        z = z.contiguous()
        buf = [torch.empty_like(z) for _ in range(world_size)]
        torch.distributed.all_gather(buf, z)
        return torch.cat(buf, dim=0)

    def _compute_cell_batch_variance(self, z: torch.Tensor) -> dict[str, float]:
        """
        Variance across weather states, computed separately for each HEALPix cell.

        This is the diagnostic `dimension_variance` cannot give. That metric flattens
        [B, N, D] to [B*N, D] first, so it reports variance across cells AND batch mixed
        together. A representation that is identical for every weather state but varies
        over the globe -- the collapse mode that matters when SIGReg is applied along the
        wrong axis -- still shows a large variance there and looks healthy.

        Here the reduction is over B only, holding the cell fixed, which is the same axis
        the per-cell SIGReg term regularises.

        `batch_to_cell_var_ratio` is the headline number: across-batch variance divided by
        across-cell variance. It falls towards 0 exactly in the failure mode above, and
        stays O(1) for a representation that genuinely distinguishes weather states.

        Args:
            z: Latent representations [B, N, D]. A 2D tensor has no cell axis and is
               skipped -- use dimension_variance for that case.

        Returns:
            Dictionary with the per-cell variance summary, or {} if not computable.
        """
        z = z.detach()
        if z.dtype in (torch.bfloat16, torch.float16):
            z = z.float()

        if z.ndim != 3:
            return {}
        if z.numel() == 0:
            logger.warning("Empty tensor in cell-batch variance computation")
            return {}
        if torch.isnan(z).any() or torch.isinf(z).any():
            logger.warning("NaN/Inf values in tensor for cell-batch variance computation")
            return {}

        z = self._gather_batch_across_ranks(z)
        num_states = z.shape[0]
        if num_states < 2:
            logger.warning(
                "Need at least 2 weather states for cell-batch variance, got "
                f"{num_states}. Raise model_input.*.num_samples or use more ranks."
            )
            return {}

        # Across-batch variance, per cell and per dimension -> [N, D].
        var_batch = z.var(dim=0, unbiased=True)
        # Across-cell variance of the per-cell mean state -> [D]. Reference scale.
        var_cells = z.mean(dim=0).var(dim=0, unbiased=True)

        var_batch_per_cell = var_batch.mean(dim=-1)  # [N]
        mean_batch = var_batch.mean()
        mean_cells = var_cells.mean()
        # Across-cell variance of each *individual* state, averaged over states -> the
        # spatial variance a state actually carries. Unlike `var_cells` (which is computed
        # on the batch-mean state and is driven to sigma^2/N by any per-cell centring, e.g.
        # per-cell SIGReg), this denominator does not depend on N and is the right
        # reference for the across-state variance: ratio ~1 healthy, >>1 every state is a
        # near-constant field (global-code collapse), ~0 states are alike (frozen
        # climatology).
        var_spatial = z.var(dim=1, unbiased=True).mean()

        return {
            "cell_batch_var_mean": mean_batch.item(),
            "cell_batch_var_min": var_batch_per_cell.min().item(),
            "cell_batch_var_max": var_batch_per_cell.max().item(),
            "cell_var_mean": mean_cells.item(),
            "batch_to_cell_var_ratio": (mean_batch / (mean_cells + 1e-12)).item(),
            "state_spatial_var_mean": var_spatial.item(),
            "batch_to_spatial_var_ratio": (mean_batch / (var_spatial + 1e-12)).item(),
            "num_states": float(num_states),
        }

    def _compute_temporal_contrast(
        self, z_t: torch.Tensor, z_next: torch.Tensor
    ) -> dict[str, float]:
        """
        How different the two branches' encodings of one trajectory are, relative to how
        different weather states are from each other.

            rho = mean_{s,c} ||z_t[s,c] - z_next[s,c]||^2  /  mean_c tr Cov_s(z_t[.,c])

        The check for the one collapse a temporal objective invites and SIGReg cannot see:
        the encoder keeping only what t and t+dt share (orography, land-sea mask, the date
        features) and dropping the weather. Then z_t ~ z_next everywhere, the predictor is
        the identity, the prediction loss is ~0 and the across-batch spread is still
        non-degenerate. rho -> 0 is that failure; for a healthy encoder the 6-h change is a
        sizeable fraction of the across-state spread (order 0.1-1).

        Args:
            z_t: context-branch encoder tokens [B, N, D] (same normalised space as z_next).
            z_next: target-branch encoder tokens [B, N, D].
        """
        z_t = z_t.detach()
        z_next = z_next.detach()
        if z_t.dtype in (torch.bfloat16, torch.float16):
            z_t = z_t.float()
        if z_next.dtype in (torch.bfloat16, torch.float16):
            z_next = z_next.float()
        if z_t.ndim != 3 or z_t.shape != z_next.shape or z_t.numel() == 0:
            return {}
        if torch.isnan(z_t).any() or torch.isnan(z_next).any():
            logger.warning("NaN in tensors for temporal contrast computation")
            return {}
        num = (z_t - z_next).square().sum(dim=-1).mean()  # mean over states and cells
        z_all = self._gather_batch_across_ranks(z_t)
        if z_all.shape[0] < 2:
            return {}
        den = z_all.var(dim=0, unbiased=True).sum(dim=-1).mean()  # tr Cov over states, mean over cells
        num_all = self._gather_batch_across_ranks(num.reshape(1)).mean()
        return {
            "temporal_contrast": (num_all / (den + 1e-12)).item(),
            "temporal_dist_sq": num_all.item(),
            "temporal_state_spread": den.item(),
        }

    def _compute_prototype_entropy(self, probs: torch.Tensor) -> float:
        """
        Compute normalized entropy of DINO prototype assignments.

        Low entropy indicates collapse to few prototypes. Entropy is normalized
        to [0, 1] range where 1 means uniform distribution.

        Args:
            probs: Post-softmax prototype assignment probabilities [B, K].

        Returns:
            Normalized entropy in [0, 1].
        """
        probs = probs.detach()

        # Average across batch to get prototype usage distribution
        avg_probs = probs.mean(dim=0)

        # Compute entropy
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        # Normalize by maximum possible entropy (uniform distribution)
        num_prototypes = probs.shape[1]
        max_entropy = torch.log(torch.tensor(float(num_prototypes), device=probs.device))

        normalized_entropy = entropy / (max_entropy + 1e-8)

        return normalized_entropy.item()

    def _compute_collapse_metrics(
        self, cf, batch_size, target_and_aux_calculators, preds, targets_and_auxs
    ) -> None:
        """
        Extract latent tensors from predictions and targets, then compute collapse metrics.

        This method extracts the student and teacher latent representations from the
        model outputs. It supports two modes:

        1. SSL training (JEPA/DINO/iBOT): Extracts latents from SSL-specific keys
        2. Forecasting: Extracts latents from 'latent_state' at each forecast step

        For forecasting, a list of latent tensors is passed to enable per-step metrics.
        """
        student_latent: torch.Tensor | list[torch.Tensor] | None = None
        teacher_latent: torch.Tensor | list[torch.Tensor] | None = None
        prototype_probs = None
        ema_beta = None
        loss_type = None

        # Helper to extract tensor from various latent formats
        def extract_latent_tensor(
            latent_data: torch.Tensor | LatentState | list | dict | None,
        ) -> torch.Tensor | None:
            """Extract tensor from various latent data formats."""
            if latent_data is None:
                return None
            if isinstance(latent_data, torch.Tensor):
                return latent_data
            if isinstance(latent_data, LatentState):
                # Use patch_tokens as the primary latent representation
                # For forecast steps > 0, patch_tokens is None, so fall back to z_pre_norm
                if latent_data.patch_tokens is not None:
                    return latent_data.patch_tokens
                if latent_data.z_pre_norm is not None:
                    # z_pre_norm includes register/class tokens, extract patch tokens only
                    # This assumes the same token layout as patch_tokens
                    return latent_data.z_pre_norm
                return None
            if isinstance(latent_data, list) and len(latent_data) > 0:
                return extract_latent_tensor(latent_data[0])
            if isinstance(latent_data, dict):
                # Try common keys
                for key in ["latent", "patch_tokens", "z_pre_norm"]:
                    if key in latent_data:
                        return extract_latent_tensor(latent_data[key])
            return None

        encoder_tokens_target: torch.Tensor | None = None
        deep_targets: list[torch.Tensor] | None = None
        # Find SSL loss type and extract teacher latents
        for _loss_name, target_aux in targets_and_auxs.items():
            if not hasattr(target_aux, "latent") or not target_aux.latent:
                continue

            # Handle both list[dict] and dict formats
            if isinstance(target_aux.latent, list):
                target_latent_dict = target_aux.latent[0] if target_aux.latent else {}
            else:
                target_latent_dict = target_aux.latent

            # Try SSL-specific keys first
            for ssl_type in ["JEPA", "DINO", "iBOT"]:
                if ssl_type in target_latent_dict:
                    loss_type = ssl_type
                    teacher_latent = extract_latent_tensor(target_latent_dict[ssl_type])
                    break
            # Target-branch encoder tokens: published explicitly by SharedEncoderTarget;
            # for an EMA teacher the identity JEPA target *is* the encoder output.
            if "encoder_tokens" in target_latent_dict:
                encoder_tokens_target = extract_latent_tensor(target_latent_dict["encoder_tokens"])
            elif loss_type == "JEPA" and teacher_latent is not None:
                encoder_tokens_target = teacher_latent
            deep = getattr(target_aux, "latent_deep", None)
            if isinstance(deep, dict) and "JEPA" in deep:
                deep_targets = [extract_latent_tensor(t) for t in deep["JEPA"]]

        encoder_tokens_student: torch.Tensor | None = None
        if preds.latent and len(preds.latent) > 0 and "encoder_tokens" in preds.latent[0]:
            encoder_tokens_student = extract_latent_tensor(preds.latent[0]["encoder_tokens"])
        # Extract student latents from predictions
        if preds.latent and len(preds.latent) > 0:
            # First, try SSL-specific keys (JEPA/DINO/iBOT) from first step
            pred_latent_dict = preds.latent[0]
            for ssl_type in ["JEPA", "DINO", "iBOT"]:
                if ssl_type in pred_latent_dict:
                    student_latent = extract_latent_tensor(pred_latent_dict[ssl_type])
                    loss_type = ssl_type
                    break

            # With deep SSL enabled the loss never sees preds.latent[0]["JEPA"] (the head
            # applied to the unfused final tokens): it trains the per-level projections of
            # the head applied to the FUSED levels, preds.latent_deep[0]["JEPA"][i]. The
            # unfused tensor is a dead end that shares weights with the trained path and
            # can wander off (E2f-noclock: var_mean 600, rank 25, while JEPA_L0 = 0.05).
            # Monitor the final-level prediction, the tensor the loss actually compares
            # with the target-branch encoder tokens.
            deep = getattr(preds, "latent_deep", None)
            if loss_type == "JEPA" and deep:
                deep0 = deep[0] if isinstance(deep, list) else deep
                if isinstance(deep0, dict) and "JEPA" in deep0 and deep0["JEPA"]:
                    student_latent = extract_latent_tensor(deep0["JEPA"][-1])

            # If no SSL keys found, extract from latent_state for all forecast steps
            if student_latent is None:
                student_latents_list: list[torch.Tensor] = []
                for step_latent_dict in preds.latent:
                    if "latent_state" in step_latent_dict:
                        step_tensor = extract_latent_tensor(step_latent_dict["latent_state"])
                        if step_tensor is not None:
                            student_latents_list.append(step_tensor)

                # Use list if multiple steps, single tensor otherwise
                if len(student_latents_list) > 1:
                    student_latent = student_latents_list
                    n_steps = len(student_latents_list)
                    logger.debug(f"Collapse monitor - forecasting mode: {n_steps} steps")
                elif len(student_latents_list) == 1:
                    student_latent = student_latents_list[0]

        # Get EMA beta from target_and_aux_calculators
        for _calc_name, calculator in target_and_aux_calculators.items():
            if isinstance(calculator, EMATeacher):
                step = batch_size * cf.general.istep
                ema_beta = calculator.get_current_beta(step)
                break

        # Debug logging
        if student_latent is not None:
            if isinstance(student_latent, list):
                shapes = [t.shape for t in student_latent]
                logger.debug(f"Collapse monitor - student (list): {len(shapes)} steps")
            else:
                logger.debug(f"Collapse monitor - student: shape={student_latent.shape}")
        else:
            logger.debug("Collapse monitor - student_latent is None")

        if teacher_latent is not None:
            if isinstance(teacher_latent, list):
                logger.debug(f"Collapse monitor - teacher (list): {len(teacher_latent)} steps")
            else:
                shape = teacher_latent.shape if isinstance(teacher_latent, torch.Tensor) else "N/A"
                logger.debug(f"Collapse monitor - teacher: shape={shape}")

        # Compute metrics if we have valid student latent
        has_valid_latent = student_latent is not None and (
            isinstance(student_latent, torch.Tensor)
            or (isinstance(student_latent, list) and len(student_latent) > 0)
        )

        if has_valid_latent:
            # Prepare teacher latent (must match student format if provided)
            teacher_for_metrics = None
            if teacher_latent is not None:
                is_valid_tensor = isinstance(teacher_latent, torch.Tensor)
                is_valid_list = isinstance(teacher_latent, list) and len(teacher_latent) > 0
                if is_valid_tensor or is_valid_list:
                    teacher_for_metrics = teacher_latent

            self.compute_metrics(
                student_latent=student_latent,
                teacher_latent=teacher_for_metrics,
                prototype_probs=prototype_probs,
                ema_beta=ema_beta,
                loss_type=loss_type,
                encoder_tokens_student=encoder_tokens_student,
                encoder_tokens_target=encoder_tokens_target,
                deep_targets=deep_targets,
            )
        else:
            logger.debug(
                f"Collapse monitor - skipping compute_metrics: "
                f"student_latent is {'None' if student_latent is None else type(student_latent)}"
            )
