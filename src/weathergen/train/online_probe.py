# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Online linear probes of physical quantities during SSL training.

WHY
---
Representation collapse is not visible in the training loss (MotionJEPA, 2609.23881:
"collapse is not visible in the training loss"), and a loss can keep falling while the
latent drops exactly the features a forecast needs -- the fast ones. The only detector
that catches this during training is a probe of known physical quantities from the frozen
(detached) latent. Three small linear heads are trained alongside the SSL run with their
own optimiser; no gradient reaches the encoder:

  state   : z_t                 -> y_t          what the latent knows about the state at t
  tend    : [z_t, z_{t+dt}]     -> y_{t+dt}-y_t what the pair knows about the 6-h CHANGE
  tend_t  : z_t                 -> y_{t+dt}-y_t how predictable the change is from t alone

y are per-cell means of a few channels (2t, z_500, t_850, ...) of the model's own input
data, in the normalised source space. The reported number is the NMSE on the accumulation
window (MSE / variance of the target): 1.0 = the head knows nothing, 0 = perfect.
Slow-feature collapse shows as `tend` NMSE -> 1 while `state` stays low; a global-code
collapse (E2b) shows as both -> 1.

WHERE THE DATA COMES FROM
-------------------------
The per-cell physical values are computed from the raw input window of each sample
(``StreamData.source_raw``, an ``IOReaderData`` with coords and normalised data). In the
training stage the sampler deletes the raw data to save memory; ``StreamData`` keeps it
when ``train_logging.online_probe.enabled`` is set (~13 MB per sample at o96). Points are
assigned to HEALPix cells exactly as the tokenizer does (nested ordering), so the probe's
cell axis is the latent's cell axis.

Config (under ``train_logging``)::

    online_probe:
      enabled: true
      stream: ERA5_in                    # stream whose raw input is probed (default: first)
      channels: [2t, 10u, 10v, z_500, t_850, u_850, v_850, q_850]
      lr: 1.0e-3
      compute_frequency: 10              # every N optimiser steps (training); every batch in validation
      max_cells: 2048                    # random cell subset per step (cost control)

Metrics: ``probe.<train|val>.<state|tend|tend_t>.<channel>`` and ``...mean``, logged
through the same path as the collapse metrics.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from astropy_healpix.healpy import ang2pix

from weathergen.datasets.tokenizer_utils import theta_phi_to_standard_coords

logger = logging.getLogger(__name__)

_HEADS = ("state", "tend", "tend_t")


class OnlineProbe:
    """Linear probes trained on detached latents; see module docstring."""

    def __init__(self, config: dict[str, Any] | None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        self.stream = config.get("stream", None)
        self.channel_names = list(config.get("channels", ["2t", "10u", "10v", "z_500", "t_850", "u_850", "v_850", "q_850"]))
        self.lr = float(config.get("lr", 1e-3))
        self.compute_frequency = int(config.get("compute_frequency", 10))
        self.max_cells = int(config.get("max_cells", 2048))
        self.healpix_level: int | None = None
        self.device = None
        self.heads: torch.nn.ModuleDict | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._channel_idx: dict[str, list[int]] = {}  # per stage: indices into rdata.data
        self._warned: set[str] = set()
        # accumulators[stage][head][channel] -> [sum_sq_err, sum_y, sum_y2, n]
        self._acc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
        )

    # ------------------------------------------------------------------ setup

    def setup(self, cf, device) -> None:
        """Resolve stream, channel indices and HEALPix level once the config is final."""
        if not self.enabled:
            return
        self.device = device
        self.healpix_level = int(cf.healpix_level)
        streams = list(cf.streams)
        names = [s.get("name") for s in streams]
        if self.stream is None:
            self.stream = names[0]
        if self.stream not in names:
            logger.warning("online_probe: stream %r not in %s; probe disabled.", self.stream, names)
            self.enabled = False
            return
        si = streams[names.index(self.stream)]
        for stage_key, attr in (("train", "train_source_channels"), ("val", "val_source_channels")):
            chans = list(si.get(attr, []) or [])
            idx = []
            kept = []
            for c in self.channel_names:
                if c in chans:
                    idx.append(chans.index(c))
                    kept.append(c)
                else:
                    logger.warning("online_probe: channel %r not among %s source channels of %s; skipped.", c, stage_key, self.stream)
            self._channel_idx[stage_key] = idx
            if stage_key == "train":
                self.channel_names = kept
        if not self._channel_idx.get("train"):
            logger.warning("online_probe: no probe channel found; probe disabled.")
            self.enabled = False

    def _build_heads(self, dim: int) -> None:
        k = len(self.channel_names)
        self.heads = torch.nn.ModuleDict(
            {
                "state": torch.nn.Linear(dim, k),
                "tend": torch.nn.Linear(2 * dim, k),
                "tend_t": torch.nn.Linear(dim, k),
            }
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=self.lr)

    def should_compute(self, istep: int) -> bool:
        return self.enabled and self.compute_frequency > 0 and istep % self.compute_frequency == 0

    # ------------------------------------------------------------------ data

    def _cell_means(self, rdata, stage_key: str) -> torch.Tensor | None:
        """Per-cell mean of the probe channels from one raw input window -> [C, k], NaN where empty."""
        if rdata is None or getattr(rdata, "data", None) is None:
            return None
        coords = rdata.coords
        data = rdata.data
        coords = coords.numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
        data_t = data if isinstance(data, torch.Tensor) else torch.as_tensor(np.asarray(data))
        if coords.shape[0] == 0:
            return None
        thetas, phis = theta_phi_to_standard_coords(coords)
        cells = torch.as_tensor(ang2pix(2**self.healpix_level, thetas, phis, nest=True), dtype=torch.long)
        idx = self._channel_idx[stage_key]
        vals = data_t[:, idx].float()
        num_cells = 12 * 4**self.healpix_level
        sums = torch.zeros(num_cells, vals.shape[1]).index_add_(0, cells, vals)
        counts = torch.zeros(num_cells).index_add_(0, cells, torch.ones(cells.shape[0]))
        means = sums / counts.clamp(min=1).unsqueeze(1)
        means[counts == 0] = float("nan")
        return means.to(self.device)

    @staticmethod
    def _latent(latent_dict: dict | None, keys: tuple[str, ...]) -> torch.Tensor | None:
        if not isinstance(latent_dict, dict):
            return None
        for k in keys:
            v = latent_dict.get(k, None)
            if v is None:
                continue
            v = getattr(v, "patch_tokens", v)
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _extract(self, batch, preds, targets_and_auxs, stage_key: str):
        z_t = self._latent(preds.latent[0] if preds.latent else None, ("encoder_tokens",))
        z_n = None
        for _name, ta in targets_and_auxs.items():
            lat = getattr(ta, "latent", None)
            if isinstance(lat, list):
                lat = lat[0] if lat else None
            z_n = self._latent(lat, ("encoder_tokens", "JEPA"))
            if z_n is not None:
                break
        if z_t is None or z_n is None or z_t.shape != z_n.shape:
            return None
        src = batch.get_source_samples().get_samples()
        tgt = batch.get_target_samples().get_samples()
        if len(src) != z_t.shape[0] or len(tgt) != z_n.shape[0]:
            return None
        y_t, y_n = [], []
        for s_src, s_tgt in zip(src, tgt, strict=True):
            sd_s = s_src.streams_data.get(self.stream)
            sd_t = s_tgt.streams_data.get(self.stream)
            if sd_s is None or sd_t is None:
                return None
            m_s = self._cell_means(sd_s.source_raw[0] if sd_s.source_raw else None, stage_key)
            m_t = self._cell_means(sd_t.source_raw[0] if sd_t.source_raw else None, stage_key)
            if m_s is None or m_t is None:
                if "noraw" not in self._warned:
                    logger.warning("online_probe: no raw source data in the batch (train_logging.online_probe.enabled must be set before the dataset is built); skipping.")
                    self._warned.add("noraw")
                return None
            y_t.append(m_s)
            y_n.append(m_t)
        return z_t.detach().float(), z_n.detach().float(), torch.stack(y_t), torch.stack(y_n)

    # ------------------------------------------------------------------ step

    def step(self, batch, preds, targets_and_auxs, stage, istep: int) -> None:
        """One probe update (training) or evaluation (validation) on the current batch."""
        if not self.enabled:
            return
        stage_key = "train" if str(stage).lower().startswith("train") else "val"
        if stage_key == "train" and not self.should_compute(istep):
            return
        try:
            out = self._extract(batch, preds, targets_and_auxs, stage_key)
        except Exception as e:  # a monitor must never take the run down
            if "extract" not in self._warned:
                logger.warning("online_probe: extraction failed (%s); probe skipped from now on.", e)
                self._warned.add("extract")
            self.enabled = False
            return
        if out is None:
            return
        z_t, z_n, y_t, y_n = out  # [B,C,D], [B,C,D], [B,C,k], [B,C,k]
        if self.heads is None:
            self._build_heads(z_t.shape[-1])
        b, c, d = z_t.shape
        z_t = z_t.reshape(b * c, d)
        z_n = z_n.reshape(b * c, d)
        y_t = y_t.reshape(b * c, -1)
        y_n = y_n.reshape(b * c, -1)
        valid = torch.isfinite(y_t).all(dim=1) & torch.isfinite(y_n).all(dim=1)
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        if 0 < self.max_cells < idx.numel():
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[: self.max_cells]]
        z_t, z_n, y_t, y_n = z_t[idx], z_n[idx], y_t[idx], y_n[idx]
        dy = y_n - y_t
        inputs = {"state": z_t, "tend": torch.cat([z_t, z_n], dim=1), "tend_t": z_t}
        targets = {"state": y_t, "tend": dy, "tend_t": dy}

        train = stage_key == "train"
        with torch.set_grad_enabled(train):
            losses = []
            for h in _HEADS:
                pred = self.heads[h](inputs[h])
                err = (pred - targets[h]) ** 2
                losses.append(err.mean())
                acc = self._acc[stage_key][h]
                e = err.detach().sum(dim=0)
                y = targets[h].detach()
                for j, ch in enumerate(self.channel_names):
                    a = acc[ch]
                    a[0] += e[j].item()
                    a[1] += y[:, j].sum().item()
                    a[2] += (y[:, j] ** 2).sum().item()
                    a[3] += y.shape[0]
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                torch.stack(losses).sum().backward()
                self.optimizer.step()

    # ------------------------------------------------------------------ report

    def get_cached_metrics(self, stage) -> dict[str, float]:
        """NMSE per head and channel over the accumulation window; clears the window."""
        stage_key = "train" if str(stage).lower().startswith("train") else "val"
        out: dict[str, float] = {}
        acc = self._acc.pop(stage_key, None)
        if not acc:
            return out
        for h in _HEADS:
            vals = []
            for ch in self.channel_names:
                sq, sy, sy2, n = acc[h][ch]
                if n < 2:
                    continue
                var = max(sy2 / n - (sy / n) ** 2, 1e-12)
                nmse = (sq / n) / var
                out[f"probe.{stage_key}.{h}.{ch}"] = float(nmse)
                vals.append(nmse)
            if vals:
                out[f"probe.{stage_key}.{h}.mean"] = float(np.mean(vals))
        return out
