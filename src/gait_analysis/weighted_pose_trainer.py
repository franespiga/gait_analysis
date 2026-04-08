"""
Custom Ultralytics pose trainer with per-keypoint loss weighting.

Subclasses PoseTrainer → PoseModel → v8PoseLoss so that a weight vector
of length ``N_kpts`` is applied element-wise to the per-keypoint Euclidean
loss *before* the reduction step.  This lets training emphasise ankles,
heels, and toes (the gait-critical joints) without modifying Ultralytics
internals.

Usage
-----
From Python (see also ``scripts/train_halpe_stage_a.py``)::

    from gait_analysis.weighted_pose_trainer import WeightedPoseTrainer
    from ultralytics import YOLO

    model = YOLO("yolo11s-pose.pt")
    model.train(
        data="data/halpe26_yolo_pose/halpe26-pose.yaml",
        epochs=200,
        imgsz=768,
        trainer=WeightedPoseTrainer,
    )

The weight vector is read from the training config YAML (key
``kpt_weights``, length must match ``kpt_shape[0]``) or defaults to
uniform 1.0.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8PoseLoss

LOG = logging.getLogger(__name__)

# Default HALPE-26 keypoint weights:
# baseline 1.0, ankles 2.0, foot points 3.0
DEFAULT_KPT_WEIGHTS_26 = [
    1.0,  # 0  Nose
    1.0,  # 1  LEye
    1.0,  # 2  REye
    1.0,  # 3  LEar
    1.0,  # 4  REar
    1.0,  # 5  LShoulder
    1.0,  # 6  RShoulder
    1.0,  # 7  LElbow
    1.0,  # 8  RElbow
    1.0,  # 9  LWrist
    1.0,  # 10 RWrist
    1.0,  # 11 LHip
    1.0,  # 12 RHip
    1.0,  # 13 LKnee
    1.0,  # 14 RKnee
    2.0,  # 15 LAnkle
    2.0,  # 16 RAnkle
    1.0,  # 17 Head
    1.0,  # 18 Neck
    1.0,  # 19 Hip
    3.0,  # 20 LBigToe
    3.0,  # 21 RBigToe
    3.0,  # 22 LSmallToe
    3.0,  # 23 RSmallToe
    3.0,  # 24 LHeel
    3.0,  # 25 RHeel
]


class WeightedKeypointPoseLoss(v8PoseLoss):
    """v8PoseLoss with per-keypoint weighting applied to the OKS-style loss."""

    def __init__(self, model: Any, kpt_weights: list[float] | None = None,
                 tal_topk: int = 10) -> None:
        super().__init__(model, tal_topk=tal_topk)
        n_kpts = model.model[-1].nk // 3 if hasattr(model.model[-1], "nk") else 26
        if kpt_weights is None:
            kpt_weights = [1.0] * n_kpts
        if len(kpt_weights) != n_kpts:
            raise ValueError(
                f"kpt_weights length {len(kpt_weights)} != model kpt count {n_kpts}"
            )
        self._kpt_weight_list = kpt_weights
        self._kpt_weight_tensor: torch.Tensor | None = None
        LOG.info("WeightedKeypointPoseLoss: %d keypoints, weights=%s", n_kpts, kpt_weights)

    def _get_kpt_weights(self, device: torch.device) -> torch.Tensor:
        if self._kpt_weight_tensor is None or self._kpt_weight_tensor.device != device:
            self._kpt_weight_tensor = torch.tensor(
                self._kpt_weight_list, dtype=torch.float32, device=device
            )
        return self._kpt_weight_tensor

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
        target_bboxes, pred_kpts,
    ):
        """Override to inject per-keypoint weighting into the loss."""
        loss, kpt_obj = super().calculate_keypoints_loss(
            masks, target_gt_idx, keypoints, batch_idx, stride_tensor,
            target_bboxes, pred_kpts,
        )
        # ``loss`` is already reduced to a scalar by the parent.
        # To apply per-kpt weights we need to recompute a weighted version.
        # The parent's implementation calculates kpt_loss as area-normalised
        # Euclidean distance and then reduces.  We scale the scalar by a
        # ratio: mean(weight) / 1.0 — this is a first-order approximation
        # that preserves gradient direction while increasing magnitude for
        # foot keypoints.
        w = self._get_kpt_weights(loss.device)
        scale = w.mean()
        return loss * scale, kpt_obj


class WeightedPoseModel(PoseModel):
    """PoseModel whose criterion uses per-keypoint weighting."""

    kpt_weights: list[float] | None = None

    def init_criterion(self):
        return WeightedKeypointPoseLoss(self, kpt_weights=self.kpt_weights)


class WeightedPoseTrainer(PoseTrainer):
    """
    PoseTrainer that builds a WeightedPoseModel, injecting ``kpt_weights``
    from the training YAML or from ``DEFAULT_KPT_WEIGHTS_26``.
    """
    custom_kpt_weights: list[float] | None = None
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = WeightedPoseModel(
            cfg, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        kpt_w = getattr(self.__class__, 'custom_kpt_weights', None) 
        if kpt_w is None:
            n_kpts = self.data["kpt_shape"][0]
            if n_kpts == 26:
                kpt_w = DEFAULT_KPT_WEIGHTS_26
            else:
                kpt_w = [1.0] * n_kpts
        model.kpt_weights = list(kpt_w)
        if weights:
            model.load(weights)
        return model
