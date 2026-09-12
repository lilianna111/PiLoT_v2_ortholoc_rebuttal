from __future__ import annotations
import warnings

warnings.simplefilter("ignore")
import time
import os
import os.path
from pathlib import Path
from loguru import logger
from typing import Any
import numpy as np
import torch


def _patch_torch_load_for_imcui_checkpoints() -> None:
    """imcui/MASt3R checkpoints may fail on PyTorch 2.6+ due to weights_only=True default."""
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs = {**kwargs, "weights_only": False}
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load  # type: ignore[method-assign]


_patch_torch_load_for_imcui_checkpoints()

from imcui.ui.utils import get_model, get_feature_model, get_matcher_zoo, ransac_zoo, match_features, match_dense, \
    extract_features, load_config
from imcui.hloc.utils import base_model as imcui_base_model
from ortholoc.image_matching import Matcher
from ortholoc.correspondences import Correspondences2D2D

CONFIG = load_config(os.path.join(os.path.dirname(__file__), "matchers_imcui.yaml"))
DENSE_MATCHERS_CONFIG = load_config(os.path.join(os.path.dirname(__file__), "dense_matchers.yaml"))
FEAT_EXTRACTORS_CONFIG = load_config(os.path.join(os.path.dirname(__file__), "feature_extractors.yaml"))
FEAT_MATCHERS_CONFIG = load_config(os.path.join(os.path.dirname(__file__), "feature_matchers.yaml"))
MATCHER_ZOO: dict[str, Any] = get_matcher_zoo(CONFIG["matcher_zoo"])
RANSAC_ZOO: dict[str, Any] = ransac_zoo

_REPO_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "model"
_CHECKPOINT_DIR = os.environ.get("ORTHOLOC_CHECKPOINT_DIR", "").strip()
_LOCAL_MODEL_DIR = Path(_CHECKPOINT_DIR).expanduser().resolve() if _CHECKPOINT_DIR else _REPO_MODEL_DIR
_LOCAL_CHECKPOINTS = {
    "duster/duster_vit_large.pth": (
        "ORTHOLOC_DUSTER_CKPT",
        _LOCAL_MODEL_DIR / "duster_vit_large.pth",
    ),
    "eloftr/eloftr_outdoor.ckpt": (
        "ORTHOLOC_ELOFTR_CKPT",
        _LOCAL_MODEL_DIR / "eloftr_outdoor.ckpt",
    ),
    "gim/gim_dkm_100h.ckpt": (
        "ORTHOLOC_GIM_DKM_CKPT",
        _LOCAL_MODEL_DIR / "gim_dkm_100h.ckpt",
    ),
    "loftr/minima_loftr.ckpt": (
        "ORTHOLOC_MINIMA_LOFTR_CKPT",
        _LOCAL_MODEL_DIR / "minima_loftr.ckpt",
    ),
    "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth": (
        "ORTHOLOC_MAST3R_CKPT",
        _LOCAL_MODEL_DIR / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    ),
    "roma/minima_roma.pth": (
        "ORTHOLOC_MINIMA_ROMA_CKPT",
        _LOCAL_MODEL_DIR / "minima_roma.pth",
    ),
    "roma/roma_outdoor.pth": (
        "ORTHOLOC_ROMA_CKPT",
        _LOCAL_MODEL_DIR / "roma_outdoor.pth",
    ),
    "roma/dinov2_vitl14_pretrain.pth": (
        "ORTHOLOC_ROMA_DINOV2_CKPT",
        _LOCAL_MODEL_DIR / "dinov2_vitl14_pretrain.pth",
    ),
}


def _resolve_local_checkpoint(env_name: str, default_path: Path) -> Path | None:
    env_path = os.environ.get(env_name, "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())
    candidates.append(default_path)
    return next((path for path in candidates if path.is_file()), None)


def _apply_local_checkpoints() -> None:
    """Prefer repo-local matcher checkpoints to avoid online downloads."""
    local_checkpoints = {}
    for hf_filename, (env_name, default_path) in _LOCAL_CHECKPOINTS.items():
        local_checkpoint = _resolve_local_checkpoint(env_name, default_path)
        if local_checkpoint is not None:
            local_checkpoints[hf_filename] = str(local_checkpoint)

    if not local_checkpoints:
        return

    original_download_model = imcui_base_model.BaseModel._download_model

    def patched_download_model(self, repo_id=None, filename=None, **kwargs):
        if filename in local_checkpoints:
            return local_checkpoints[filename]
        return original_download_model(self, repo_id=repo_id, filename=filename, **kwargs)

    imcui_base_model.BaseModel._download_model = patched_download_model  # type: ignore[method-assign]
    for hf_filename, local_path in local_checkpoints.items():
        logger.info(f"{hf_filename}: using local weights ({local_path})")


_apply_local_checkpoints()


class MatcherIMCUI(Matcher):
    def __init__(
        self,
        name: str,
        device: str = 'cuda',
        extract_max_keypoints: int | None = None,
        angles: list[float] | None = None,
        keypoint_threshold: float = 0.015,
    ) -> None:
        # load model
        t0 = time.time()
        logger.info(f'Loading model {name}')
        self.model = MATCHER_ZOO[name]
        self.match_conf = self.model["matcher"]

        efficiency = self.model["info"].get("efficiency", "high")
        if efficiency == "low":
            logger.warning("Matcher {} is time-consuming, please wait for a while".format(self.model["info"].get(
                "name", "unknown")))

        # get models with config
        if self.model["dense"]:
            matcher_name = CONFIG['matcher_zoo'][name]['matcher']
            if matcher_name in DENSE_MATCHERS_CONFIG and "preprocessing" in DENSE_MATCHERS_CONFIG[matcher_name]:
                self.match_conf["preprocessing"].update(DENSE_MATCHERS_CONFIG[matcher_name]["preprocessing"])
            if matcher_name in DENSE_MATCHERS_CONFIG and "model" in DENSE_MATCHERS_CONFIG[matcher_name]:
                self.match_conf["model"].update(DENSE_MATCHERS_CONFIG[matcher_name]["model"])
            if angles is None:
                if 'xfeat' in name:
                    angles = [0, 90, 180, 270]
                else:
                    angles = [0]
        else:
            t0 = time.time()
            logger.info('Loading feature model')
            matcher_name = CONFIG['matcher_zoo'][name]['matcher']
            self.extract_conf = self.model["feature"]
            # update extract config
            if extract_max_keypoints is not None:
                self.extract_conf["model"]["max_keypoints"] = extract_max_keypoints
            self.extract_conf["model"]["keypoint_threshold"] = keypoint_threshold
            self.extractor = get_feature_model(self.extract_conf)
            self.extractor.to(device)
            logger.info(f"Loaded feature model: {time.time() - t0:.3f}s")
            if matcher_name in FEAT_MATCHERS_CONFIG and "preprocessing" in FEAT_MATCHERS_CONFIG[matcher_name]:
                self.extract_conf["preprocessing"].update(FEAT_MATCHERS_CONFIG[matcher_name]["preprocessing"])
            if angles is None:
                angles = [0, 90, 180, 270]

        self.matcher = get_model(self.match_conf)
        self.matcher.to(device)
        logger.info(f"Loaded model: {time.time() - t0:.3f}s")

        super().__init__(name=name, device=device, angles=angles)
        logger.info(f"Setting angles to {self.angles}")

    def __call__(self, img0: np.ndarray, img1: np.ndarray, covisible_only: bool = True, normalized: bool = True,
                 silent: bool = False, *args: Any, **kwargs: Any) -> Correspondences2D2D:
        t1 = time.time()
        if not silent:
            logger.info("Matching...")
        if self.model["dense"]:
            pred = match_dense.match_images(self.matcher, img0, img1, self.match_conf["preprocessing"],
                                            device=self.device)
        else:
            pred0 = extract_features.extract(self.extractor, img0, self.extract_conf["preprocessing"])
            pred1 = extract_features.extract(self.extractor, img1, self.extract_conf["preprocessing"])
            pred = match_features.match_images(self.matcher, pred0, pred1)

        pts0 = pred["mkeypoints0_orig"]
        pts1 = pred["mkeypoints1_orig"]
        confidences = pred["mconf"] if "mconf" in pred else None

        correspondences = self.build_correspondences(pts0=pts0, pts1=pts1, h0=img0.shape[0], w0=img0.shape[1],
                                                     h1=img1.shape[0], w1=img1.shape[1], confidences=confidences,
                                                     covisible_only=covisible_only, normalized=normalized)

        if not silent:
            logger.info(f"Matched images: {time.time() - t1:.3f}s")
        return correspondences
