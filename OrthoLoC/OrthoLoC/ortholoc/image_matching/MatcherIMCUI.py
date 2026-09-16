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
    original_load_state_dict_from_url = torch.hub.load_state_dict_from_url

    def patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs = {**kwargs, "weights_only": False}
        return original_torch_load(*args, **kwargs)

    def patched_load_state_dict_from_url(url, *args, **kwargs):
        local_checkpoint = _resolve_local_checkpoint(url)
        if local_checkpoint is not None:
            logger.info(f"Using local checkpoint for {Path(url).name}: {local_checkpoint}")
            map_location = kwargs.get("map_location", "cpu")
            return original_torch_load(local_checkpoint, map_location=map_location, weights_only=False)

        raise FileNotFoundError(
            f"Local checkpoint not found for {url}. "
            f"Searched directories: {', '.join(str(path) for path in _local_checkpoint_search_dirs())}. "
            "Online download is disabled in this OrthoLoC integration."
        )

    torch.load = patched_torch_load  # type: ignore[method-assign]
    torch.hub.load_state_dict_from_url = patched_load_state_dict_from_url  # type: ignore[method-assign]


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

_MAST3R_HF_FILENAME = "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
_DEFAULT_LOCAL_CHECKPOINT_DIR = "/media/amax/PS2000/checkpoint"


def _local_checkpoint_search_dirs() -> list[Path]:
    dirs: list[Path] = []

    env_dir = os.environ.get("ORTHOLOC_CHECKPOINT_DIR", "").strip()
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    else:
        dirs.append(Path(_DEFAULT_LOCAL_CHECKPOINT_DIR))

    dirs.append(Path(__file__).resolve().parent.parent.parent / "model")
    return dirs


def _resolve_local_checkpoint(filename: str | None) -> Path | None:
    if not filename:
        return None

    basename = Path(filename).name
    for directory in _local_checkpoint_search_dirs():
        candidate = (directory / basename).resolve()
        if candidate.is_file():
            return candidate
    return None


def _apply_local_checkpoint_overrides() -> None:
    """Prefer local checkpoints and disable online downloads for known matcher weights."""
    original_download_model = imcui_base_model.BaseModel._download_model

    def patched_download_model(self, repo_id=None, filename=None, **kwargs):
        local_checkpoint = _resolve_local_checkpoint(filename)
        if local_checkpoint is not None:
            logger.info(f"Using local checkpoint for {Path(filename).name}: {local_checkpoint}")
            return str(local_checkpoint)

        if filename:
            searched = ", ".join(str(path) for path in _local_checkpoint_search_dirs())
            raise FileNotFoundError(
                f"Local checkpoint not found for {filename}. "
                f"Searched directories: {searched}. "
                "Online download is disabled in this OrthoLoC integration."
            )

        return original_download_model(self, repo_id=repo_id, filename=filename, **kwargs)

    imcui_base_model.BaseModel._download_model = patched_download_model  # type: ignore[method-assign]
    logger.info(
        "Enabled local checkpoint overrides for OrthoLoC. Search dirs: "
        + ", ".join(str(path) for path in _local_checkpoint_search_dirs())
    )


_apply_local_checkpoint_overrides()


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