"""Gravity priors: signed world Up expressed in the OpenCV camera frame."""

from pathlib import Path
import numpy as np


def require_gravity_backend():
    try:
        from ortholoc import _gravity_pnp
    except ImportError as exc:
        raise RuntimeError('Gravity PnP backend is missing for this Python. Run: '
                           'bash scripts/build_gravity_pnp.sh in the active OrthoLoC environment') from exc
    return _gravity_pnp


def camera_up_from_roll_pitch(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Use this project's camera/gimbal convention, NOT generic body IMU Euler angles.

    In main.py: R_enu_from_cam = R.from_euler('xyz', [pitch - 180, roll, yaw]).
    R_enu_from_cam.T @ [0, 0, 1] is independent of yaw.
    """
    roll, pitch = np.deg2rad([roll_deg, pitch_deg - 180.0])
    return np.array([-np.sin(roll), np.sin(pitch) * np.cos(roll), np.cos(pitch) * np.cos(roll)])


def ecef_up(lon_deg: float, lat_deg: float) -> np.ndarray:
    """Subtracting an ECEF origin does not rotate ECEF Z onto local vertical."""
    lon, lat = np.deg2rad([lon_deg, lat_deg])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def load_gravity_priors(path: str, prior_format: str = 'roll_pitch') -> dict[str, np.ndarray]:
    """Read exact image basenames, never align priors by row index or use GT fallback.

    roll_pitch:     image.jpg roll_deg pitch_deg (camera/gimbal convention above)
    camera_up:      image.jpg ux uy uz (signed Up, OpenCV camera frame)
    For body IMU measurements apply the body-to-camera extrinsic before writing camera_up.
    """
    if prior_format not in ('roll_pitch', 'camera_up'):
        raise ValueError(f"Unknown gravity prior format: {prior_format}")
    priors = {}
    for line_num, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        expected = 3 if prior_format == 'roll_pitch' else 4
        if len(fields) != expected:
            raise ValueError(f"{path}:{line_num}: expected {expected} fields for {prior_format}")
        name = Path(fields[0]).name
        if name in priors:
            raise ValueError(f"{path}:{line_num}: duplicate image {name}")
        values = np.asarray(fields[1:], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}:{line_num}: nonfinite gravity prior")
        up = camera_up_from_roll_pitch(*values) if prior_format == 'roll_pitch' else values
        if np.linalg.norm(up) < 1e-12:
            raise ValueError(f"{path}:{line_num}: zero gravity vector")
        priors[name] = up / np.linalg.norm(up)
    if not priors:
        raise ValueError(f"No gravity priors in {path}")
    return priors
