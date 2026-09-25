"""Convert one camera view of an sDANNCE save_data_AVG0.mat 3D-pose file into a
per-camera 2D .slp that laras-labeler (sleap-io) can load alongside that camera's video.

DANNCE poses are triangulated 3D world coordinates (mm), not pixel coordinates, so they
are re-projected into a chosen camera's 2D image plane using that camera's calibration
(calibration.json, standard OpenCV convention: K, rotation_matrix, translation_vector,
r_distort=[k1,k2], t_distort=[p1,p2]). Verified by re-projecting frame 0 onto Camera1's
actual video frame and visually confirming the points land on the rat.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import sleap_io as sio


def project_camera(pts3d_flat: np.ndarray, cam_params: dict) -> np.ndarray:
    K = np.array(cam_params["camera_matrix"], dtype=np.float64)
    R = np.array(cam_params["rotation_matrix"], dtype=np.float64)
    t = np.array(cam_params["translation_vector"], dtype=np.float64).reshape(3)
    dist = np.array(cam_params["r_distort"][:2] + cam_params["t_distort"][:2], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    uv, _ = cv2.projectPoints(pts3d_flat, rvec, t, K, dist)
    return uv.reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cage_dir", help="e.g. .../day1/cage1RB")
    ap.add_argument("--mat", default="sDANNCE/predict01/save_data_AVG0.mat")
    ap.add_argument("--calib", default="calibration/calibration.json")
    ap.add_argument("--camera", default="Camera1")
    ap.add_argument("--out", default=None, help="output .slp path")
    ap.add_argument("--max-frames", type=int, default=None, help="truncate for a quick test")
    args = ap.parse_args()

    cage_dir = Path(args.cage_dir)
    cal = json.load(open(cage_dir / args.calib))
    cam_idx = cal["camera_names"].index(args.camera)
    cam_params = cal["camera_params"][cam_idx]

    m = scipy.io.loadmat(cage_dir / args.mat)
    pred = m["pred"]  # (F, T, 3, N)
    p_max = m["p_max"]  # (F, T, N)
    if args.max_frames:
        pred = pred[: args.max_frames]
        p_max = p_max[: args.max_frames]
    F, T, _, N = pred.shape
    print(f"{F} frames, {T} animals, {N} keypoints")

    pts3d = pred.transpose(0, 1, 3, 2).reshape(-1, 3).astype(np.float64)  # (F*T*N, 3)
    uv = project_camera(pts3d, cam_params).reshape(F, T, N, 2)

    tracks_arr = np.concatenate([uv, p_max[..., None]], axis=-1).astype("float32")  # (F,T,N,3)

    video_path = cage_dir / "videos" / args.camera / "0.mp4"
    video = sio.Video.from_filename(str(video_path))
    if video.shape[0] != F:
        print(f"WARNING: video has {video.shape[0]} frames, poses have {F} -- check alignment")

    skeleton = sio.Skeleton(nodes=[f"kp{i:02d}" for i in range(N)], name="dannce23")
    tracks = [sio.Track(name=f"rat{i}") for i in range(T)]

    labels = sio.Labels.from_numpy(
        tracks_arr, videos=[video], skeletons=[skeleton], tracks=tracks,
        first_frame=0, return_confidence=True,
    )

    out = Path(args.out) if args.out else cage_dir / f"{args.camera.lower()}_poses.slp"
    labels.save(str(out))
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
