#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node for geometry-constrained fusion denoising of a depth-camera point cloud
with a registered single-line LiDAR point cloud.

Main steps
----------
1. Read registered camera and LiDAR point clouds from PCD/PLY files.
2. Estimate a straight LiDAR centerline.
3. Project camera and LiDAR points onto the LiDAR centerline.
4. Unroll both point clouds into a common parameterized domain.
5. Build cross-modal correspondences.
6. Train an ensemble regressor for residual prediction.
7. Apply entropy-based prediction filtering.
8. Map corrected points back to 3D space.
9. Save intermediate and final PCD files.

ROS parameters
--------------
~camera_pcd               : Registered camera point cloud (.pcd/.ply)
~lidar_pcd                : Registered LiDAR point cloud (.pcd/.ply)
~output_dir               : Output directory for generated point clouds
~voxel_size               : Voxel size for optional downsampling
~n_dense                  : Number of sampled centerline points
~extend_ratio             : Centerline extension ratio at both ends
~y_mode                   : 'arc' or 'theta' for unrolled y coordinate
~k_geom                   : Neighborhood size for LiDAR geometry estimation
~k_cov                    : Neighborhood size for camera covariance estimation
~max_s_diff               : Correspondence threshold in longitudinal coordinate
~max_theta_diff           : Correspondence threshold in angular coordinate
~max_3d_dist              : Correspondence threshold in 3D distance
~lambda_res               : Mahalanobis residual weight coefficient
~beta_curv                : Curvature-related weight coefficient
~n_estimators             : Number of trees in the random forest ensemble
~max_depth                : Maximum tree depth
~min_samples_leaf         : Minimum leaf size in the random forest
~random_state             : Random seed
~lambda_n                 : Normal-direction refinement weight
~lambda_t                 : Tangential refinement weight
~entropy_quantile         : Entropy percentile for prediction filtering
~keep_unmatched           : Whether to merge unmatched original camera points
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
import rospy
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)



def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi



def read_point_cloud_any(path: str) -> Dict[str, Optional[np.ndarray]]:
    ext = Path(path).suffix.lower()
    if ext not in [".pcd", ".ply"]:
        raise ValueError("Only .pcd and .ply files are supported.")

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError(f"Empty point cloud: {path}")

    cols = np.asarray(pcd.colors)
    if cols.size == 0 or len(cols) != len(pts):
        cols = None

    return {"points": pts.astype(np.float64), "colors": cols}



def save_point_cloud_pcd(points: np.ndarray, out_path: str, colors: Optional[np.ndarray] = None) -> None:
    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None and len(colors) == len(points):
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Failed to save PCD file: {out_path}")



def voxel_downsample_with_color(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))

    if colors is not None and len(colors) == len(points):
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd_ds = pcd.voxel_down_sample(voxel_size)
    pts_ds = np.asarray(pcd_ds.points)
    cols_ds = np.asarray(pcd_ds.colors)
    if cols_ds.size == 0 or len(cols_ds) != len(pts_ds):
        cols_ds = None
    return pts_ds, cols_ds


# -----------------------------------------------------------------------------
# Straight centerline estimation from LiDAR
# -----------------------------------------------------------------------------
def estimate_straight_centerline_from_lidar(
    points: np.ndarray,
    n_dense: int = 1200,
    extend_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    mean = pts.mean(axis=0)
    x = pts - mean

    _, _, vh = np.linalg.svd(x, full_matrices=False)
    direction = vh[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    proj = x @ direction
    s_min = proj.min()
    s_max = proj.max()
    length = s_max - s_min

    s_min = s_min - extend_ratio * length
    s_max = s_max + extend_ratio * length

    s_dense = np.linspace(s_min, s_max, n_dense)
    centerline = mean[None, :] + s_dense[:, None] * direction[None, :]
    return centerline, s_dense, mean, direction



def build_frames_for_straight_centerline(
    centerline: np.ndarray,
    direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    t = np.tile(direction[None, :], (len(centerline), 1))

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u0 = normalize(np.cross(ref, direction).reshape(1, 3))[0]
    v0 = normalize(np.cross(direction, u0).reshape(1, 3))[0]

    u = np.tile(u0[None, :], (len(centerline), 1))
    v = np.tile(v0[None, :], (len(centerline), 1))
    return t, u, v


# -----------------------------------------------------------------------------
# Unrolling and inverse mapping
# -----------------------------------------------------------------------------
def project_points_to_centerline(
    points: np.ndarray,
    centerline: np.ndarray,
    s_dense: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> Dict[str, np.ndarray]:
    tree = cKDTree(centerline)
    _, idx = tree.query(points, k=1)

    c = centerline[idx]
    uu = u[idx]
    vv = v[idx]
    ss = s_dense[idx]

    q = points - c
    qu = np.sum(q * uu, axis=1)
    qv = np.sum(q * vv, axis=1)

    r = np.sqrt(qu ** 2 + qv ** 2)
    theta = np.arctan2(qv, qu)

    return {
        "idx": idx,
        "s": ss,
        "r": r,
        "theta": theta,
        "center": c,
        "u": uu,
        "v": vv,
    }



def build_unrolled_points(
    unroll_data: Dict[str, np.ndarray],
    y_mode: str = "arc",
    r_ref: Optional[float] = None,
) -> np.ndarray:
    s = unroll_data["s"]
    theta = unroll_data["theta"]

    if y_mode == "theta":
        y = theta
    elif y_mode == "arc":
        if r_ref is None:
            raise ValueError("r_ref is required when y_mode='arc'.")
        y = r_ref * theta
    else:
        raise ValueError("y_mode must be 'arc' or 'theta'.")

    z = np.zeros_like(s)
    return np.column_stack([s, y, z]).astype(np.float64)



def interp_centerline_frames_straight(
    s_query: np.ndarray,
    center_mean: np.ndarray,
    direction: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_query = np.asarray(s_query, dtype=np.float64)
    c = center_mean[None, :] + s_query[:, None] * direction[None, :]
    uu = np.tile(u0[None, :], (len(s_query), 1))
    vv = np.tile(v0[None, :], (len(s_query), 1))
    return c, uu, vv



def inverse_map_points_straight(
    s_hat: np.ndarray,
    theta_hat: np.ndarray,
    r_hat: np.ndarray,
    center_mean: np.ndarray,
    direction: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
) -> np.ndarray:
    c, uu, vv = interp_centerline_frames_straight(s_hat, center_mean, direction, u0, v0)
    pts = (
        c
        + r_hat[:, None] * np.cos(theta_hat)[:, None] * uu
        + r_hat[:, None] * np.sin(theta_hat)[:, None] * vv
    )
    return pts


# -----------------------------------------------------------------------------
# Local geometry estimation
# -----------------------------------------------------------------------------
def estimate_normals_and_curvature(points: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    tree = cKDTree(points)
    k_eff = min(k, len(points))
    _, nn_idx = tree.query(points, k=k_eff)

    normals = np.zeros_like(points)
    curvature = np.zeros((len(points),), dtype=np.float64)

    for i in range(len(points)):
        nbrs = points[nn_idx[i]]
        mu = nbrs.mean(axis=0)
        x = nbrs - mu
        c = (x.T @ x) / max(len(nbrs), 1)

        eigvals, eigvecs = np.linalg.eigh(c)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n = eigvecs[:, 0]
        if np.dot(n, points[i] - mu) < 0:
            n = -n

        normals[i] = n
        curvature[i] = eigvals[0] / (np.sum(eigvals) + 1e-12)

    normals = normalize(normals)
    return normals, curvature



def estimate_local_covariances(points: np.ndarray, k: int = 20, eps: float = 1e-6) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    tree = cKDTree(points)
    k_eff = min(k, len(points))
    _, nn_idx = tree.query(points, k=k_eff)

    covs = np.zeros((len(points), 3, 3), dtype=np.float64)
    for i in range(len(points)):
        nbrs = points[nn_idx[i]]
        mu = nbrs.mean(axis=0)
        x = nbrs - mu
        c = (x.T @ x) / max(len(nbrs), 1)
        covs[i] = c + eps * np.eye(3)
    return covs


# -----------------------------------------------------------------------------
# Cross-modal correspondence
# -----------------------------------------------------------------------------
def build_unrolled_correspondences(
    cam_unroll: Dict[str, np.ndarray],
    lidar_unroll: Dict[str, np.ndarray],
    cam_points: np.ndarray,
    lidar_points: np.ndarray,
    max_s_diff: float = 0.30,
    max_theta_diff: float = 0.35,
    max_3d_dist: float = 0.35,
) -> np.ndarray:
    mean_r = np.median(lidar_unroll["r"]) if len(lidar_unroll["r"]) > 0 else 1.0
    lidar_2d = np.column_stack([lidar_unroll["s"], mean_r * lidar_unroll["theta"]])
    cam_2d = np.column_stack([cam_unroll["s"], mean_r * cam_unroll["theta"]])

    tree = cKDTree(lidar_2d)
    _, idx = tree.query(cam_2d, k=1)

    ds = np.abs(cam_unroll["s"] - lidar_unroll["s"][idx])
    dtheta = np.abs(wrap_angle(cam_unroll["theta"] - lidar_unroll["theta"][idx]))
    d3 = np.linalg.norm(cam_points - lidar_points[idx], axis=1)

    valid = (ds <= max_s_diff) & (dtheta <= max_theta_diff) & (d3 <= max_3d_dist)

    match_idx = np.full((len(cam_points),), -1, dtype=np.int32)
    match_idx[valid] = idx[valid]
    return match_idx


# -----------------------------------------------------------------------------
# Features, targets, and weights
# -----------------------------------------------------------------------------
def compute_mahalanobis_residual(delta: np.ndarray, cov: np.ndarray) -> float:
    cov_inv = np.linalg.inv(cov)
    return np.sqrt(np.clip(delta.T @ cov_inv @ delta, 0.0, None))



def build_training_data(
    cam_points: np.ndarray,
    lidar_points: np.ndarray,
    cam_unroll: Dict[str, np.ndarray],
    lidar_unroll: Dict[str, np.ndarray],
    lidar_normals: np.ndarray,
    lidar_curvature: np.ndarray,
    cam_covs: np.ndarray,
    match_idx: np.ndarray,
    lambda_res: float = 1.0,
    beta_curv: float = 10.0,
):
    valid = match_idx >= 0
    cam_idx = np.where(valid)[0]
    lid_idx = match_idx[valid]

    pc = cam_points[cam_idx]
    pl = lidar_points[lid_idx]

    sc = cam_unroll["s"][cam_idx]
    rc = cam_unroll["r"][cam_idx]
    tc = cam_unroll["theta"][cam_idx]

    sl = lidar_unroll["s"][lid_idx]
    rl = lidar_unroll["r"][lid_idx]
    tl = lidar_unroll["theta"][lid_idx]

    ds = sl - sc
    dtheta = wrap_angle(tl - tc)
    dr = rl - rc

    normals = lidar_normals[lid_idx]
    curv = lidar_curvature[lid_idx]
    delta_xyz = pl - pc

    maha = np.zeros((len(cam_idx),), dtype=np.float64)
    for i, ci in enumerate(cam_idx):
        maha[i] = compute_mahalanobis_residual(delta_xyz[i], cam_covs[ci])

    x = np.column_stack([
        sc, tc, rc,
        sl, tl, rl,
        ds, dtheta, dr,
        normals[:, 0], normals[:, 1], normals[:, 2],
        curv,
        maha,
    ])

    y = np.column_stack([ds, dtheta, dr])
    w = np.exp(-lambda_res * maha - beta_curv * curv * np.sum(delta_xyz ** 2, axis=1))
    pre_residual = np.linalg.norm(delta_xyz, axis=1)

    return x, y, w, cam_idx, lid_idx, maha, pre_residual


# -----------------------------------------------------------------------------
# Ensemble regression and entropy filtering
# -----------------------------------------------------------------------------
def train_ensemble_regressor(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_estimators: int = 160,
    max_depth: int = 18,
    min_samples_leaf: int = 3,
    random_state: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(x, y, sample_weight=w)
    return model



def predict_with_entropy(
    model: RandomForestRegressor,
    x: np.ndarray,
    r_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tree_preds = np.stack([est.predict(x) for est in model.estimators_], axis=0)
    pred_mean = tree_preds.mean(axis=0)

    b, n, _ = tree_preds.shape
    entropy = np.zeros((n,), dtype=np.float64)

    z = np.zeros_like(tree_preds)
    z[:, :, 0] = tree_preds[:, :, 0]
    z[:, :, 1] = tree_preds[:, :, 1] * r_ref[None, :]
    z[:, :, 2] = tree_preds[:, :, 2]

    for i in range(n):
        zi = z[:, i, :]
        mu = zi.mean(axis=0, keepdims=True)
        xc = zi - mu
        c = (xc.T @ xc) / max(b - 1, 1)
        eigvals = np.linalg.eigvalsh(c)
        eigvals = np.clip(eigvals, 1e-12, None)
        eigvals = eigvals / eigvals.sum()
        entropy[i] = -np.sum(eigvals * np.log(eigvals))

    return pred_mean, entropy



def geometry_refine(
    pred_points: np.ndarray,
    lidar_match_points: np.ndarray,
    lidar_match_normals: np.ndarray,
    lambda_n: float = 0.85,
    lambda_t: float = 0.15,
) -> np.ndarray:
    delta = pred_points - lidar_match_points
    n = lidar_match_normals
    dn = np.sum(delta * n, axis=1, keepdims=True)
    delta_n = dn * n
    delta_t = delta - delta_n
    refined = pred_points - lambda_n * delta_n - lambda_t * delta_t
    return refined


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    rospy.init_node("tunnel_fusion_denoise_node", anonymous=False)

    camera_path = rospy.get_param("~camera_pcd")
    lidar_path = rospy.get_param("~lidar_pcd")
    outdir = rospy.get_param("~output_dir", "./results")

    voxel_size = float(rospy.get_param("~voxel_size", 0.05))
    n_dense = int(rospy.get_param("~n_dense", 1200))
    extend_ratio = float(rospy.get_param("~extend_ratio", 0.0))
    y_mode = rospy.get_param("~y_mode", "arc")

    k_geom = int(rospy.get_param("~k_geom", 20))
    k_cov = int(rospy.get_param("~k_cov", 20))

    max_s_diff = float(rospy.get_param("~max_s_diff", 0.30))
    max_theta_diff = float(rospy.get_param("~max_theta_diff", 0.35))
    max_3d_dist = float(rospy.get_param("~max_3d_dist", 0.35))

    lambda_res = float(rospy.get_param("~lambda_res", 1.0))
    beta_curv = float(rospy.get_param("~beta_curv", 10.0))

    n_estimators = int(rospy.get_param("~n_estimators", 160))
    max_depth = int(rospy.get_param("~max_depth", 18))
    min_samples_leaf = int(rospy.get_param("~min_samples_leaf", 3))
    random_state = int(rospy.get_param("~random_state", 42))

    lambda_n = float(rospy.get_param("~lambda_n", 0.85))
    lambda_t = float(rospy.get_param("~lambda_t", 0.15))
    entropy_quantile = float(rospy.get_param("~entropy_quantile", 0.85))
    keep_unmatched = bool(rospy.get_param("~keep_unmatched", False))

    ensure_dir(outdir)
    outdir_path = Path(outdir)

    rospy.loginfo("Reading point clouds...")
    cam_data = read_point_cloud_any(camera_path)
    lidar_data = read_point_cloud_any(lidar_path)

    cam = cam_data["points"]
    cam_colors = cam_data["colors"]
    lidar = lidar_data["points"]
    lidar_colors = lidar_data["colors"]

    rospy.loginfo(f"Original camera points: {len(cam)}")
    rospy.loginfo(f"Original LiDAR points: {len(lidar)}")

    if voxel_size > 0:
        cam, cam_colors = voxel_downsample_with_color(cam, cam_colors, voxel_size)
        lidar, lidar_colors = voxel_downsample_with_color(lidar, lidar_colors, voxel_size)
        rospy.loginfo(f"Downsampled camera points: {len(cam)}")
        rospy.loginfo(f"Downsampled LiDAR points: {len(lidar)}")

    rospy.loginfo("Estimating straight LiDAR centerline...")
    centerline, s_dense, center_mean, center_dir = estimate_straight_centerline_from_lidar(
        lidar,
        n_dense=n_dense,
        extend_ratio=extend_ratio,
    )

    centerline_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(centerline), 1))
    save_point_cloud_pcd(centerline, str(outdir_path / "lidar_centerline.pcd"), centerline_colors)

    rospy.loginfo("Building local frames...")
    _, uu, vv = build_frames_for_straight_centerline(centerline, center_dir)
    u0 = uu[0]
    v0 = vv[0]

    rospy.loginfo("Projecting point clouds onto the LiDAR centerline...")
    cam_unroll = project_points_to_centerline(cam, centerline, s_dense, uu, vv)
    lidar_unroll = project_points_to_centerline(lidar, centerline, s_dense, uu, vv)

    r_ref = np.median(lidar_unroll["r"])
    rospy.loginfo(f"Reference radius r_ref = {r_ref:.6f}")

    camera_unrolled = build_unrolled_points(cam_unroll, y_mode=y_mode, r_ref=r_ref)
    lidar_unrolled = build_unrolled_points(lidar_unroll, y_mode=y_mode, r_ref=r_ref)

    if lidar_colors is None:
        lidar_unrolled_colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(lidar_unrolled), 1))
    else:
        lidar_unrolled_colors = lidar_colors

    save_point_cloud_pcd(camera_unrolled, str(outdir_path / "camera_unrolled.pcd"), cam_colors)
    save_point_cloud_pcd(lidar_unrolled, str(outdir_path / "lidar_unrolled.pcd"), lidar_unrolled_colors)

    rospy.loginfo("Estimating local LiDAR geometry...")
    lidar_normals, lidar_curvature = estimate_normals_and_curvature(lidar, k=k_geom)

    rospy.loginfo("Estimating local camera covariances...")
    cam_covs = estimate_local_covariances(cam, k=k_cov)

    rospy.loginfo("Building cross-modal correspondences...")
    match_idx = build_unrolled_correspondences(
        cam_unroll,
        lidar_unroll,
        cam,
        lidar,
        max_s_diff=max_s_diff,
        max_theta_diff=max_theta_diff,
        max_3d_dist=max_3d_dist,
    )

    valid_ratio = np.mean(match_idx >= 0)
    rospy.loginfo(f"Valid correspondence ratio: {valid_ratio:.4f}")

    rospy.loginfo("Constructing training data...")
    x, y, w, cam_idx, lid_idx, _, pre_residual = build_training_data(
        cam,
        lidar,
        cam_unroll,
        lidar_unroll,
        lidar_normals,
        lidar_curvature,
        cam_covs,
        match_idx,
        lambda_res=lambda_res,
        beta_curv=beta_curv,
    )

    if len(x) < 100:
        raise RuntimeError("Too few valid correspondences for training. Please relax the thresholds.")

    rospy.loginfo(f"Training samples: {len(x)}")

    rospy.loginfo("Training ensemble regressor...")
    model = train_ensemble_regressor(
        x,
        y,
        w,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    rospy.loginfo("Predicting residuals and entropy...")
    pred_residual, entropy = predict_with_entropy(model, x, cam_unroll["r"][cam_idx])

    s_hat = cam_unroll["s"][cam_idx] + pred_residual[:, 0]
    theta_hat = wrap_angle(cam_unroll["theta"][cam_idx] + pred_residual[:, 1])
    r_hat = cam_unroll["r"][cam_idx] + pred_residual[:, 2]

    pred_points = inverse_map_points_straight(s_hat, theta_hat, r_hat, center_mean, center_dir, u0, v0)
    refined_points = geometry_refine(
        pred_points,
        lidar[lid_idx],
        lidar_normals[lid_idx],
        lambda_n=lambda_n,
        lambda_t=lambda_t,
    )

    post_residual = np.linalg.norm(refined_points - lidar[lid_idx], axis=1)

    tau = np.quantile(entropy, entropy_quantile)
    keep = entropy <= tau
    rospy.loginfo(f"Entropy threshold: {tau:.6f}")
    rospy.loginfo(f"Retained prediction ratio: {keep.mean():.4f}")

    rospy.loginfo("Saving matched input and reference clouds...")
    matched_cam_input = cam[cam_idx]
    matched_lidar_ref = lidar[lid_idx]

    matched_cam_colors = cam_colors[cam_idx] if cam_colors is not None else None
    if lidar_colors is not None:
        matched_lidar_colors = lidar_colors[lid_idx]
    else:
        matched_lidar_colors = np.tile(np.array([[0.8, 0.8, 0.8]]), (len(matched_lidar_ref), 1))

    save_point_cloud_pcd(matched_cam_input, str(outdir_path / "camera_matched_input_rgb.pcd"), matched_cam_colors)
    save_point_cloud_pcd(matched_lidar_ref, str(outdir_path / "lidar_matched_reference.pcd"), matched_lidar_colors)

    rospy.loginfo("Saving denoised matched camera points...")
    denoised_matched = refined_points[keep]
    denoised_matched_colors = cam_colors[cam_idx][keep] if cam_colors is not None else None
    save_point_cloud_pcd(
        denoised_matched,
        str(outdir_path / "camera_denoised_matched_rgb.pcd"),
        denoised_matched_colors,
    )

    rospy.loginfo("Saving final denoised point cloud...")
    if keep_unmatched:
        unmatched = np.where(match_idx < 0)[0]
        denoised_final = np.vstack([denoised_matched, cam[unmatched]])
        if cam_colors is not None:
            denoised_final_colors = np.vstack([denoised_matched_colors, cam_colors[unmatched]])
        else:
            denoised_final_colors = None
    else:
        denoised_final = denoised_matched
        denoised_final_colors = denoised_matched_colors

    save_point_cloud_pcd(
        denoised_final,
        str(outdir_path / "camera_denoised_final_rgb.pcd"),
        denoised_final_colors,
    )

    rospy.loginfo("Processing finished.")
    rospy.loginfo(f"Output directory: {outdir_path}")
    rospy.loginfo(f"Final result: {outdir_path / 'camera_denoised_final_rgb.pcd'}")
    rospy.loginfo(f"Mean residual before correction: {pre_residual.mean():.6f}")
    rospy.loginfo(f"Mean residual after correction: {post_residual.mean():.6f}")


if __name__ == "__main__":
    main()
