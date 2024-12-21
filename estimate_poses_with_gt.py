import numpy as np
import cv2
import os

matches_dir = 'matches'
ground_truth_dir = 'fire/map'
pose_file_template = 'frame-{0:06d}.pose.txt'
output_poses_file = 'estimated_poses_with_gt_alignment.txt'
intrinsics_file = 'fire/intrinsics.yml' # Downloaded from https://github.com/zinsmatt/7-Scenes-Calibration

fs = cv2.FileStorage(intrinsics_file, cv2.FILE_STORAGE_READ)
K = fs.getNode("K").mat()
fs.release()

# Load ground truth pose for a frame
def load_ground_truth_pose(frame_idx):
    gt_pose_filename = pose_file_template.format(frame_idx)
    gt_pose_path = os.path.join(ground_truth_dir, gt_pose_filename)
    if not os.path.exists(gt_pose_path):
        print(f"Ground truth pose not found for frame {frame_idx}")
        return None
    return np.loadtxt(gt_pose_path)

# Iterate over all matches
with open(output_poses_file, 'w') as out_f:
    for match_filename in os.listdir(matches_dir):
        if not match_filename.endswith('_matches.npz'):
            continue

        match_path = os.path.join(matches_dir, match_filename)
        npz = np.load(match_path)
        keypoints0 = npz['keypoints0']
        keypoints1 = npz['keypoints1']
        matches = npz['matches']

        matched_indices = matches > -1
        pts_map = keypoints0[matched_indices]
        pts_query = keypoints1[matches[matched_indices]]

        if len(pts_map) < 5:
            print(f"Not enough matches for {match_filename}")
            continue

        frame_name = match_filename.split('_matches.npz')[0]
        frame_idx = int(frame_name.split('-')[-1])

        gt_pose = load_ground_truth_pose(frame_idx)
        if gt_pose is None:
            continue

        R_gt = gt_pose[:3, :3]
        t_gt = gt_pose[:3, 3]

        pts_map_3d = []
        pts_query_2d = []

        for i in range(len(pts_map)):
            x_map, y_map = pts_map[i]
            z_map = 1.0
            X_map = (x_map - K[0, 2]) * z_map / K[0, 0]
            Y_map = (y_map - K[1, 2]) * z_map / K[1, 1]
            pts_map_3d.append(R_gt @ [X_map, Y_map, z_map] + t_gt)  # Apply pre-alignment
            pts_query_2d.append(pts_query[i])

        pts_map_3d = np.array(pts_map_3d, dtype=np.float32)
        pts_query_2d = np.array(pts_query_2d, dtype=np.float32)

        if len(pts_map_3d) < 5:
            print(f"Not enough valid 3D points for {match_filename}")
            continue

        # Solve PnP with pre-aligned keypoints
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_map_3d,
            pts_query_2d,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=8.0,
            confidence=0.99,
            iterationsCount=1000
        )

        if not success:
            print(f"PnP failed for {match_filename}")
            continue

        # Convert rotation vector to matrix
        R_est, _ = cv2.Rodrigues(rvec)
        t_est = tvec.flatten()

        # Construct the pose matrix
        estimated_pose = np.eye(4)
        estimated_pose[:3, :3] = R_est
        estimated_pose[:3, 3] = t_est

        # Save the pose
        out_f.write(f"{frame_name}:\n")
        np.savetxt(out_f, estimated_pose, fmt='%.6f')
        out_f.write("\n")

        print(f"Processed frame: {frame_name}")

print(f"Estimated poses have been saved to {output_poses_file}")
