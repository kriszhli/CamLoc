import numpy as np
import os


estimated_poses_file = 'estimated_poses_with_gt_alignment.txt'
ground_truth_dir = 'fire/map'
pose_file = 'frame-{0:06d}.pose.txt'

# Parse the estimated 4x4 pose matrix
def parse_pose_block(lines):
    pose = []
    for line in lines:
        pose.append([float(x) for x in line.strip().split()])
    return np.array(pose)

# Rotational error; see paper
def rotation_error(R_est, R_gt):
    R_error = R_est @ R_gt.T
    trace = np.trace(R_error)
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)

# Translation error; see paper
def translation_error(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)

with open(estimated_poses_file, 'r') as f:
    lines = f.readlines()

errors = []
num_poses = 0
i = 0

while i < len(lines):
    if lines[i].strip() == "":
        i += 1
        continue

    frame_label = lines[i].strip().rstrip(':')
    i += 1
    pose_lines = []
    for _ in range(4):
        if i < len(lines) and lines[i].strip():
            pose_lines.append(lines[i])
            i += 1
    estimated_pose = parse_pose_block(pose_lines)
    frame_idx = int(frame_label.split('-')[1].split('.')[0])

    # Ground truths
    gt_pose_filename = pose_file.format(frame_idx)
    gt_pose_path = os.path.join(ground_truth_dir, gt_pose_filename)
    if not os.path.exists(gt_pose_path):
        print(f"Ground truth pose not found for frame {frame_idx}")
        continue

    gt_pose = np.loadtxt(gt_pose_path)

    R_est = estimated_pose[:3, :3]
    t_est = estimated_pose[:3, 3]
    R_gt = gt_pose[:3, :3]
    t_gt = gt_pose[:3, 3]

    # Compute errors
    rot_err = rotation_error(R_est, R_gt)
    trans_err = translation_error(t_est, t_gt)

    errors.append((rot_err, trans_err))
    num_poses += 1

# Mean errors
mean_rot_err = np.mean([e[0] for e in errors]) if errors else 0
mean_trans_err = np.mean([e[1] for e in errors]) if errors else 0

# Print results
print(f"Evaluated {num_poses} poses")
print(f"Mean Rotation Error: {mean_rot_err:.2f}")
print(f"Mean Translation Error: {mean_trans_err:.2f}")
