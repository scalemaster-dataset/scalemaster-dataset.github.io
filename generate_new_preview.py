import numpy as np
import cv2
import os
import json
import csv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'ICRA_Final_Dataset')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'data', 'new_preview')

MAX_PLY_MB = 49  # GitHub 제한 50MB (10진수), 안전 마진 1MB
# Binary PLY: float32 x3 (12 bytes) + uint8 x3 (3 bytes) = 15 bytes/point
BYTES_PER_POINT = 15
MAX_POINTS = int(MAX_PLY_MB * 1_000_000 / BYTES_PER_POINT)  # ~3,266,666


def load_camera_matrix(camera_matrix_path):
    if not os.path.exists(camera_matrix_path):
        print(f"  Warning: camera_matrix not found, using default")
        return np.array([[1334.4711, 0.0, 962.00726],
                         [0.0, 1334.4711, 726.66614],
                         [0.0, 0.0, 1.0]], dtype=np.float32)
    try:
        K = np.loadtxt(camera_matrix_path, delimiter=',')
        if K.shape == (9,):
            K = K.reshape(3, 3)
        if K.shape != (3, 3):
            raise ValueError(f"Expected 3x3, got {K.shape}")
        print(f"  Camera: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        return K.astype(np.float32)
    except Exception as e:
        print(f"  Warning: {e}, using default camera matrix")
        return np.array([[1334.4711, 0.0, 962.00726],
                         [0.0, 1334.4711, 726.66614],
                         [0.0, 0.0, 1.0]], dtype=np.float32)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def depth_to_pointcloud(depth_image, rgb_image, K, pose,
                        confidence_image=None, confidence_threshold=0.5):
    depth_h, depth_w = depth_image.shape
    rgb_h, rgb_w = rgb_image.shape[:2]

    K_scaled = K.copy()
    if (rgb_h != depth_h) or (rgb_w != depth_w):
        rgb_image = cv2.resize(rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)
        K_scaled[0, 0] *= depth_w / rgb_w
        K_scaled[1, 1] *= depth_h / rgb_h
        K_scaled[0, 2] *= depth_w / rgb_w
        K_scaled[1, 2] *= depth_h / rgb_h

    if confidence_image is not None:
        ch, cw = confidence_image.shape[:2]
        if (ch != depth_h) or (cw != depth_w):
            confidence_image = cv2.resize(confidence_image, (depth_w, depth_h),
                                          interpolation=cv2.INTER_LINEAR)
            if len(confidence_image.shape) == 3:
                confidence_image = confidence_image[:, :, 0]
        if confidence_image.dtype == np.uint8:
            confidence_image = confidence_image.astype(np.float32) / 255.0
        elif confidence_image.dtype == np.uint16:
            confidence_image = confidence_image.astype(np.float32) / 65535.0
        else:
            confidence_image = confidence_image.astype(np.float32)
        if confidence_image.max() > 1.0:
            confidence_image /= 255.0

    fx, fy = K_scaled[0, 0], K_scaled[1, 1]
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]

    valid_mask = (depth_image > 0) & (depth_image < 50.0)
    if confidence_image is not None and confidence_threshold > 0:
        conf_mask = confidence_image >= confidence_threshold
        valid_mask = valid_mask & conf_mask
        if np.sum(valid_mask) == 0:
            valid_mask = (depth_image > 0) & (depth_image < 50.0) & (confidence_image > 0)

    u, v = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    z = depth_image[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy

    points_cam = np.stack([x, y, z], axis=1)
    colors = rgb_image[valid_mask]

    R_world = pose[:3, :3]
    t_world = pose[:3, 3]
    points_world = (R_world @ points_cam.T).T + t_world

    return points_world, colors


def load_odometry(odometry_path):
    poses = {}
    if odometry_path.endswith('.csv'):
        with open(odometry_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            fn_lower = {n.strip().lower(): n for n in fieldnames}

            frame_key = fn_lower.get('frame')
            if frame_key is None:
                for k in ['Frame', 'FRAME', 'frame_idx', 'index']:
                    if k in fieldnames:
                        frame_key = k
                        break
            if frame_key is None:
                raise ValueError(f"'frame' column not found. Columns: {fieldnames}")

            x_key  = fn_lower.get('x',  'x')
            y_key  = fn_lower.get('y',  'y')
            z_key  = fn_lower.get('z',  'z')
            qx_key = fn_lower.get('qx', 'qx')
            qy_key = fn_lower.get('qy', 'qy')
            qz_key = fn_lower.get('qz', 'qz')
            qw_key = fn_lower.get('qw', 'qw')

            for row in reader:
                try:
                    frame_idx = int(row[frame_key].strip())
                    x, y, z = float(row[x_key]), float(row[y_key]), float(row[z_key])
                    qx = float(row[qx_key]); qy = float(row[qy_key])
                    qz = float(row[qz_key]); qw = float(row[qw_key])
                    R_mat = quaternion_to_rotation_matrix(qx, qy, qz, qw)
                    pose = np.eye(4)
                    pose[:3, :3] = R_mat
                    pose[:3, 3]  = [x, y, z]
                    poses[frame_idx] = pose
                except (ValueError, KeyError) as e:
                    print(f"  Warning: skipping row: {e}")
    else:
        with open(odometry_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    frame_idx = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    R_mat = quaternion_to_rotation_matrix(qx, qy, qz, qw)
                    pose = np.eye(4)
                    pose[:3, :3] = R_mat
                    pose[:3, 3]  = [x, y, z]
                    poses[frame_idx] = pose
    return poses


def save_ply(filename, points, colors):
    """Binary little-endian PLY — 15 bytes/point, 파일 크기 예측 가능"""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # points: float32, colors: uint8 (BGR→RGB 변환)
    pts   = points.astype(np.float32)
    cols  = colors[:, ::-1].astype(np.uint8)  # BGR→RGB

    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        # 인터리브: [x, y, z, r, g, b] per point
        data = np.zeros(len(pts), dtype=[
            ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        data['x'], data['y'], data['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
        data['red'], data['green'], data['blue'] = cols[:, 0], cols[:, 1], cols[:, 2]
        data.tofile(f)


def generate_pointcloud_for_sequence(sequence_path, output_path,
                                     downsample_factor=1, confidence_threshold=0.5):
    frames_dir     = os.path.join(sequence_path, 'frames')
    depth_dir      = os.path.join(sequence_path, 'depth')
    confidence_dir = os.path.join(sequence_path, 'confidence')

    if not os.path.exists(frames_dir) or not os.path.exists(depth_dir):
        print(f"  Skip: frames or depth directory missing")
        return None

    # optimized_odometry.csv 우선, 없으면 odometry.csv fallback
    opt_odom  = os.path.join(sequence_path, 'optimized_odometry.csv')
    raw_odom  = os.path.join(sequence_path, 'odometry.csv')
    raw_odom2 = os.path.join(sequence_path, 'odometry.txt')

    if os.path.exists(opt_odom):
        odometry_path = opt_odom
        print(f"  Odometry: optimized_odometry.csv")
    elif os.path.exists(raw_odom):
        odometry_path = raw_odom
        print(f"  Odometry: odometry.csv (fallback)")
    elif os.path.exists(raw_odom2):
        odometry_path = raw_odom2
        print(f"  Odometry: odometry.txt (fallback)")
    else:
        print(f"  Skip: no odometry file found")
        return None

    K = load_camera_matrix(os.path.join(sequence_path, 'camera_matrix.csv'))
    poses = load_odometry(odometry_path)

    # Confidence
    has_confidence = False
    if os.path.exists(confidence_dir):
        conf_files = [f for f in os.listdir(confidence_dir) if f.endswith('.png')]
        has_confidence = len(conf_files) > 0
        if has_confidence:
            print(f"  Confidence: {len(conf_files)} files, threshold={confidence_threshold}")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(depth_dir)  if f.endswith('.png')])

    depth_dict = {}
    for df in depth_files:
        try:
            depth_dict[int(df.split('.')[0])] = df
        except:
            continue

    confidence_dict = {}
    if has_confidence:
        for cf in sorted([f for f in os.listdir(confidence_dir) if f.endswith('.png')]):
            try:
                confidence_dict[int(cf.split('.')[0])] = cf
            except:
                continue

    all_points, all_colors, trajectory_points = [], [], []

    print(f"  Frames: {len(frame_files)}, downsample={downsample_factor}")
    for i, frame_file in enumerate(tqdm(frame_files[::downsample_factor], desc="  Frames")):
        try:
            frame_num = int(frame_file.split('_')[1].split('.')[0])
        except:
            continue

        depth_file = depth_dict.get(frame_num) or depth_dict.get(i * downsample_factor)
        if depth_file is None:
            continue

        if frame_num not in poses:
            alt = i * downsample_factor
            if alt not in poses:
                continue
            frame_num = alt

        pose = poses[frame_num]
        trajectory_points.append(pose[:3, 3].tolist())

        rgb_image   = cv2.imread(os.path.join(frames_dir, frame_file))
        depth_image = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_UNCHANGED)

        if rgb_image is None or depth_image is None:
            continue

        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0
        elif depth_image.dtype == np.uint8:
            depth_image = depth_image.astype(np.float32) / 255.0 * 10.0
        else:
            depth_image = depth_image.astype(np.float32)

        confidence_image = None
        if has_confidence:
            cf = confidence_dict.get(frame_num) or confidence_dict.get(i * downsample_factor)
            if cf:
                confidence_image = cv2.imread(os.path.join(confidence_dir, cf), cv2.IMREAD_UNCHANGED)
                if confidence_image is not None and len(confidence_image.shape) == 3:
                    confidence_image = confidence_image[:, :, 0]

        points, colors = depth_to_pointcloud(depth_image, rgb_image, K, pose,
                                             confidence_image=confidence_image,
                                             confidence_threshold=confidence_threshold)
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)

    if not all_points:
        print(f"  Warning: no valid points generated")
        return None

    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    dist_mask = np.linalg.norm(combined_points, axis=1) < 1000.0
    combined_points = combined_points[dist_mask]
    combined_colors = combined_colors[dist_mask]

    print(f"  Points before cap: {len(combined_points)}")

    # 50MB 제한: MAX_POINTS 초과 시 uniform random subsampling
    if len(combined_points) > MAX_POINTS:
        idx = np.random.choice(len(combined_points), MAX_POINTS, replace=False)
        idx.sort()
        combined_points = combined_points[idx]
        combined_colors = combined_colors[idx]
        print(f"  Subsampled to {MAX_POINTS} points ({MAX_PLY_MB}MB limit)")

    est_mb = len(combined_points) * BYTES_PER_POINT / 1024 / 1024
    print(f"  Estimated file size: {est_mb:.1f} MB ({len(combined_points)} points)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_ply(output_path, combined_points, combined_colors)

    trajectory_path = output_path.replace('.ply', '_trajectory.json')
    with open(trajectory_path, 'w') as f:
        json.dump(trajectory_points, f)

    print(f"  Saved: {output_path}")
    return output_path, trajectory_path


def main():
    dataset_path = os.path.abspath(DATASET_PATH)
    output_dir   = os.path.abspath(OUTPUT_DIR)

    # 시퀀스 탐색: frames + depth + odometry 갖춘 폴더만
    sequences = []
    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        if not os.path.isdir(item_path):
            continue
        has_frames = os.path.exists(os.path.join(item_path, 'frames'))
        has_depth  = os.path.exists(os.path.join(item_path, 'depth'))
        has_odom   = (os.path.exists(os.path.join(item_path, 'optimized_odometry.csv')) or
                      os.path.exists(os.path.join(item_path, 'odometry.csv')) or
                      os.path.exists(os.path.join(item_path, 'odometry.txt')))
        if has_frames and has_depth and has_odom:
            sequences.append(item)

    print(f"Found {len(sequences)} sequences")
    print(f"Output: {output_dir}\n")

    sequence_info = []
    for seq_name in sequences:
        print(f"{'='*60}")
        print(f"Sequence: {seq_name}")
        seq_path    = os.path.join(dataset_path, seq_name)
        output_path = os.path.join(output_dir, f"{seq_name}.ply")

        result = generate_pointcloud_for_sequence(
            seq_path, output_path,
            downsample_factor=5,
            confidence_threshold=0.5
        )
        if result:
            sequence_info.append({
                'name': seq_name,
                'ply_path': f"data/new_preview/{seq_name}.ply",
                'trajectory_path': f"data/new_preview/{seq_name}_trajectory.json"
            })

    info_path = os.path.join(output_dir, 'sequences.json')
    with open(info_path, 'w') as f:
        json.dump(sequence_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {len(sequence_info)}/{len(sequences)} sequences processed")
    print(f"sequences.json -> {info_path}")


if __name__ == '__main__':
    main()
