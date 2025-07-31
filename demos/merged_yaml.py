import argparse
import yaml
import json
import numpy as np
import cv2

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def rvec_to_R(rvec):
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=float))
    return R.tolist()

def main():
    parser = argparse.ArgumentParser(
        description="合并左右相机标定参数并输出符合多摄像机规范的 JSON"
    )
    parser.add_argument('--left',         required=True,  help='左相机 calibration YAML 文件')
    parser.add_argument('--right',        required=True,  help='右相机 calibration YAML 文件')
    parser.add_argument('--output',       required=True,  help='输出 JSON 文件路径（*.json）')
    parser.add_argument('--calib-source', default='combined_calib',
                        help='calibDataSource 字段')
    parser.add_argument('--left-name',    default='01_01',
                        help='左相机 name 字段')
    parser.add_argument('--right-name',   default='01_02',
                        help='右相机 name 字段')
    parser.add_argument('--left-type',    default='vga', help='左相机 type 字段')
    parser.add_argument('--right-type',   default='vga', help='右相机 type 字段')
    parser.add_argument('--resolution',   nargs=2, type=int, default=[640,480],
                        help='图像分辨率: 宽 高')
    parser.add_argument('--left-panel',   type=int, default=1, help='左相机 panel 字段')
    parser.add_argument('--left-node',    type=int, default=1, help='左相机 node 字段')
    parser.add_argument('--right-panel',  type=int, default=1, help='右相机 panel 字段')
    parser.add_argument('--right-node',   type=int, default=2, help='右相机 node 字段')
    args = parser.parse_args()

    # 载入两份 YAML
    left  = load_yaml(args.left)
    right = load_yaml(args.right)

    # 平均内参 K
    K_avg = (
        (np.array(left['K'],  dtype=float) +
         np.array(right['K'], dtype=float)) / 2.0
    ).tolist()

    # 平均畸变系数 distCoef
    dL = np.array(left['dist'],  dtype=float).flatten()
    dR = np.array(right['dist'], dtype=float).flatten()
    dist_avg = ((dL + dR) / 2.0).tolist()

    # 提取首张图的外参
    extL = left['extrinsics'][0]
    extR = right['extrinsics'][0]

    # 旋转矩阵 R
    R_L = extL.get('R') if 'R' in extL else rvec_to_R(extL['rvec'])
    R_R = extR.get('R') if 'R' in extR else rvec_to_R(extR['rvec'])

    camL = {
        "name":       args.left_name,
        "type":       args.left_type,
        "resolution": args.resolution,
        "K":          K_avg,
        "distCoef":   dist_avg,
        "R":          R_L,
        "t":          extL['tvec'],
        "panel":      args.left_panel,
        "node":       args.left_node
    }
    camR = {
        "name":       args.right_name,
        "type":       args.right_type,
        "resolution": args.resolution,
        "K":          K_avg,
        "distCoef":   dist_avg,
        "R":          R_R,
        "t":          extR['tvec'],
        "panel":      args.right_panel,
        "node":       args.right_node
    }

    combined = {
        "calibDataSource": args.calib_source,
        "cameras":         [camL, camR]
    }

    save_json(combined, args.output)
    print(f"✅ 合并完成，输出文件：{args.output}")

if __name__ == "__main__":
    main()
