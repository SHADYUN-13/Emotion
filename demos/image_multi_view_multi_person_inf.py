#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import math
from math import cos, sin
from pathlib import Path
import time

import numpy as np
import torch
import cv2
import yaml
import matplotlib.pyplot as plt

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from scipy.optimize import linear_sum_assignment
from pathlib import Path

CROP_DIR = Path("E:/emotion/frames/crop")
CROP_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Crop directory: {CROP_DIR.resolve()} exists → {CROP_DIR.exists()}")

def load_video_object_boxes(matched_dir):
    """
    支持两种行格式：
      ① name cx cy cz minX minY minZ maxX maxY maxZ          (10 字段)
      ② name x1 y1 z1 ... x8 y8 z8                           (25 字段)
    返回 { vid : { name : (bmin, bmax) } }
    """
    video_boxes = {}
    for txt in Path(matched_dir).glob("*_3d.txt"):
        vid   = txt.stem.split("_")[0]
        boxes = {}
        for ln in open(txt):
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split()
            name  = parts[0]
            try:
                nums = list(map(float, parts[1:]))
            except ValueError:
                print(f"[WARN] 跳过无法解析的行: {ln[:40]}...")
                continue

            if len(nums) == 9:  # name + 9 = 10 字段
                bmin = np.array(nums[3:6]);
                bmax = np.array(nums[6:9])
            elif len(nums) == 24:  # name + 24 = 25 字段
                corners = np.array(nums).reshape(8, 3)
                bmin, bmax = corners.min(0), corners.max(0)
            else:
                print(f"[WARN] 行数字段数量异常 ({len(nums)}): {ln[:40]}...")
                continue

            boxes[name] = (bmin, bmax)
        video_boxes[vid] = boxes
    print("[DEBUG] VIDEO_BOXES keys =", list(video_boxes.keys()))
    return video_boxes




def ray_aabb_intersection(origin, direction, box_min, box_max, eps=1e-6):
    """
    Robust Slab 算法：射线(origin + t*direction) 与 AABB(box_min, box_max) 相交检测
    返回 t_enter >= 0 的最小交点，若无交则返回 None
    """
    t_enter = -float("inf")
    t_exit  =  float("inf")

    for i in range(3):
        o = origin[i]
        d = direction[i]
        b0 = box_min[i]
        b1 = box_max[i]

        if abs(d) < eps:
            # 射线在该轴上平行，若起点不在 slab 内则无交
            if o < b0 or o > b1:
                return None
            # else 在 slab 内，跳过这一轴
            continue

        # 计算 t0, t1，并确保 t0 <= t1
        t0 = (b0 - o) / d
        t1 = (b1 - o) / d
        if t0 > t1:
            t0, t1 = t1, t0

        t_enter = max(t_enter, t0)
        t_exit  = min(t_exit,  t1)

        # 如果某轴上已无重叠，则无交
        if t_enter > t_exit:
            return None

    # 最终判定：交点必须在射线前方
    if t_exit < 0:
        return None
    return max(t_enter, 0.0)


def ray_sphere_intersection(origin, direction, center, radius):
    L = center - origin
    t_ca = np.dot(L, direction)
    if t_ca < 0:
        return None
    d2 = np.dot(L, L) - t_ca ** 2
    if d2 > radius ** 2:
        return None
    thc = math.sqrt(radius ** 2 - d2)
    t0 = t_ca - thc
    t1 = t_ca + thc
    if t0 > 0:
        return t0
    elif t1 > 0:
        return t1
    else:
        return None


def ray_triangle_intersection(orig, dir, v0, v1, v2, eps=1e-6):
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = edge1.dot(h)
    if abs(a) < eps:
        return None
    f = 1.0 / a
    s = orig - v0
    u = f * s.dot(h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * dir.dot(q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * edge2.dot(q)
    return t if t > eps else None


def load_calibration(path):
    data = json.load(open(path, 'r'))
    cams = {}
    for cam in data.get('cameras', []):
        if cam.get('type', '').lower() != 'hd':
            continue
        cams[cam['name']] = {
            'K': np.array(cam['K'], float),
            'R': np.array(cam['R'], float),
            't': np.array(cam['t'], float).flatten()
        }
    return cams


def compute_center(R, t):
    return -R.T.dot(t)


def build_projection(K, R, t):
    return K @ np.hstack((R, t.reshape(3, 1)))


def multi_triangulate(projections, pts2d_list):
    # pts2d_list: list of (u,v) per cam
    A = []
    for P, (u, v) in zip(projections, pts2d_list):
        P1, P2, P3 = P[0], P[1], P[2]
        A.append(u * P3 - P1)
        A.append(v * P3 - P2)
    A = np.vstack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def euler_to_gaze_vector(yaw, pitch):
    y = math.radians(yaw)
    p = math.radians(pitch)
    x = math.sin(y) * math.cos(p)
    yy = -math.sin(p)
    z = math.cos(y) * math.cos(p)
    vec = np.array([x, yy, z], dtype=float)
    return vec / np.linalg.norm(vec)


def transform_to_world(g_cam, R_cam):
    return R_cam.T.dot(g_cam)


def average_gaze_world(g_list, weights=None):
    if weights is None:
        weights = [1.0] * len(g_list)
    S = sum(w * g for w, g in zip(weights, g_list))
    return S / np.linalg.norm(S)

def process_detections(cam_id, det, args, cams):
    """
    cam_path: 单张图的文件路径 (str or Path)
    det:       Tensor shape (N,9)，每行[x1,y1,x2,y2,conf,cls,pitch_norm,yaw_norm,roll_norm]
    args:      包含 conf/iou 阈值等，但此处不直接用
    cams:      标定字典，如 cams[cam_id] = {'K':…, 'R':…, 't':…}
    Returns:   list of tuples for each detection:
               (cam_id, C_cam, P, pt2d, g_w, score)
    """
    # 1. 从文件名提取 cam_id，例如 "00_16_clip_frame_017.jpg" → "00_16"
    # stem   = Path(cam_path).stem
    # cam_id = "_".join(stem.split("_", 2)[:2])

    # 2. 取出相机内外参
    cam = cams[cam_id]
    K = cam['K']       # 3×3 内参
    R = cam['R']       # 3×3 旋转矩阵
    t = cam['t'].reshape(3, 1)  # 3×1 平移向量

    # 相机中心 C_cam = -R^T * t
    C_cam = (-R.T @ t).flatten()  # shape (3,)

    # 投影矩阵 P = K [R | t]
    P = K @ np.hstack((R, t))     # shape (3,4)

    results = []
    # 3. 遍历每个检测框
    for row in det.cpu().numpy():  # shape (9,)
        x1, y1, x2, y2, score = row[:5]
        pitch_norm, yaw_norm = row[6], row[7]

        # 3.1 计算 2D 图像坐标点 (框中心)
        pt2d = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # 3.2 将归一化的 pitch, yaw 转为弧度
        pitch = (pitch_norm - 0.5) * np.pi       # [-π/2, π/2]
        yaw   = (yaw_norm   - 0.5) * 2 * np.pi   # [-π, π]

        # 3.3 在相机坐标系下构造视线向量 g_cam
        #    假设：z 向前、x 向右、y 向下
        g_cam = np.array([
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch) * np.cos(yaw),
            np.cos(yaw) * np.cos(pitch)
        ])

        # 3.4 把视线向量从相机系 → 世界系： g_w = R^T * g_cam
        g_w = R.T @ g_cam  # shape (3,)
        g_w = -g_w
        # 3.5 收集结果
        results.append((C_cam, P, pt2d, g_w, score))

    return results


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to meet `new_shape`, keeping aspect ratio.
    """
    shape = img.shape[:2]  # current shape: (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) and compute padding
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2
    dh //= 2

    # Resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # Pad
    top, bottom = dh, new_shape[0] - new_unpad[1] - dh
    left, right = dw, new_shape[1] - new_unpad[0] - dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img

def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., thickness=2):
    """在 2D 图像上画头部三轴（X红、Y绿、Z蓝）及朝向线（青黄）。"""
    p = math.radians(pitch); y = -math.radians(yaw); r = math.radians(roll)
    if tdx is None or tdy is None:
        h,w = img.shape[:2]; tdx,tdy = w/2,h/2
    face_x, face_y = tdx, tdy
    # 3轴投影
    # x1 = size*(cos(y)*cos(r)) + face_x
    # y1 = size*(cos(p)*sin(r)+cos(r)*sin(p)*sin(y)) + face_y
    # x2 = size*(-cos(y)*sin(r)) + face_x
    # y2 = size*(cos(p)*cos(r)-sin(p)*sin(y)*sin(r)) + face_y
    x3 = size*sin(y) + face_x
    y3 = size*(-cos(y)*sin(p)) + face_y
    # 朝向延伸线
    scale_r=2
    endx = tdx + (x3-face_x)*scale_r
    endy = tdy + (y3-face_y)*scale_r
    cv2.line(img,(int(face_x),int(face_y)),(int(endx),int(endy)),(0,255,255),thickness)

    return img


def plot_multi_person_gaze(centers, head_dict, gaze_dict, object_boxes=None, out_path=None):
    """
    centers:   dict of {cam_id: np.array([X,Y,Z])}      # 世界坐标系下各相机中心
    head_dict: dict of {pid: np.array([X,Y,Z])}        # 每个人的三维头部中心
    gaze_dict: dict of {pid: np.array([vx,vy,vz])}     # 每个人的世界坐标系视线单位向量
    out_path:  str or None                            # 保存路径，不传则 plt.show()
    """

    def to_plot(pt):
        # 世界系 (X,Y,Z) -> 绘图系 (x, y, z) 对应 (X, Z, -Y)
        x, y, z = pt
        return x, z, -y

    fig = plt.figure(figsize=(10,10))
    ax  = fig.add_subplot(111, projection='3d')

    # 算一个合理的箭头长度（所有点的最大范围 * 比例）
    all_pts = np.vstack(list(centers.values()) + list(head_dict.values()))
    arrow_len = np.max(np.abs(all_pts)) * 0.3

    # 1) 绘制相机中心
    first_cam = True
    for cam_id, C in centers.items():
        x, y, z = to_plot(C)
        ax.scatter(x, y, z, c='k', marker='^', s=50,
                   label='Camera' if first_cam else None)
        ax.text(x, y, z, cam_id, fontsize=8)
        first_cam = False

    # 2) 绘制每个人的头部和视线
    colors = ['r','g','b','c','m','y']
    for pid, head3D in head_dict.items():
        clr = colors[pid % len(colors)]
        hx, hy, hz = to_plot(head3D)

        # 头部中心
        ax.scatter(hx, hy, hz, c=clr, marker='*', s=100,
                   label=f'Head {pid}')
        ax.text(hx, hy, hz, f'Head {pid}', fontsize=8, color=clr)

        # 各相机到头部的虚线
        for C in centers.values():
            cx, cy, cz = to_plot(C)
            ax.plot([cx, hx], [cy, hy], [cz, hz],
                    c=clr, linestyle='--', linewidth=1)

        # 视线箭头
        gv = gaze_dict[pid]
        gv = gv / np.linalg.norm(gv)
        vx, vy, vz = to_plot(gv)
        ax.quiver(hx, hy, hz,
                  vx*arrow_len, vy*arrow_len, vz*arrow_len,
                  color=clr, arrow_length_ratio=0.2,
                  linewidth=2, label=f'Gaze {pid}')

    if object_boxes:
        first_obj = True
        for obj_name, corners3d in object_boxes.items():
            if obj_name.startswith('head'):
                continue
            center_w = np.mean(corners3d, axis=0)
            ox, oy, oz = to_plot(center_w)
            # 用方块标记物体
            ax.scatter(ox, oy, oz,
                       c='k', marker='s', s=50,
                       label='Object' if first_obj else None)
            ax.text(ox, oy, oz, obj_name, fontsize=8, color='k')
            first_obj = False

    # 3) 坐标轴参考 (原点 + 三色轴线)
    L = np.max(np.abs(all_pts))
    # X 轴 (红), Y 轴 (绿), Z 轴 (蓝)
    ax.quiver(0,0,0, L,0,0, color='gray', lw=1.5 , arrow_length_ratio=0.1); ax.text(L,0,0,'X',color='gray')
    ax.quiver(0,0,0, 0,L,0, color='gray', lw=1.5 , arrow_length_ratio=0.1); ax.text(0,L,0,'Y',color='gray')
    ax.quiver(0,0,0, 0,0,L, color='gray', lw=1.5 , arrow_length_ratio=0.1); ax.text(0,0,L,'Z',color='gray')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Multi-Person Gaze')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1))
    ax.view_init(elev=45)

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def match_two_views(P1, pts1, P2, pts2):
    """
    对视角1和视角2的2D点做匹配：
    - P1, P2: 3×4 投影矩阵
    - pts1: [(u1,v1), ...] 视角1的2D中心列表
    - pts2: [(u2,v2), ...] 视角2的2D中心列表
    返回 [(i,j), ...] 最优匹配索引
    """
    M, N = len(pts1), len(pts2)
    C = np.full((M, N), 1e6)
    for i, (u1, v1) in enumerate(pts1):
        for j, (u2, v2) in enumerate(pts2):
            # 三角测量一个候选 3D 点
            X = multi_triangulate([P1, P2], [np.array([[u1],[v1]]), np.array([[u2],[v2]])])
            # 重投影回两视角
            def reproj(P, X):
                x, y, w = (P @ np.hstack((X,1))).flatten()
                return x/w, y/w
            uh1, vh1 = reproj(P1, X)
            uh2, vh2 = reproj(P2, X)
            # 计算重投影误差
            C[i,j] = abs(u1-uh1) + abs(v1-vh1) + abs(u2-uh2) + abs(v2-vh2)
    row_ind, col_ind = linear_sum_assignment(C)
    return list(zip(row_ind, col_ind))

def visualize_and_save(im0, det, imgsz, save_path, id_map=None):
    """
    im0: 原始 BGR 图 (H, W, 3)
    det: 单张图的 detections, Tensor of shape (N, 9)
    imgsz: 推理时 letterbox 后的边长, e.g. 1280
    save_path: 最终保存路径 (Path or str)
    """
    H_pad, W_pad = imgsz, imgsz
    # 1) 把 pad 后的坐标映射回原图
    boxes = scale_coords(
        (H_pad, W_pad),
        det[:, :4],
        im0.shape[:2]
    ).cpu().numpy()

    viz = im0.copy()
    for idx, (box, row) in enumerate(zip(boxes, det.cpu().numpy())):
        # 2) 箱体 & 置信度
        x1, y1, x2, y2 = map(int, box)
        conf = row[4]
        cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(viz, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if id_map is not None and idx in id_map:
            pid = id_map[idx]
            cv2.putText(viz, f"ID {pid}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 3) 视线箭头
        pitch = (row[6] - 0.5) * 180
        yaw   = (row[7] - 0.5) * 360
        viz = plot_3axis_Zaxis(
            viz,
            yaw, pitch, 0,
            tdx=(x1 + x2)//2, tdy=(y1 + y2)//2,
            size=int(max(y2 - y1, x2 - x1) * 0.8),
            thickness=2
        )

    # 4) 保存结果
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), viz)
    # print(f"Saved visualization to {save_path}")


def measure_two_camera(args):
    print("[DEBUG] >>> enter measure_two_camera")
    # 0. 设备 & 模型加载
    device = select_device(args.device)
    cams = load_calibration(args.calib)  # 加载标定参数
    angle_thres_rad = math.radians(args.angle_thres)
    matched_dir = Path(args.cam1_folder).parent.parent / "matched"
    print(f"[DEBUG] Using matched_dir = {matched_dir}")
    VIDEO_BOXES = load_video_object_boxes(matched_dir)
    model = attempt_load(args.weights, map_location=device)
    model.to(device).eval()
    # 同步一次，排除 model.load 的异步开销
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 1. 准备两路视频帧路径列表，并检查帧数一致
    cam1_paths = sorted(Path(args.cam1_folder).glob('*'))
    cam2_paths = sorted(Path(args.cam2_folder).glob('*'))
    assert len(cam1_paths) == len(cam2_paths), "两个文件夹帧数不一致"

    # 2. 预先计算输入尺寸和 NMS 的 num_angles
    imgsz = check_img_size(args.imgsz, s=int(model.stride.max()))
    num_angles = 3

    # ----- 初始化跨帧跟踪器 -----
    next_person_id = 0
    prev_positions = np.zeros((0, 3), dtype=float)  # 上一帧三维位置
    prev_ids = []  # 上一帧全局 ID 列表
    track_thresh = args.track_thresh

    # 3. 逐帧循环
    for idx, (p1, p2) in enumerate(zip(cam1_paths, cam2_paths), start=1):

        vid = Path(p1).parent.stem.split("_")[-1]
        print(f"[DEBUG] Processing video={vid}  frame idx={idx}")
        boxes = VIDEO_BOXES.get(vid, {})
        print(f"[DEBUG] vid={vid} 载入物体数={len(boxes)}")
        if not boxes:
            pass

        # ---- 读图 & 预处理 ----
        im0_1 = cv2.imread(str(p1))  # BGR, H×W×3
        im0_2 = cv2.imread(str(p2))

        # letterbox 保持长宽比 + 填充
        img1 = letterbox(im0_1, imgsz)
        img2 = letterbox(im0_2, imgsz)

        # BGR→RGB, HWC→CHW, 转为 contiguous
        img1 = img1[:, :, ::-1].transpose(2, 0, 1).copy()
        img2 = img2[:, :, ::-1].transpose(2, 0, 1).copy()
        img1 = torch.from_numpy(img1).to(device).float() / 255.0
        img2 = torch.from_numpy(img2).to(device).float() / 255.0
        batch = torch.stack([img1, img2], 0)  # (2,3,H,W)

        # ---- 推理 + NMS ----
        with torch.no_grad():
            pred = model(batch, augment=True, scales=args.scales)[0]
            dets = non_max_suppression(
                pred, args.conf_thres, args.iou_thres, num_angles = num_angles
            )

        # ---- 2D可视化 ----
        # 直接指定输出目录
        base_vis_dir = Path("E:/emotion/frames/output")
        vis_subdir = base_vis_dir / vid
        vis_subdir.mkdir(parents=True, exist_ok=True)

        cam_id1 = Path(p1).parent.parent.stem
        cam_id2 = Path(p2).parent.parent.stem
        fn1 = f"{idx:03d}_{cam_id1}_viz.jpg"
        fn2 = f"{idx:03d}_{cam_id2}_viz.jpg"
        visualize_and_save(im0_1, dets[0], imgsz, vis_subdir / fn1)
        visualize_and_save(im0_2, dets[1], imgsz, vis_subdir / fn2)

        # ---- 多视角匹配 + 3D 三角化 + 视线平均 ----
        # 1) 收集 per_cam
        per_cam = {}
        for cam_path, det in zip([p1, p2], dets):
            if det is None or len(det) == 0:
                continue
            cam_id = Path(cam_path).parent.parent.stem
            cam_res = process_detections(cam_id, det, args, cams)
            per_cam.setdefault(cam_id, []).extend(cam_res)


        # 2) 只有两台相机都有检测时再匹配
        if len(per_cam) == 2:
            cam_ids = list(per_cam.keys())
            ref_cam, src_cam = cam_ids[0], cam_ids[1]
            ref_res, src_res = per_cam[ref_cam], per_cam[src_cam]

            # 构造投影矩阵
            P_ref = cams[ref_cam]['K'] @ np.hstack((cams[ref_cam]['R'], cams[ref_cam]['t'].reshape(3, 1)))
            P_src = cams[src_cam]['K'] @ np.hstack((cams[src_cam]['R'], cams[src_cam]['t'].reshape(3, 1)))

            # 提取 2D 点
            pts_ref = [r[2] for r in ref_res]
            pts_src = [r[2] for r in src_res]

            # 3) 两视角匹配，返回 [(i_ref, i_src), …]
            pairs = match_two_views(P_ref, pts_ref, P_src, pts_src)

            # 4) 三角化 & 视线平均
            head_dict = {}
            gaze_dict = {}
            for i_ref, i_src in pairs:
                C_ref, proj_ref, pt2d_ref, gw_ref, score_ref = ref_res[i_ref]
                C_src, proj_src, pt2d_src, gw_src, score_src = src_res[i_src]

                head3D = multi_triangulate([proj_ref, proj_src], [pt2d_ref, pt2d_src])
                gaze3D = average_gaze_world([gw_ref, gw_src], weights=[score_ref, score_src])
                print(f"Person {i_ref}: 3D position = {head3D}, 3D gaze dir = {gaze3D}")
                head_dict[i_ref] = head3D
                gaze_dict[i_ref] = gaze3D

            # 5) 提取相机中心
            centers = {
                cam_id: cam_res_list[0][0]  # cam_res_list[0][1] 就是 C_cam
                for cam_id, cam_res_list in per_cam.items()
            }

            # ------ 跨帧 ID 分配 ------
            local_idxs = list(head_dict.keys())
            curr_positions = np.stack([head_dict[i] for i in local_idxs], axis=0)

            curr_id_map = {}  # local_idx -> persistent ID

            # 如果有上一帧，做距离匹配
            if prev_positions.shape[0] > 0:
                cost = np.linalg.norm(prev_positions[:, None, :] - curr_positions[None, :, :], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < track_thresh:
                        curr_id_map[local_idxs[c]] = prev_ids[r]

            # 新出现的人分配新的 ID
            for i in local_idxs:
                if i not in curr_id_map:
                    curr_id_map[i] = next_person_id
                    next_person_id += 1

            # 更新上一帧数据
            prev_positions = np.stack([head_dict[i] for i in local_idxs], axis=0)
            prev_ids = [curr_id_map[i] for i in local_idxs]

            # 构造全局 dict，用于 3D 可视化
            persistent_head = {curr_id_map[i]: head_dict[i] for i in local_idxs}
            persistent_gaze = {curr_id_map[i]: gaze_dict[i] for i in local_idxs}

            # 打印
            for i in local_idxs:
                pid = curr_id_map[i]
                pos = head_dict[i]
                gaz = gaze_dict[i]
                print(f"[Frame {idx}] Person {pid}: Pos={pos}, Gaze={gaz}")

            # 3D 绘图并保存
            matched_fn = f"{idx:03d}_matched.png"
            boxes = VIDEO_BOXES.get(vid, {})  # 本帧对应视频的物体 AABB
            plot_multi_person_gaze(centers, persistent_head, persistent_gaze,
                                   object_boxes=boxes,
                                   out_path=vis_subdir / matched_fn)
            boxes = VIDEO_BOXES.get(vid, {})
            print(f"[DEBUG] Found {len(boxes)} static object boxes for vid={vid}")
            angle_limit = angle_thres_rad  # 弧度阈值
            dist_limit = getattr(args, 'dist_thres', 1e9)

            for pid, origin in persistent_head.items():
                direction = persistent_gaze[pid]
                direction = direction / np.linalg.norm(direction)

                best_obj, best_angle, best_dist = None, float('inf'), None

                for obj_name, (box_min, box_max) in boxes.items():
                    if obj_name.startswith('head'):
                        continue

                    # ---------- 生成 8 个角 ----------
                    xs = [box_min[0], box_max[0]]
                    ys = [box_min[1], box_max[1]]
                    zs = [box_min[2], box_max[2]]
                    corners8 = np.array([[x, y, z] for x in xs for y in ys for z in zs])  # (8,3)

                    # ---------- 逐角点夹角 ----------
                    vecs = corners8 - origin  # (8,3)
                    dists = np.linalg.norm(vecs, axis=1)  # (8,)
                    vecs_n = vecs / (dists[:, None] + 1e-12)  # 单位化
                    dots = np.clip(vecs_n.dot(direction), -1.0, 1.0)
                    angs = np.arccos(dots)  # (8,) 弧度

                    min_idx = np.argmin(angs)
                    ang = angs[min_idx]
                    dist = dists[min_idx]

                    print(f"[ANGLE] pid={pid} -> {obj_name:<12} "
                          f"minAng={math.degrees(ang):5.1f}°  dist={dist:4.2f}m")


                    # ---------- 阈值 + 更新最优 ----------
                    if ang < best_angle:
                        best_obj, best_angle, best_dist = obj_name, ang, dist
                        best_corner = corners8[min_idx]  # 用于可视化

                if best_obj is None:
                    print(f"[MISS] pid={pid} 当前帧无可用物体")
                    continue

                print(f"[HIT ] pid={pid} → {best_obj}  "
                      f"minAngle={math.degrees(best_angle):.1f}°  dist={best_dist:.2f}m")

                # 1) 取得盒 8 角顶点
                box_min, box_max = boxes[best_obj]
                xs = [box_min[0], box_max[0]]
                ys = [box_min[1], box_max[1]]
                zs = [box_min[2], box_max[2]]
                corners8 = np.array([[x, y, z] for x in xs for y in ys for z in zs])  # (8,3)

                # 2) 取正确参考相机
                im_ref = im0_1 if ref_cam == cam_id1 else im0_2
                h0, w0 = im_ref.shape[:2]
                P_ref = cams[ref_cam]['K'] @ np.hstack((cams[ref_cam]['R'],
                                                        cams[ref_cam]['t'].reshape(3, 1)))

                # 3) 投影到像素坐标
                homo = np.hstack((corners8, np.ones((8, 1))))
                pts_cam = (P_ref @ homo.T).T
                pts_pix = pts_cam[:, :2] / pts_cam[:, 2:3]
                x1, y1 = np.floor(pts_pix.min(axis=0)).astype(int)
                x2, y2 = np.ceil(pts_pix.max(axis=0)).astype(int)
                x1, x2 = max(0, x1), min(w0, x2)
                y1, y2 = max(0, y1), min(h0, y2)

                # 4) 裁剪并保存
                crop = im_ref[y1:y2, x1:x2]

                # 新增：自适应放大到224×224
                if crop.size > 0:
                    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)

                crop_subdir = CROP_DIR / vid
                crop_subdir.mkdir(parents=True, exist_ok=True)
                save_path = crop_subdir / f"{vid}_frame{idx:03d}_p{pid}_{best_obj}.jpg"
                cv2.imwrite(str(save_path), crop)

                print(f"[DEBUG] saved crop for {best_obj} → {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure two-camera inference latency")
    parser.add_argument('--device',       default='cpu',        help='cuda device, e.g. 0 or cpu')
    parser.add_argument('--weights',      required=True,        help='model.pt path')
    parser.add_argument('--calib',        required=True,        help='calibration file path')
    parser.add_argument('--cam1_folder', required=True, help='path to folder of camera 1 frames')
    parser.add_argument('--cam2_folder', required=True, help='path to folder of camera 2 frames')
    parser.add_argument('--imgsz',        type=int, default=640, help='inference image size')
    parser.add_argument('--conf-thres',   type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres',    type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--scales',       nargs='+', type=float, default=[1.0], help='multi-scale list')
    parser.add_argument('--track-thresh',type=float,default=200.0,help='跨帧匹配阈值，世界坐标系距离 (默认200)')
    parser.add_argument('--angle-thres', type=float, default=35.0,help='视线与物体顶点最小夹角阈值（度）')
    parser.add_argument('--dist-thres', type=float, default=4.0,help='最大交互距离 (米)，0=无限制')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    measure_two_camera(args)

