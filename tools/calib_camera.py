#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目棋盘格标定 → 输出 5 参数畸变 JSON
目录结构：
    calib/
      left/   imgL_000.jpg ...
      right/  imgR_000.jpg ...

棋盘规格：PATTERN 内角点 (cols, rows)
方格边长：SQUARE (mm) —— 可改为实际尺寸
"""

import cv2
import numpy as np
import glob, json
from pathlib import Path

# ---------- 用户可修改 ----------
LEFT_DIR   = Path(r"D:\xyl\emotion\capture\capture\left")
RIGHT_DIR  = Path(r"D:\xyl\emotion\capture\capture\right")
PATTERN    = (11, 8)          # 棋盘内角点 (cols, rows)
SQUARE     = 50.0             # 方格边长 (mm)
CRITERIA   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              30, 1e-4)
FLAGS_CB   = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
# ---------------------------------

# 1) 棋盘 3D 坐标 (0,0,0)...(n-1,m-1,0)
objp = np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
objp *= SQUARE

objpoints, imgpoints_L, imgpoints_R = [], [], []

left_imgs  = sorted(glob.glob(str(LEFT_DIR  / "*.*")))
right_imgs = sorted(glob.glob(str(RIGHT_DIR / "*.*")))
assert left_imgs and len(left_imgs) == len(right_imgs), \
       "左右图片数量不一致或为空！"

# 2) 逐对检测角点
for lp, rp in zip(left_imgs, right_imgs):
    imgL, imgR = cv2.imread(lp), cv2.imread(rp)
    grayL, grayR = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), \
                   cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    okL, cornersL = cv2.findChessboardCorners(grayL, PATTERN, FLAGS_CB)
    okR, cornersR = cv2.findChessboardCorners(grayR, PATTERN, FLAGS_CB)

    if not okL or not okR:
        miss = []
        if not okL: miss.append(f"L:{Path(lp).name}")
        if not okR: miss.append(f"R:{Path(rp).name}")
        print("[WARN]", " / ".join(miss), "棋盘检测失败，跳过")
        continue

    cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), CRITERIA)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), CRITERIA)

    imgpoints_L.append(cornersL)
    imgpoints_R.append(cornersR)
    objpoints.append(objp)

print(f"[INFO] 成功匹配双视角棋盘 {len(objpoints)} 组")
h, w = cv2.imread(left_imgs[0]).shape[:2]

# 3) 单目内参 (5 参数畸变)
retL, K, dist, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_L, (w, h),
    None, None, flags=0, criteria=CRITERIA)    # flags=0 → 仅 k1,k2,p1,p2,k3
retR, _,     _, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_R, (w, h),
    None, None, flags=0, criteria=CRITERIA)

print(f"[INFO] RMS-left  = {retL:.4f}  RMS-right = {retR:.4f}")

# 4) 双目外参 (固定内参)
_, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
    objpoints, imgpoints_L, imgpoints_R,
    K, dist, K, dist, (w, h),
    criteria=CRITERIA, flags=cv2.CALIB_FIX_INTRINSIC)

# 只保留前 5 个畸变系数
dist5 = dist.flatten()[:5]

# 5) 组织并写 JSON
def cam_block(name, Rmat, tvec, panel, node):
    return {
        "name": name,
        "resolution": [w, h],
        "K": K.tolist(),
        "distCoef": dist5.tolist(),
        "R": Rmat.tolist(),
        "t": tvec.reshape(3,1).tolist(),
        "panel": panel,
        "node":  node
    }

data = {
    "cameras": [
        cam_block("left",  np.eye(3), np.zeros(3), panel=1, node=1),
        cam_block("right", R,        t.flatten(), panel=1, node=2)
    ]
}

with open("stereo_calib.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("[INFO] 已写入 stereo_calib.json (5 参数畸变)")
