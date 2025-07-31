import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

UNIT_SCALE = 0.001

def backproject_bbox_to_world(bbox, depth, K, R, t, img_size):
    """
    bbox: (x1n,y1n,x2n,y2n) 归一化坐标
    depth: Z_w，三维中心点的深度
    K, R, t: 相机内外参
    img_size: (H, W)
    返回：8 个世界坐标角点的 ndarray (8,3)
    """
    H, W = img_size
    x1n, y1n, x2n, y2n = bbox
    # 四个像素角 (u,v)
    pix = np.array([
      [x1n * W, y1n * H],
      [x2n * W, y1n * H],
      [x2n * W, y2n * H],
      [x1n * W, y2n * H]
    ])  # (4,2)
    invK = np.linalg.inv(K)
    corners_w = []
    for u,v in pix:
        ray_cam = invK @ np.array([u, v, 1.0])
        P_cam = ray_cam * depth                # 在相机系下的三维点
        # 变换到世界系： Xw = R^T * (P_cam - t)
        Xw = R.T.dot(P_cam - t.flatten())
        corners_w.append(Xw)
    # 八个角：四个 z=depth 平面，复制上下
    corners_w = np.vstack([
        corners_w,
        [c + np.array([0,0,0]) for c in corners_w]  # 同平面就重复，保证 8 点
    ])
    return corners_w  # (8,3)

def save_visualization(image, objects, save_path):
    h, w = image.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = [int(v * dim) for v, dim in zip(obj['bbox'], (w, h, w, h))]
        cx, cy = int(obj['cx'] * w), int(obj['cy'] * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        label = obj.get('label', obj['name'])
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(save_path, image)


def save_txt(objects, save_path):
    with open(save_path, 'w') as f:
        for obj in objects:
            corners = obj['corners']
            # 将 name 改为 label，这样输出才会带序号
            line = f"{obj['label']} {obj['cx']:.6f} {obj['cy']:.6f} " + " ".join(f"{c:.6f}" for c in corners)
            f.write(line + "\n")



def detect_objects(image_path, model, conf_thres=0.25, iou_thres=0.45):
    """
    对单张图像进行 YOLO 检测，返回 BGR 图像和对象信息列表。
    对于 "person" 类仅输出头部 bbox，其它类别照常输出原 bbox。
    每个对象包含: name, bbox(归一化), cx, cy, corners
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    results = model(image_path, conf=conf_thres, iou=iou_thres)[0]
    objects = []
    names = model.names
    for box in results.boxes:
        cls = int(box.cls.cpu().item())
        label = names[cls]
        # 原始像素坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        # 归一化坐标
        x1n, y1n, x2n, y2n = x1 / w, y1 / h, x2 / w, y2 / h
        if label == 'person':
            # 只取头部：取上 25% 区域，可根据需要调整比例
            head_height = (y2n - y1n) * 0.25
            hx1, hy1 = x1n, y1n
            hx2, hy2 = x2n, y1n + head_height
            cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
            corners = [hx1, hy1, hx2, hy1, hx2, hy2, hx1, hy2]
            objects.append({
                'name': 'head',
                'bbox': (hx1, hy1, hx2, hy2),
                'cx': cx,
                'cy': cy,
                'corners': corners
            })
        else:
            # 保留其它类别原 bbox
            cx, cy = (x1n + x2n) / 2, (y1n + y2n) / 2
            corners = [x1n, y1n, x2n, y1n, x2n, y2n, x1n, y2n]
            objects.append({
                'name': label,
                'bbox': (x1n, y1n, x2n, y2n),
                'cx': cx,
                'cy': cy,
                'corners': corners
            })
    return img, objects


def triangulate(pt1, pt2, P1, P2):
    pts4d = cv2.triangulatePoints(P1, P2, np.array(pt1).reshape(2,1), np.array(pt2).reshape(2,1))
    pts4d /= pts4d[3]
    return pts4d[:3].reshape(3)


if __name__ == '__main__':
    # 根目录指向包含 left/ 和 right/ 的 frames 文件夹
    base = r"E:/emotion/frames"
    calib_file = os.path.join(base, r"D:\xyl\emotion\DirectMHP-main\test_imgs\CMU\multi_ultimatum\stereo_calib.json")
    weights = r"D:\xyl\emotion\DirectMHP-main\weights\yolov5s6.pt"
    views = {"left": "left", "right": "right"}

    out_vis = os.path.join(base, "objects")
    out_txt = os.path.join(base, "txt")
    out_mat = os.path.join(base, "matched")
    os.makedirs(out_vis, exist_ok=True)
    os.makedirs(out_txt, exist_ok=True)
    os.makedirs(out_mat, exist_ok=True)

    for d in (out_vis, out_txt, out_mat): os.makedirs(d, exist_ok=True)

    # 加载标定
    with open(calib_file) as f:
        cams = json.load(f)["cameras"]
    cams_sorted = sorted(cams, key=lambda c: int(c['node']))
    camL, camR = cams_sorted[0], cams_sorted[1]

    def make_P(cam):
        K = np.array(cam['K'])
        R = np.array(cam['R'])
        t = np.array(cam['t'], dtype=np.float32).reshape(3,1)
        return K @ np.hstack((R, t))

    P1 = make_P(camL)
    P2 = make_P(camR)
    C1 = -np.array(camL['R']).T @ np.array(camL['t'], dtype=np.float32).reshape(3,1)
    C2 = -np.array(camR['R']).T @ np.array(camR['t'], dtype=np.float32).reshape(3,1)

    model = YOLO(weights)
    dets = {'left': {}, 'right': {}}
    vis_paths = {'left': {}, 'right': {}}
    conf_thres = 0.15  # 置信度阈值，
    iou_thres = 0.20  # NMS IoU 阈值

    # 仅处理每个子文件夹的第一张图片
    for cam_name, subfold in views.items():
        inp = os.path.join(base, subfold)
        for session in sorted(os.listdir(inp)):
            subdir = os.path.join(inp, session)
            if not os.path.isdir(subdir): continue
            imgs = [f for f in sorted(os.listdir(subdir)) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
            if not imgs: continue
            for fn in imgs:
                img_path = os.path.join(subdir, fn)
                img, objs = detect_objects(img_path, model, conf_thres, iou_thres)

                # 统一提取子文件夹编号作为帧 ID
                frame_id = session.rsplit('_', 1)[-1]
                vp = os.path.join(out_vis, f"{cam_name}_{frame_id}.jpg")
                tp = os.path.join(out_txt, f"{cam_name}_{frame_id}.txt")

                # 给每个类别打上序号，生成 obj['label']
                objs.sort(key=lambda x: (x['name'], x['cx']))
                counts2d = defaultdict(int)
                for o in objs:
                    counts2d[o['name']] += 1
                    o['label'] = f"{o['name']}{counts2d[o['name']]}"

                save_visualization(img.copy(), objs, vp)
                save_txt(objs, tp)
                dets[cam_name][frame_id] = objs
                vis_paths[cam_name][frame_id] = vp

    # 匹配帧并三角化（head + 其它）
    common = sorted(set(dets['left']) & set(dets['right']))
    for fid in common:
        objsL = dets['left'][fid]
        objsR = dets['right'][fid]
        world_pts = []
        # ----- 收集所有已三角化物体的深度 -----
        class_depths = defaultdict(list)  # {类别名: [所有三角化深度]}
        object_depths = {}  # {obj_label: 深度}
        # 先处理双视图均能检测到的
        for oL in objsL:
            if oL['name'] == 'head':
                continue
            matchesR = [o for o in objsR if o['name'] == oL['name']]
            if matchesR:
                h, w = cv2.imread(vis_paths['left'][fid]).shape[:2]
                u1, v1 = oL['cx'] * w, oL['cy'] * h
                oR = matchesR[0]
                u2, v2 = oR['cx'] * w, oR['cy'] * h
                Xw = triangulate((u1, v1), (u2, v2), P1, P2)
                object_depths[oL['label']] = Xw[2]
                cls = ''.join([c for c in oL['label'] if not c.isdigit()])
                class_depths[cls].append(Xw[2])

        # 1) 专门三角化 head
        headL = next((o for o in objsL if o['name'] == 'head'), None)
        headR = next((o for o in objsR if o['name'] == 'head'), None)
        if headL and headR:
            h, w = cv2.imread(vis_paths['left'][fid]).shape[:2]
            u1, v1 = headL['cx'] * w, headL['cy'] * h
            u2, v2 = headR['cx'] * w, headR['cy'] * h
            head3D = triangulate((u1, v1), (u2, v2), P1, P2)
            world_pts.append({
                'name': headL['label'],
                 'center': head3D,
                 'bbox_l': headL['bbox']
            })

        # 2) 三角化：先双目匹配，其次单目回投补全左右漏检
        # a) 先处理左视图检测出的所有物体
        for oL in objsL:
            if oL['name'] == 'head':
                continue
            h, w = cv2.imread(vis_paths['left'][fid]).shape[:2]
            u1, v1 = oL['cx'] * w, oL['cy'] * h

            matchesR = [o for o in objsR if o['name'] == oL['name']]
            if matchesR:
                # 双目三角化
                oR = matchesR[0]
                u2, v2 = oR['cx'] * w, oR['cy'] * h
                Xw = triangulate((u1, v1), (u2, v2), P1, P2)
            else:
                # 单视图：用同类物体均值深度，否则默认 2000mm
                cls = ''.join([c for c in oL['label'] if not c.isdigit()])
                if class_depths[cls]:
                    Z_est = np.mean(class_depths[cls])
                else:
                    Z_est = 2000.0  # 合理默认值 mm
                invK = np.linalg.inv(np.array(camL['K'], dtype=float))
                R_L = np.array(camL['R'], dtype=float)
                t_L = np.array(camL['t'], dtype=float).reshape(3)
                uv1 = np.array([u1, v1, 1.0])
                P_cam = invK.dot(uv1) * Z_est
                Xw = R_L.T.dot(P_cam - t_L)
            world_pts.append({
                'name': oL['label'],
                'center': Xw,
                'bbox_l': oL['bbox']
            })

        # b) 再处理只在右视图检测到、左视图漏检的物体
        for oR in objsR:
            if oR['name'] == 'head':
                continue
            if any(oL['name'] == oR['name'] for oL in objsL):
                continue
            h2, w2 = cv2.imread(vis_paths['right'][fid]).shape[:2]
            u2, v2 = oR['cx'] * w2, oR['cy'] * h2

            # 用同类物体均值深度，否则默认 2000mm
            cls = ''.join([c for c in oR['label'] if not c.isdigit()])
            if class_depths[cls]:
                Z_est = np.mean(class_depths[cls])
            else:
                Z_est = 2000.0  # 合理默认值 mm
            invK2 = np.linalg.inv(np.array(camR['K'], dtype=float))
            R_R = np.array(camR['R'], dtype=float)
            t_R = np.array(camR['t'], dtype=float).reshape(3)
            uv2 = np.array([u2, v2, 1.0])
            P_cam2 = invK2.dot(uv2) * Z_est
            Xw2 = R_R.T.dot(P_cam2 - t_R)
            world_pts.append({
                'name': oR['label'],
                'center': Xw2,
                'bbox_l': oR['bbox']
            })

        # 写入 TXT
        world_txt = os.path.join(out_mat, f"{fid}_3d.txt")
        with open(world_txt, "w") as f:
            f.write("# name  x1 y1 z1  x2 y2 z2  ... x8 y8 z8\n")
            for obj in world_pts:
                name = obj["name"]
                depth = obj["center"][2]  # Z 深度(米)
                # 反投 8 角
                corners_w = backproject_bbox_to_world(
                    obj["bbox_l"], depth / UNIT_SCALE,  # 若无缩放把 /UNIT_SCALE 去掉
                    np.array(camL["K"], float),
                    np.array(camL["R"], float),
                    np.array(camL["t"], float),
                    img_size=(h, w)
                ) * UNIT_SCALE  # 全部转换到米


                # 展平成 24 个数字
                coords = " ".join(f"{p:.6f}" for p in corners_w.flatten())
                f.write(f"{name} {coords}\n")

        print(f"[Frame {fid}] Saved 3D positions with indexed labels to {world_txt}")

        # 可视化 3D 散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*C1.flatten(), marker='^', s=80, label='CamL')
        ax.scatter(*C2.flatten(), marker='^', s=80, label='CamR')
        counts_plot = defaultdict(int)
        for obj in world_pts:
            name = obj['name']
            X = obj['center']  # 三维中心点
            counts_plot[name] += 1
            indexed_name = f"{name}{counts_plot[name]}"
            ax.scatter(*X, marker='o', s=50)
            ax.text(X[0], X[1], X[2], indexed_name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plot_path = os.path.join(out_mat, f"{fid}_3d.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
