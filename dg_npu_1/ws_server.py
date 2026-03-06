import asyncio
import websockets
import psutil
import cv2
import numpy as np
import subprocess
import sys
import os
import time
import json
from pathlib import Path
import degirum as dg
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from utils import Rect, distance, get_rect_center, save_rects_to_json, load_rects_from_json, _init_logger, skeleton_map, \
aaa, OneEuroFilter, OneEuro2D, CentroidTracker,video_path,if_show_server

# ---------------- 全局变量初始化 ----------------
GAME_ROOT = os.path.dirname(os.path.abspath(__file__))
EXT_SCRIPT = {
    "1": os.path.join(GAME_ROOT, 'game1', 'ws_server.py'),
    "2": os.path.join(GAME_ROOT, 'game2', 'ws_server.py'),
    "3": os.path.join(GAME_ROOT, 'game3', 'ws_server.py'),
}
camera_paused = False
ext_proc = None
current_game_id = ""  # 全局变量，用于存储当前指令

tracker = CentroidTracker(max_dist=600, max_missed=30)
active_connections = set()
capture_task = None
model = None # Initialize model as None, load it in start_server
camera = None
json_path = "game_play.json"
rects = load_rects_from_json(json_path)
drawing_new = False
temp_start = (-1, -1)
temp_end = (-1, -1)
conf_number = 0.15

# 修正：初始化全局变量
last_confirm_time = 0.0
pose_history = {}
kpt_filters_bank = {}
prev_ts = None # 修正：初始化 prev_ts
last_broadcasted_game_id = None # 用于节流广播
hand_confirm_history = {}      # {track_id: [0/1, ...]}
hand_confirm_frame_count = {}  # {track_id: int} 连续举手帧数
HAND_CONFIRM_REQUIRED_FRAMES = 3   # 需要连续满足的帧数（约0.67秒@30fps）
HAND_CONFIRM_COOLDOWN = 1.0         # 触发后冷却时间（秒）
hand_confirm_last_trigger = {}      # {track_id: float} 上次触发时间戳
shutdown_event = None

# ============= 窗口显示控制（首页） =============
show_server_window = if_show_server
toggle_server_window_flag = False   # 由 handler 协程置 True，capture_and_infer 读取后处理

# ================= 广播数据给所有客户端 =================
async def broadcast_data(data):
    if not active_connections:
        return
    message = json.dumps(data)
    tasks = []
    for ws in list(active_connections):
        tasks.append(asyncio.create_task(ws.send(message)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
# ================= 鼠标回调 =================
def detect_hand_confirm1(track_id, kpts, game_id_to_confirm):
    details = {
        'left_raised': False,
        'right_raised': False,
        'left_wrist_y': None,
        'right_wrist_y': None,
        'nose_y': None,
        'left_shoulder_y': None,
        'right_shoulder_y': None,
        'left_wrist_conf': None,
        'right_wrist_conf': None,
    }

    nose = kpts[0]
    left_shoulder = kpts[5]
    right_shoulder = kpts[6]
    left_wrist = kpts[9]
    right_wrist = kpts[10]

    # 记录 y 坐标用于调试
    details['nose_y'] = float(nose[1])
    details['left_shoulder_y'] = float(left_shoulder[1])
    details['right_shoulder_y'] = float(right_shoulder[1])

    # 检查左手是否举起
    details['left_wrist_y'] = float(left_wrist[1])
    # 条件：左腕 y < 左肩 y 且 左腕 y < 鼻子 y
    if left_wrist[1] < left_shoulder[1] and left_wrist[1] < nose[1]:
        details['left_raised'] = True

    # 检查右手是否举起
    details['right_wrist_y'] = float(right_wrist[1])
    # 条件：右腕 y < 右肩 y 且 右腕 y < 鼻子 y
    if right_wrist[1] < right_shoulder[1] and right_wrist[1] < nose[1]:
        details['right_raised'] = True
    if details['left_raised']==True or details['right_raised']==True:
        asyncio.create_task(broadcast_data({"cmd": "switch_game", "game_id": game_id_to_confirm}))
        print(f"\n[动作确认] 检测到举手，选择游戏 {game_id_to_confirm}", flush=True)
        return True
    else:
        return False
def detect_hand_confirm2(track_id, kpts, game_id_to_confirm):
    details = {
        'left_raised': False,
        'right_raised': False,
    }

    nose           = kpts[0]
    left_shoulder  = kpts[5]
    right_shoulder = kpts[6]
    left_wrist     = kpts[9]
    right_wrist    = kpts[10]

    nose_y           = float(nose[1])
    left_shoulder_y  = float(left_shoulder[1])
    right_shoulder_y = float(right_shoulder[1])
    left_wrist_y     = float(left_wrist[1])
    right_wrist_y    = float(right_wrist[1])

    # 肩膀间距，用来判断骨骼点是否合理
    shoulder_width = abs(float(left_shoulder[0]) - float(right_shoulder[0]))

    # ✅ 合理性检查：肩膀间距太小说明人太远或骨骼点乱了，跳过
    if shoulder_width < 30:
        return False

    # 检查左手是否举起
    if left_wrist_y < left_shoulder_y and left_wrist_y < nose_y:
        details['left_raised'] = True

    # 检查右手是否举起
    if right_wrist_y < right_shoulder_y and right_wrist_y < nose_y:
        details['right_raised'] = True

    if details['left_raised'] or details['right_raised']:
        asyncio.create_task(broadcast_data({"cmd": "switch_game", "game_id": game_id_to_confirm}))
        print(f"\n[动作确认] 检测到举手，选择游戏 {game_id_to_confirm}", flush=True)
        return True

    return False
def detect_hand_confirm(track_id, kpts, game_id_to_confirm):
    global hand_confirm_frame_count, hand_confirm_last_trigger

    nose           = kpts[0]
    left_shoulder  = kpts[5]
    right_shoulder = kpts[6]
    left_wrist     = kpts[9]
    right_wrist    = kpts[10]

    nose_y           = float(nose[1])
    left_shoulder_y  = float(left_shoulder[1])
    right_shoulder_y = float(right_shoulder[1])
    left_wrist_y     = float(left_wrist[1])
    right_wrist_y    = float(right_wrist[1])

    shoulder_width = abs(float(left_shoulder[0]) - float(right_shoulder[0]))
    if shoulder_width < 30:
        hand_confirm_frame_count[track_id] = 0
        return False

    # 判断是否举手
    left_raised  = left_wrist_y  < left_shoulder_y  and left_wrist_y  < nose_y
    right_raised = right_wrist_y < right_shoulder_y and right_wrist_y < nose_y
    hand_raised  = left_raised or right_raised

    # 连续帧计数：举手则+1，否则清零
    if hand_raised:
        hand_confirm_frame_count[track_id] = hand_confirm_frame_count.get(track_id, 0) + 1
    else:
        hand_confirm_frame_count[track_id] = 0
        return False

    # 未达到所需连续帧数，继续等待
    if hand_confirm_frame_count[track_id] < HAND_CONFIRM_REQUIRED_FRAMES:
        return False

    # 冷却检查：避免连续重复触发
    now = time.time()
    last = hand_confirm_last_trigger.get(track_id, 0.0)
    if now - last < HAND_CONFIRM_COOLDOWN:
        return False

    # 满足条件，触发确认
    hand_confirm_frame_count[track_id] = 0          # 重置计数
    hand_confirm_last_trigger[track_id] = now        # 记录触发时间
    asyncio.create_task(broadcast_data({"cmd": "switch_game", "game_id": game_id_to_confirm}))
    print(f"\n[动作确认] 连续举手 {HAND_CONFIRM_REQUIRED_FRAMES} 帧，选择游戏 {game_id_to_confirm}", flush=True)
    return True
def mouse_callback(event, x, y, flags, param):
    global rects, drawing_new, temp_start, temp_end
    # 锁定的矩形跳过操作
    for rect in rects:
        if rect.locked:
            rect.selected_point = -1
            rect.selected_whole = False
    # 1. 左键按下：选顶点/选整体/新建矩形
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = False
        # 优先选未锁定矩形的顶点
        for rect in rects:
            if rect.locked:
                continue
            for i, (px, py) in enumerate(rect.points):
                if distance((x, y), (px, py)) < 10:
                    # 取消其他矩形选中状态
                    for r in rects:
                        if r != rect:
                            r.selected_point = -1
                            r.selected_whole = False
                    rect.selected_point = i
                    clicked = True
                    break
            if clicked:
                break
        if not clicked:
            # 选未锁定矩形的内部（整体拖动）
            for rect in rects:
                if rect.locked:
                    continue
                if cv2.pointPolygonTest(np.array(rect.points), (x, y), False) >= 0:
                    # 取消其他矩形选中状态
                    for r in rects:
                        if r != rect:
                            r.selected_point = -1
                            r.selected_whole = False
                    rect.selected_whole = True
                    center = get_rect_center(rect.points)
                    rect.drag_offset = (center[0] - x, center[1] - y)
                    clicked = True
                    break
        # 新建矩形
        if not clicked:
            drawing_new = True
            temp_start = (x, y)
            temp_end = (x, y)
    # 2. 鼠标移动：拖顶点/拖整体/更新临时矩形
    elif event == cv2.EVENT_MOUSEMOVE:
        # 拖顶点
        for rect in rects:
            if rect.locked or rect.selected_point == -1:
                continue
            rect.points[rect.selected_point] = (x, y)
        # 拖整体
        for rect in rects:
            if rect.locked or not rect.selected_whole:
                continue
            new_center = (x + rect.drag_offset[0], y + rect.drag_offset[1])
            old_center = get_rect_center(rect.points)
            dx = new_center[0] - old_center[0]
            dy = new_center[1] - old_center[1]
            for i in range(4):
                rect.points[i] = (rect.points[i][0] + dx, rect.points[i][1] + dy)
        # 更新临时矩形
        if drawing_new:
            temp_end = (x, y)
    # 3. 左键释放：重置选中状态/完成新建
    elif event == cv2.EVENT_LBUTTONUP:
        for rect in rects:
            rect.selected_point = -1
            rect.selected_whole = False
        # 完成新建
        if drawing_new:
            drawing_new = False
            new_rect_points = [
                temp_start, (temp_end[0], temp_start[1]), temp_end, (temp_start[0], temp_end[1])
            ]
            rects.append(Rect(new_rect_points, locked=False))
    # 4. 右键单击：撤回最后一个未锁定的矩形
    elif event == cv2.EVENT_RBUTTONDOWN:
        if rects:
            # 优先撤回最后一个未锁定的矩形
            for i in reversed(range(len(rects))):
                if not rects[i].locked:
                    del rects[i]
                    # print("已撤回最后一个未锁定矩形")
                    break
def are_both_feet_in_rect(kpts, rect, conf_thresh=0.3):
    """
    判断左右脚(15,16)是否【同时】在 rect 内

    kpts: (17,2) or (17,3)  像素坐标
    rect: Rect，rect.points = [(x,y), ...]
    conf_thresh: 关键点置信度阈值

    return:
        True  -> 左右脚都在 rect 内
        False -> 任意一只脚不在 / 不可靠
    """
    if kpts is None or len(kpts) < 17:
        return False

    pts_rect = np.array(rect.points, np.int32)

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    for idx in (LEFT_ANKLE, RIGHT_ANKLE):
        kp = kpts[idx]

        # 支持 (x,y) / (x,y,conf)
        if len(kp) == 3:
            x, y, conf = kp
            if conf < conf_thresh:
                return False
        else:
            x, y = kp

        inside = cv2.pointPolygonTest(
            pts_rect, (float(x), float(y)), False
        )

        if inside < 0:
            return False  # 任意一只脚不在，直接 False

    return True
def smooth_kpts_one_euro(person, dt, key_name="track_id", min_cutoff=1.2, beta=0.006, d_cutoff=1.0, conf_thresh=0.3,max_jump=60):
    global pose_history, kpt_filters_bank
    if key_name not in person:
        return person["kpts"]
    pid = person[key_name]
    kpts = np.array(person["kpts"], dtype=np.float32)
    scores = np.array(person["scores"], dtype=np.float32)
    N = kpts.shape[0]
    if pid not in pose_history:
        pose_history[pid] = kpts.copy()
    if pid not in kpt_filters_bank or len(kpt_filters_bank[pid]) != N:
        kpt_filters_bank[pid] = [OneEuro2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(N)]
    prev = pose_history[pid]
    smooth = prev.copy()
    for i in range(N):
        x_new, y_new = kpts[i]
        x_old, y_old = prev[i]
        conf = scores[i]
        if conf < conf_thresh:
            continue
        dist = np.hypot(x_new - x_old, y_new - y_old)
        if dist > max_jump:
            scale = max_jump / (dist + 1e-6)
            x_new = x_old + (x_new - x_old) * scale
            y_new = y_old + (y_new - y_old) * scale
        x_s, y_s = kpt_filters_bank[pid][i].filter((x_new, y_new), dt)
        smooth[i, 0] = x_s
        smooth[i, 1] = y_s
    pose_history[pid] = smooth
    return smooth.tolist()
def compute_hist(frame, bbox):
    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W - 1, int(x2))
    y2 = min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32).tolist()
# ================= 摄像头推理任务 =================
async def capture_and_infer():
    global camera_paused, camera, rects, prev_ts, current_game_id, last_broadcasted_game_id
    global show_server_window, toggle_server_window_flag
    print("摄像头推理任务已启动")

    try:
        while True:
            # 检查是否需要切换窗口
            if toggle_server_window_flag:
                toggle_server_window_flag = False
                show_server_window = not show_server_window
                if show_server_window:
                    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Pose Estimation", 1920, 1200)
                    if camera is not None:
                        frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cv2.setMouseCallback("Pose Estimation", mouse_callback, param=(frame_h, frame_w))
                    print("[ws_server] 显示首页窗口")
                else:
                    cv2.destroyWindow("Pose Estimation")
                    cv2.waitKey(1)
                    print("[ws_server] 隐藏首页窗口")

            if camera_paused:
                if camera is not None:
                    camera.release()
                    cv2.destroyAllWindows()
                    camera = None
                    print(">>> 摄像头已关闭，外部脚本正在运行...")
                await asyncio.sleep(0.2)
                continue

            if camera is None:
                print(">>> 正在重新打开摄像头...")
                try:
                    camera = cv2.VideoCapture(video_path) # Use default camera index 0
                    if not camera.isOpened():
                        print("错误: 无法打开摄像头. 请检查摄像头是否连接或被占用.")
                        camera = None # Ensure camera is None if opening fails
                        await asyncio.sleep(2)
                        continue
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
                    time.sleep(0.5)
                    # 1. 获取视频的原始FPS
                    fps = camera.get(cv2.CAP_PROP_FPS)
                    if fps == 0:
                        print("警告：无法获取视频的FPS，将使用默认值 30。")
                        fps = 30
                    # 强制限制 fps 用于逻辑计算的最大值，避免因相机fps过高导致hold帧数过大
                    if fps > 30:
                        print(f"检测到 FPS={fps} 较高，强制限制逻辑 FPS 为 30 以适配处理速度")
                        fps = 30
                    if show_server_window:
                        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Pose Estimation", 1920, 1200)
                        frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cv2.setMouseCallback("Pose Estimation", mouse_callback, param=(frame_h, frame_w))
                    prev_ts = time.time() # 修正：在摄像头成功打开后初始化 prev_ts
                except Exception as e:
                    print(f"打开摄像头时发生错误: {e}")
                    camera = None
                    await asyncio.sleep(2)
                    continue

            fps = camera.get(cv2.CAP_PROP_FPS)
            if fps <= 0: # 如果获取到的 FPS 无效（例如0或负数），则使用默认值
                # print(f"警告: 获取到无效的FPS ({fps})，使用默认值 30.")
                fps = 30

            ret, frame = camera.read()
            if not ret:
                print("警告: 无法读取摄像头帧. 可能是摄像头断开或出现故障.")
                await asyncio.sleep(0.01)
                continue

            ts = time.time()
            dt = ts - prev_ts if prev_ts is not None else 1.0 / fps
            prev_ts = ts

            # 推理
            if model is None:
                print("错误: 模型未加载. 跳过推理.")
                await asyncio.sleep(0.1)
                continue

            inference_result = model(frame)
            # 用一个列表存每个人的数据：bbox + kpts + scores
            persons = []
            for p in inference_result.results:
                score = p.get("score")
                label = p.get("label")
                if label == "person":
                    bbox = p.get("bbox")
                    l = []
                    s = []
                    for lm in p.get("landmarks"):
                        l.append(lm.get("landmark"))
                        s.append(lm.get("score"))
                    if score > conf_number:
                        persons.append({
                            "bbox": bbox,
                            "kpts": l,
                            "scores": s,
                            "score": score,
                            "low_conf": False
                        })
            # 按 bbox 左上角 x1 从小到大排序
            persons.sort(key=lambda person: person["bbox"][0])
            # ==========================================================

            in_persons = []  # 在大框里的所有人
            out_persons = []  # 不在大框里的（你可以忽略）
            for person in persons:
                kpts = person["kpts"]
                for rect in rects:
                    rect.in_if = False
                for rid, rect in enumerate(rects, 1):
                    if are_both_feet_in_rect(kpts, rect):
                        rect.in_if = True
                        person["rect_id"] = rid
                        in_persons.append(person)
                        break
                    else:
                        out_persons.append(person)
            for p in in_persons:
                h = compute_hist(frame, p["bbox"])
                if h is not None:
                    p["hist"] = h
            in_persons = tracker.update(in_persons)
            all_in_box = len(in_persons)
            if all_in_box == 0:
                cv2.putText(frame, f"Please stand in the box", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)
            elif all_in_box > 1:
                cv2.putText(frame, f"Only ONE person allowed! ({all_in_box})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 0, 255), 2)
            else:
                # 画框 + 写名字 + 画骨架 + 检测运动状态
                for idx, person in enumerate(in_persons, 1):
                    bbox = person["bbox"]
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    smoothed_kpts = smooth_kpts_one_euro(
                        person,
                        dt,
                        key_name="track_id",
                        min_cutoff=0.2,
                        beta=0.05,
                        d_cutoff=1.0,
                        conf_thresh=0.6,  # 提高骨骼点置信度阈值到 0.6
                        max_jump=100
                    )

                    # 修正：确保 current_game_id 是字符串，与 EXT_SCRIPT 的键匹配
                    current_game_id = str(person.get("rect_id", 1))
                    asyncio.create_task(broadcast_data({"cmd": "num","value": current_game_id}))

                    # ====== 举手确认逻辑集成 ======
                    if detect_hand_confirm(person["track_id"], smoothed_kpts, current_game_id):
                        await start_ext_script(current_game_id)
                        print(f"Confirm Detected! Starting game {current_game_id}", flush=True)

                    skeleton_color = (0, 255, 0)  # Green
                    for j in skeleton_map:
                        s_id = j.get("srt_kpt_id")
                        d_id = j.get("dst_kpt_id")
                        # 检查索引是否在 smoothed_kpts 范围内
                        if s_id < len(smoothed_kpts) and d_id < len(smoothed_kpts):
                            sx = int(smoothed_kpts[s_id][0])
                            sy = int(smoothed_kpts[s_id][1])
                            dx = int(smoothed_kpts[d_id][0])
                            dy = int(smoothed_kpts[d_id][1])
                            if sx == 0 or sy == 0 or dx == 0 or dy == 0:
                                continue
                            cv2.line(frame, (sx, sy), (dx, dy), skeleton_color, 2)
                        else:
                            print(f"警告: 骨骼点索引超出范围. s_id={s_id}, d_id={d_id}, len(smoothed_kpts)={len(smoothed_kpts)}")

            # 显示当前选中的游戏 ID
            cv2.putText(frame, f"Selected Game: {current_game_id}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(255, 255, 0), 2)

            # 绘制所有矩形（确保在串流之前绘制，以便网页端可见）
            for idx, rect in enumerate(rects):
                # 边框颜色：锁定(灰色)、未锁定(绿色)、选中(蓝色)
                if rect.locked:
                    border_color = (128, 128, 128)
                elif rect.selected_point != -1 or rect.selected_whole:
                    border_color = (255, 0, 0)  # 选中的矩形边框变蓝
                else:
                    border_color = (0, 255, 0)

                # 绘制边框
                cv2.polylines(frame, [np.array(rect.points, np.int32)], isClosed=True, color=border_color,
                              thickness=2)

                # 绘制顶点：锁定(灰色)、选中(黄色)、未选中(红色)
                for i, (px, py) in enumerate(rect.points):
                    if rect.locked:
                        pt_color = (128, 128, 128)
                    elif rect.selected_point == i:
                        pt_color = (0, 255, 255)
                    else:
                        pt_color = (0, 0, 255)
                    cv2.circle(frame, (px, py), 5, pt_color, -1)

                # 绘制状态标签
                px, py = rect.points[0]  # 左上角
                label = rect.name if rect.name else f"box{idx + 1}"
                cv2.putText(frame, label, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 绘制临时矩形(蓝色虚线)
            if drawing_new and temp_start != (-1, -1):
                temp_points = [temp_start, (temp_end[0], temp_start[1]), temp_end, (temp_start[0], temp_end[1])]
                cv2.polylines(frame, [np.array(temp_points, np.int32)], isClosed=True, color=(255, 0, 0),
                              thickness=1,
                              lineType=cv2.LINE_AA)
            if show_server_window:
                cv2.imshow("Pose Estimation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_rects_to_json(rects, json_path)

            await asyncio.sleep(0.001) # 稍微减少 CPU 占用

    except asyncio.CancelledError:
        print("摄像头任务收到 cancel")

    except Exception as e:
        print(f"摄像头推理任务发生未预期错误: {e}")

    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        print("摄像头推理任务已结束")
# ================= 外部脚本管理 =================
async def start_ext_script(cmd):
    global camera_paused, ext_proc, current_game_id
    cmd_path = EXT_SCRIPT.get(cmd)
    if not cmd_path:
        return f"错误: 未知的游戏ID {cmd}"

    if ext_proc and ext_proc.poll() is None:
        return "外部脚本已在运行"

    camera_paused = True
    # 等待摄像头释放
    await asyncio.sleep(0.5)

    try:
        # 启动外部脚本进程
        ext_proc = subprocess.Popen(
            [sys.executable, cmd_path],
            cwd=str(Path(cmd_path).parent)
        )
        current_game_id = cmd # 修正：确保 current_game_id 更新为当前启动的游戏ID
        return f"已启动外部脚本 (PID: {ext_proc.pid})，摄像头已关闭"
    except Exception as e:
        camera_paused = False
        return f"启动失败: {str(e)}"
async def stop_ext_script():
    global camera_paused, ext_proc, current_game_id
    if not ext_proc or ext_proc.poll() is not None:
        camera_paused = False
        return "外部脚本未在运行或已结束"

    pid = ext_proc.pid
    print(f"开始强制关闭外部脚本及其子进程 (主进程 PID: {pid})...")

    # --- 策略一：使用 psutil 递归杀死进程树 ---
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        print(f"找到 {len(children)} 个子进程。")

        # 先杀子进程
        for child in children:
            try:
                print(f"正在终止子进程 (PID: {child.pid})...")
                child.kill()
            except psutil.NoSuchProcess:
                print(f"子进程 {child.pid} 已不存在。")

        # 等待子进程终止
        gone, alive = psutil.wait_procs(children, timeout=3)
        if alive:
            for p in alive:
                print(f"警告：子进程 {p.pid} 未能终止，再次尝试强制杀死。")
                p.kill()

        # 最后杀父进程
        try:
            print(f"正在终止主进程 (PID: {parent.pid})...")
            parent.kill()
            parent.wait(timeout=3)
        except psutil.NoSuchProcess:
            print(f"主进程 {parent.pid} 已不存在。")

    except psutil.NoSuchProcess:
        print(f"进程 (PID: {pid}) 在 psutil 处理前就已消失。")
    except Exception as e:
        print(f"使用 psutil 关闭进程时出错: {e}")

    # --- 策略二：使用系统命令作为后备（双重保险）---
    # 即使 psutil 认为完成了，也再次执行以防万一
    if os.name == 'nt':
        # Windows: /T 会杀死进程树
        print(f"后备策略 (Windows): 使用 taskkill /F /T /PID {pid}")
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        # Linux/macOS: 使用 pkill 杀死整个进程组
        # 注意：这需要 start_ext_script 中使用了 preexec_fn=os.setsid
        print(f"后备策略 (Linux/macOS): 使用 pkill -9 -P {pid}")
        subprocess.run(f"pkill -9 -P {pid}", shell=True, capture_output=True)
        # 再次尝试杀死主进程
        try:
            ext_proc.kill()
        except:
            pass

    ext_proc = None
    current_game_id = ""

    # --- 策略三：关键延时 ---
    # 给操作系统足够的时间来完成硬件驱动层面的资源释放
    print("进程已清理。等待 1.5 秒以便系统释放摄像头资源...")
    await asyncio.sleep(1.5)

    camera_paused = False  # 现在才设置标志，触发摄像头恢复
    print("标志已设置，主循环将尝试重新打开摄像头。")

    return "外部脚本已强制关闭，摄像头正在恢复..."
# ================= WebSocket 处理 =================
async def handle_client(websocket):
    global capture_task, server, shutdown_event

    active_connections.add(websocket)
    print(f"客户端连接: {websocket.remote_address}")

    # 确保摄像头任务正在运行
    if capture_task is None or capture_task.done():
        print("启动或重启摄像头任务...")
        capture_task = asyncio.create_task(capture_and_infer())

    try:
        async for message in websocket:
            # 如果服务器正在关闭，则不再处理新消息
            if shutdown_event and shutdown_event.is_set():
                break

            cmd = message.strip().lower()
            print(f"收到指令: '{cmd}'")

            # 尝试解析 JSON 指令
            try:
                json_data = json.loads(message)
                json_cmd = json_data.get("cmd", "")
            except Exception:
                json_data = None
                json_cmd = ""

            if json_cmd == "sy_window":
                # 切换首页（ws_server.py 自身）的 OpenCV 窗口
                global toggle_server_window_flag
                toggle_server_window_flag = True
                print("[SERVER] 收到 sy_window，将切换首页窗口")
                continue
            elif json_cmd == "dss_window":
                # 广播给 dg_dds.py（深蹲游戏）切换窗口
                await broadcast_data({"cmd": "dss_window"})
                print("[SERVER] 已广播 dss_window")
                continue
            elif json_cmd == "123_window":
                # 广播给 dg_123.py（123木头人游戏）切换窗口
                await broadcast_data({"cmd": "123_window"})
                print("[SERVER] 已广播 123_window")
                continue
            elif json_cmd == "kht_window":
                # 广播给 dg_kht.py（开合跳游戏）切换窗口
                await broadcast_data({"cmd": "kht_window"})
                print("[SERVER] 已广播 kht_window")
                continue

            if cmd in ("1", "2", "3"):
                response = await start_ext_script(cmd)
                await websocket.send(response)
            elif cmd == "q":
                response = await stop_ext_script()
                await websocket.send(response)
            elif cmd in ("exit", "esc"):
                print("收到退出指令，开始全局关闭流程...")
                # 尝试通知客户端，忽略可能发生的连接关闭错误
                try:
                    await websocket.send("服务端即将关闭，断开连接...")
                except websockets.exceptions.ConnectionClosed:
                    pass  # 如果连接已经关了，就不用管了

                if shutdown_event:
                    shutdown_event.set()  # 触发全局关闭事件
                break  # 跳出循环，此客户端处理结束

    except websockets.exceptions.ConnectionClosedOK:
        print(f"客户端 {websocket.remote_address} 已正常断开。")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"客户端 {websocket.remote_address} 异常断开: {e}")
    finally:
        active_connections.discard(websocket)
        print(f"客户端 {websocket.remote_address} 连接处理结束。")
# ================= 主程序 =================
async def start_server():
    """
    初始化并启动 WebSocket 服务器，并管理整个应用的生命周期。
    """
    global server, model, shutdown_event, capture_task, ext_proc

    # 1. 在事件循环内部创建 Event 对象，解决 "attached to a different loop" 问题
    shutdown_event = asyncio.Event()

    # 2. 加载模型
    try:
        model = dg.load_model(**aaa)
        print("DeGirum模型加载成功。")
    except Exception as e:
        print(f"致命错误: 无法加载DeGirum模型: {e}")
        print("请确保DeGirum NPU已连接且模型配置正确。程序即将退出。")
        return

    # 3. 启动 WebSocket 服务器
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8765
    )
    print("WebSocket 服务端已启动于 ws://0.0.0.0:8765")
    print("等待客户端连接或关闭信号...")

    # 4. 等待关闭信号
    #   - shutdown_event.wait() 会一直阻塞，直到有地方调用 shutdown_event.set()
    #   - 或者，如果主程序被 Ctrl+C 中断，asyncio.run 会引发 CancelledError，也会让程序继续往下走
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        print("\n检测到程序中断 (例如 Ctrl+C)，开始清理...")

    # 5. --- 开始有序的清理流程 ---
    print("\n检测到关闭事件，开始执行有序清理...")

    # 5.1. 停止接受新连接，并等待现有连接的关闭握手完成
    if server:
        server.close()
        await server.wait_closed()
        print("WebSocket 服务器已关闭，不再接受新连接。")

    # 5.2. 强制关闭所有仍然活跃的客户端连接
    if active_connections:
        print(f"正在强制关闭 {len(active_connections)} 个剩余的客户端连接...")
        await asyncio.gather(
            *(ws.close(code=1001, reason='Server is shutting down') for ws in list(active_connections)),
            return_exceptions=True
        )

    # 5.3. 关闭仍在运行的外部脚本
    if ext_proc and ext_proc.poll() is None:
        print("正在关闭仍在运行的外部脚本...")
        await stop_ext_script()

    # 5.4. 停止摄像头推理任务
    if capture_task and not capture_task.done():
        print("正在取消摄像头推理任务...")
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            print("摄像头推理任务已成功取消。")

    print("\n服务端所有资源已安全释放，程序退出。")
if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("检测到 Ctrl+C，正在关闭服务端...")
    except Exception as e:
        print(f"主程序发生未预期错误: {e}")

