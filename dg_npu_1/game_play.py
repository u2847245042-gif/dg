# -*- coding: utf-8 -*-
from __future__ import annotations
import subprocess
import os
import json
import threading
import queue
import time
import sys
import cv2
import numpy as np
import degirum as dg

from utils import Rect, distance, get_rect_center, save_rects_to_json, load_rects_from_json, _init_logger, skeleton_map, \
    aaa, OneEuroFilter, OneEuro2D, CentroidTracker

from websockets.sync.client import connect

STOP_FLAG = threading.Event()
ws_send_queue = queue.Queue()
ws_receiver_queue = queue.Queue()
ip_id = 8765

def ws_sender_thread():
    uri = f"ws://127.0.0.1:{ip_id}"
    while not STOP_FLAG.is_set():
        try:
            print("尝试连接 WebSocket 服务器:", uri)
            with connect(uri) as websocket:
                print("dg_dds 已连接 WebSocket 服务器")

                while not STOP_FLAG.is_set():
                    msg = ws_send_queue.get()
                    if msg is None:
                        break
                    websocket.send(msg)
        except Exception as e:
            if STOP_FLAG.is_set():
                break
            print("连接 WebSocket 失败，将在 2 秒后重试:", e)
            time.sleep(2)
def ws_receiver_thread():
    global rects, persons, marked_ids, INPUT_Q
    while not STOP_FLAG.is_set():
        try:
            uri = f"ws://127.0.0.1:{ip_id}"
            with connect(uri) as ws:
                print("[WS Receiver] 接收通道已连接")
                while not STOP_FLAG.is_set():
                    try:
                        msg = ws.recv(timeout=0.2)  # 小超时，方便及时退出
                        if not msg:
                            continue

                        data = json.loads(msg)
                        print(000000, msg)
                        print(f"[WS Receiver] 收到: {data}")

                        # 处理 clear_marked 命令
                        if data.get("cmd") == "clear_marked":
                            print("[WS Receiver] 已清零 rects.count & persons.is_marked & marked_ids")
                            # 这里可以添加具体的清零逻辑

                        # 处理输入命令 (1,2,3,q,esc)
                        elif data.get("cmd") == "input":
                            value = data.get("value")
                            if value in ["1", "2", "3", "q", "esc"]:
                                print(f"[WS Receiver] 收到输入命令: {value}，放入 INPUT_Q")

                                # 如果是 esc 命令，设置 STOP_FLAG
                                if value == "esc":
                                    STOP_FLAG.set()
                                    print("[WS Receiver] 收到 esc 命令，设置 STOP_FLAG")
                                else:
                                    # 将命令放入 INPUT_Q
                                    try:
                                        INPUT_Q.put_nowait(value)
                                        print(f"[WS Receiver] 已放入 INPUT_Q: {value}")
                                    except Exception as e:
                                        print(f"[WS Receiver] 放入 INPUT_Q 失败: {e}")

                    except json.JSONDecodeError:
                        print(f"[WS Receiver] 非JSON消息: {msg}")
                    except TimeoutError:
                        continue  # 超时正常，继续循环
                    except Exception as e:
                        print(f"[WS Receiver] 接收消息出错: {e}")
                        break

        except Exception as e:
            if not STOP_FLAG.is_set():
                print("[WS Receiver] 接收通道断开，2 秒后重连：", e)
                time.sleep(2)

    print("[WS Receiver] 接收线程已退出")
threading.Thread(target=ws_sender_thread, daemon=True).start()
threading.Thread(target=ws_receiver_thread, daemon=True).start()

# ====== 挥手识别相关全局变量 ======
wave_history = {}  # 存储每个 track_id 的手部位置历史 {track_id: {'left': [], 'right': []}}
WAVE_BUFFER_SIZE = 15  # 15帧窗口
WAVE_THRESHOLD = 100  # 降低阈值，从120降到100
last_wave_time = 0  # 防止短时间内重复触发
last_confirm_time = 0  # 举手确认冷却
current_game_id = 1  # 当前选中的游戏 ID
hand_confirm_history = {}  # {track_id: [0/1, ...]}


def is_hand_raise_pose(kpts, margin_scale=0.4, margin_min=40):
    if kpts is None or len(kpts) < 11:
        return False
    head_ids = (0, 1, 2, 3, 4)
    shoulder_ids = (5, 6)
    wrist_ids = (9, 10)

    head_ys = []
    for hid in head_ids:
        x, y = kpts[hid]
        if x > 0 and y > 0:
            head_ys.append(y)
    if not head_ys:
        return False
    head_y = min(head_ys)

    shoulder_ys = []
    for sid in shoulder_ids:
        x, y = kpts[sid]
        if x > 0 and y > 0:
            shoulder_ys.append(y)
    shoulder_y = sum(shoulder_ys) / len(shoulder_ys) if shoulder_ys else (head_y + 100.0)

    margin = max(margin_min, int(abs(shoulder_y - head_y) * margin_scale))
    for wid in wrist_ids:
        x, y = kpts[wid]
        if x > 0 and y > 0 and y < head_y - margin:
            return True
    return False
def detect_wave(track_id, kpts):
    """
    重构后的挥手检测逻辑：
    使用“主导方向”投票机制，确保左右识别绝对准确。
    """
    global wave_history, last_wave_time, current_game_id

    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    if is_hand_raise_pose(kpts):
        return None

    if track_id not in wave_history:
        wave_history[track_id] = {'left': [], 'right': []}

    now = time.time()
    triggered = False
    direction = 0

    for side, idx in [('left', LEFT_WRIST), ('right', RIGHT_WRIST)]:
        wrist_pos = kpts[idx]
        if wrist_pos[0] == 0 and wrist_pos[1] == 0:
            continue

        history = wave_history[track_id][side]
        history.append((wrist_pos[0], wrist_pos[1]))
        if len(history) > WAVE_BUFFER_SIZE:
            history.pop(0)

        if len(history) < WAVE_BUFFER_SIZE:
            continue

        # 1. 计算每一帧之间的位移 deltas
        xs = [p[0] for p in history]
        ys = [p[1] for p in history]
        dx_deltas = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        dy_deltas = [ys[i] - ys[i - 1] for i in range(1, len(ys))]

        # 2. 统计主导方向（投票）
        pos_votes = sum(1 for d in dx_deltas if d > 1)  # 降低判定门槛，从2降到1
        neg_votes = sum(1 for d in dx_deltas if d < -1)  # 降低判定门槛，从-2降到-1

        # 3. 计算总位移
        dx_total = xs[-1] - xs[0]
        dy_total = ys[-1] - ys[0]
        total_range_x = max(xs) - min(xs)

        # 4. 判定逻辑
        if now - last_wave_time > 1.8:  # 冷却时间
            if total_range_x > WAVE_THRESHOLD:
                # 降低投票比例门槛，从60%降到50%
                vote_threshold = len(dx_deltas) * 0.5

                # 判定方向
                current_dir = 0
                if pos_votes > vote_threshold and dx_total > WAVE_THRESHOLD * 0.4:
                    current_dir = 1
                elif neg_votes > vote_threshold and dx_total < -WAVE_THRESHOLD * 0.4:
                    current_dir = -1

                if current_dir != 0:
                    # 5. 垂直位移抑制：垂直总位移不能超过水平总位移的一半
                    if abs(dy_total) < abs(dx_total) * 0.6:
                        direction = current_dir
                        triggered = True
                        wave_history[track_id][side] = []  # 清空该手缓冲区
                        break

    if triggered:
        last_wave_time = now
        old_id = current_game_id
        if direction == 1:
            current_game_id = min(3, current_game_id + 1)
            msg = f"Wave Right ({side}): {old_id} -> {current_game_id}"
        else:
            current_game_id = max(1, current_game_id - 1)
            msg = f"Wave Left ({side}): {old_id} -> {current_game_id}"

        # 同步给 WebSocket
        try:
            ws_send_queue.put_nowait(json.dumps({
                "cmd": "game_selection",
                "value": current_game_id,
                "msg": msg
            }))
        except Exception:
            pass
        return msg

    return None
def detect_hand_confirm(track_id, kpts):
    """
    优化后的举手确认动作：
    手腕高于头顶（Y坐标更小）
    """
    global last_confirm_time, INPUT_Q

    now = time.time()
    if now - last_confirm_time < 2.0:
        return False

    raised = is_hand_raise_pose(kpts, margin_scale=0.4, margin_min=40)

    history = hand_confirm_history.setdefault(track_id, [])
    history.append(1 if raised else 0)
    if len(history) > 8:
        del history[:-8]

    if sum(history) >= 4 and history[-1] == 1:
        hand_confirm_history[track_id] = []
        last_confirm_time = now
        try:
            ws_send_queue.put_nowait(json.dumps({
                "cmd": "confirm_selection",
                "value": current_game_id
            }))
            INPUT_Q.put(str(current_game_id))
            print(f"\n[动作确认] 检测到举手，选择游戏 {current_game_id}", flush=True)
        except Exception:
            pass
        return True

    return False
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
def smooth_kpts_one_euro_display(person, dt, key_name="track_id", min_cutoff=0.05, beta=0.005, d_cutoff=1.0,conf_thresh=0.3, max_jump=60):
    global pose_history_display, kpt_filters_bank_display
    if key_name not in person:
        return person["kpts"]
    pid = person[key_name]
    kpts = np.array(person["kpts"], dtype=np.float32)
    scores = np.array(person["scores"], dtype=np.float32)
    N = kpts.shape[0]
    if pid not in pose_history_display:
        pose_history_display[pid] = kpts.copy()
    if pid not in kpt_filters_bank_display or len(kpt_filters_bank_display[pid]) != N:
        kpt_filters_bank_display[pid] = [OneEuro2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in
                                         range(N)]
    prev = pose_history_display[pid]
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
        x_s, y_s = kpt_filters_bank_display[pid][i].filter((x_new, y_new), dt)
        smooth[i, 0] = x_s
        smooth[i, 1] = y_s
    pose_history_display[pid] = smooth
    return smooth.tolist()
def get_body_center(kpts):
    major_ids = [5, 6, 11, 12]  # 左肩 右肩 左胯 右胯
    xs = []
    ys = []
    for i in major_ids:
        x, y = kpts[i]
        if x > 0:
            xs.append(x)
        if y > 0:
            ys.append(y)
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    return cx, cy

# ==========================================================
log_dir = f"game_play"
log = _init_logger(log_dir)
json_path = "game_play.json"
rects = load_rects_from_json(json_path)
drawing_new = False
temp_start = (-1, -1)
temp_end = (-1, -1)
video_path = 0
conf_number = 0.5
target_file = "yolov8n.pt"

# 使用 __file__ 获取脚本所在的绝对路径，确保在任何地方启动都能找到模型
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 yolov8n.pt 在脚本同级目录或者上一级
if os.path.exists(os.path.join(current_script_dir, target_file)):
    MODEL_PATH = os.path.join(current_script_dir, target_file)
else:
    # 兼容之前可能在上一级目录的情况
    MODEL_PATH = os.path.join(os.path.dirname(current_script_dir), target_file)

# ==========================================================
PREVIEW_WIN = "YOLO Preview (Menu Only)"
PREVIEW_FPS_LIMIT = 30
preview_stop_evt = threading.Event()
preview_pause_evt = threading.Event()
preview_thread = None
win_inited = False
cap_inited = False
pose_history = {}
kpt_filters_bank = {}
pose_history_display = {}
kpt_filters_bank_display = {}
# ====== 子进程管理 ======
GAME_ROOT = current_script_dir
SCRIPTS = {
    1: os.path.join(GAME_ROOT, 'game1', 'ws_server.py'),
    2: os.path.join(GAME_ROOT, 'game2', 'ws_server.py'),
    3: os.path.join(GAME_ROOT, 'game3', 'ws_server.py'),
}
PROCS: dict[int, subprocess.Popen] = {}
EVENT_Q: "queue.Queue[tuple[str,int,object]]" = queue.Queue()
# ====== 统一输入（避免 input() 阻塞）======
INPUT_Q: "queue.Queue[str]" = queue.Queue()


def stdin_reader():
    """
    后台线程：阻塞读 stdin，把整行输入放入队列。
    主线程用 INPUT_Q.get(timeout=...) 轮询，就能响应 ESC。
    """
    while not STOP_FLAG.is_set():
        try:
            line = sys.stdin.readline()
            if not line:
                break
            INPUT_Q.put(line.rstrip("\n"))
        except Exception:
            break
def prompt_wait_line(prompt: str, timeout=0.1) -> str | None:
    """
    打印提示后等待输入（不阻塞主线程）：
    - 返回一行输入（strip 后的小写由调用方决定）
    - 如果 STOP_FLAG 置位，返回 None
    """
    print(prompt, end="", flush=True)
    while not STOP_FLAG.is_set():
        try:
            line = INPUT_Q.get(timeout=timeout)
            return line
        except queue.Empty:
            continue
    return None
def preview_loop():
    global rects, pose_history, kpt_filters_bank, cap_inited, win_inited
    prev_ts = None
    tracker = CentroidTracker(max_dist=600, max_missed=30)
    cap = None
    last_t = 0.0
    try:
        while not preview_stop_evt.is_set() and not STOP_FLAG.is_set():
            if preview_pause_evt.is_set():
                if cap is not None:
                    cap.release()
                    cap = None

                try:
                    if cv2.getWindowProperty(PREVIEW_WIN, cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow(PREVIEW_WIN)
                except cv2.error:
                    pass

                time.sleep(0.05)
                continue
            if cap is None:
                cap = cv2.VideoCapture(video_path, cv2.CAP_V4L2)
                if not cap.isOpened():
                    time.sleep(0.5)
                    continue
                cap_inited = False
                win_inited = False

            # ✅ 只做一次 cap.set + sleep
            if not cap_inited:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
                cap.set(cv2.CAP_PROP_FPS, 30)  # 90 很多摄像头根本达不到，反而会抖
                time.sleep(0.2)  # 只睡一次
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_inited = True

            # ✅ 只做一次窗口初始化
            if not win_inited:
                try:
                    cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(PREVIEW_WIN, 1920, 1200)
                    cv2.setMouseCallback(PREVIEW_WIN, mouse_callback, param=(frame_h, frame_w))
                except cv2.error:
                    print("Warning: Could not create preview window (headless environment?)", flush=True)
                win_inited = True

            now = time.time()
            if now - last_t < 1.0 / max(1, PREVIEW_FPS_LIMIT):
                time.sleep(0.001)
                continue
            last_t = now

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            try:
                ts = time.time()
                dt = ts - prev_ts if prev_ts is not None else 1.0 / fps
                prev_ts = ts
                # 推理
                inference_result = yolo_model(frame)
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
                # ========= 用大框筛选人 =========
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
                    cv2.putText(frame, f"Please stand in the box", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                                2)
                elif all_in_box > 1:
                    cv2.putText(frame, f"Only ONE person allowed! ({all_in_box})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                (0, 0, 255), 2)
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
                        draw_kpts = smooth_kpts_one_euro_display(
                            person,
                            dt,
                            key_name="track_id",
                            min_cutoff=0.05,
                            beta=0.005,
                            d_cutoff=1.0,
                            conf_thresh=0.3,
                            max_jump=60
                        )
                        # ====== 挥手检测逻辑集成 ======
                        wave_msg = detect_wave(person["track_id"], smoothed_kpts)
                        if wave_msg:
                            print(f"Gesture Detected: {wave_msg}", flush=True)

                        # current_game_id = int(person.get("rect_id", 1))
                        # ws_send_queue.put_nowait(json.dumps({"cmd": "num", "value": current_game_id}))

                        # ====== 举手确认逻辑集成 ======
                        if detect_hand_confirm(person["track_id"], smoothed_kpts):
                            print(f"Confirm Detected! Starting game {current_game_id}", flush=True)
                        skeleton_color = (0, 255, 0)  # Green
                        for j in skeleton_map:
                            s_id = j.get('srt_kpt_id')
                            d_id = j.get('dst_kpt_id')
                            sx = int(draw_kpts[s_id][0])
                            sy = int(draw_kpts[s_id][1])
                            dx = int(draw_kpts[d_id][0])
                            dy = int(draw_kpts[d_id][1])
                            if sx == 0 or sy == 0 or dx == 0 or dy == 0:
                                continue
                            cv2.line(frame, (sx, sy), (dx, dy), skeleton_color, 2)
                # 显示当前选中的游戏 ID
                cv2.putText(frame, f"Selected Game: {current_game_id}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255, 255, 0), 2)

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

                annotated = frame
            except Exception as e:
                # print(f"Error in preview_loop: {e}")
                annotated = frame

            # ==========================================================

            try:
                cv2.imshow(PREVIEW_WIN, annotated)
            except cv2.error:
                pass

            try:
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                key = 255

            if key == ord('q'):
                STOP_FLAG.set()
                preview_stop_evt.set()
                break
            elif key == ord('r'):  # R：清空所有矩形
                rects = []
                print("已清空所有矩形")
            elif key == ord('s'):  # S：手动保存数据
                save_rects_to_json(rects, json_path)
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
def start_preview():
    global preview_thread
    if preview_thread is None or not preview_thread.is_alive():
        preview_stop_evt.clear()
        preview_pause_evt.clear()
        preview_thread = threading.Thread(target=preview_loop, daemon=True)
        preview_thread.start()
    else:
        preview_pause_evt.clear()
def pause_preview():
    preview_pause_evt.set()
def stop_preview():
    global preview_thread
    preview_stop_evt.set()
    preview_pause_evt.clear()

    # 等预览线程自己 finally 里关闭窗口
    if preview_thread is not None:
        preview_thread.join(timeout=1.5)
def reader_thread(pid: int, p: subprocess.Popen):
    try:
        while not STOP_FLAG.is_set():
            line = p.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                EVENT_Q.put(("event", pid, obj))
            except Exception:
                EVENT_Q.put(("log", pid, line))
    except Exception as e:
        EVENT_Q.put(("log", pid, f"[reader_thread error] {e}"))
def send_cmd(pid: int, obj: dict) -> bool:
    p = PROCS.get(pid)
    if not p or p.poll() is not None:
        return False
    try:
        p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        p.stdin.flush()
        return True
    except Exception:
        return False
def shutdown_all():
    """ESC 时：全部结束（预览+子进程+兜底 kill）"""
    STOP_FLAG.set()

    # # 0) 停止预览
    # stop_preview()
    try:
        ws_send_queue.put_nowait(json.dumps({"cmd": "exit"}, ensure_ascii=False))
        ws_send_queue.put_nowait(None)  # 让 sender 线程跳出
    except Exception:
        pass

    # 1) 优雅 shutdown
    for pid in list(PROCS.keys()):
        send_cmd(pid, {"cmd": "shutdown"})
    time.sleep(0.2)

    # 2) terminate
    for pid, p in list(PROCS.items()):
        if p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    time.sleep(0.2)

    # 3) kill 兜底
    for pid, p in list(PROCS.items()):
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
def flush_queue():
    while True:
        try:
            EVENT_Q.get_nowait()
        except queue.Empty:
            break
def drain_and_print(current_id: int | None = None) -> bool:
    back_menu = False
    while True:
        try:
            kind, pid, payload = EVENT_Q.get_nowait()
        except queue.Empty:
            break

        if kind == "log":
            print(f"[{pid}] {payload}", flush=True)
            continue
        print(f"[{pid} EVENT] {payload}", flush=True)
        if (
                isinstance(payload, dict)
                and payload.get("event") == "back_to_menu"
                and current_id is not None
                and pid == current_id
        ):
            back_menu = True

    return back_menu
def start_child(pid: int):
    """启动或重启单个子进程"""
    # 如果已经在跑，先停掉
    stop_child(pid)

    script = SCRIPTS.get(pid)
    if not script or not os.path.exists(script):
        print(f"❌ 脚本不存在: {script}", flush=True)
        return

    print(f"启动 {script}", flush=True)
    p = subprocess.Popen(
        [sys.executable, script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(script)
    )
    PROCS[pid] = p
    threading.Thread(target=reader_thread, args=(pid, p), daemon=True).start()
def stop_child(pid: int):
    """停止单个子进程"""
    p = PROCS.get(pid)
    if p:
        print(f"正在停止子进程 {pid}...", flush=True)
        try:
            # 1. 尝试优雅退出
            send_cmd(pid, {"cmd": "exit"})
            time.sleep(0.3)
            if p.poll() is None:
                p.terminate()
                time.sleep(0.2)
            if p.poll() is None:
                p.kill()
        except Exception as e:
            print(f"停止子进程 {pid} 出错: {e}")
        finally:
            if pid in PROCS:
                del PROCS[pid]
def menu_loop() -> int | None:
    # 菜单态：开启预览
    start_preview()

    while not STOP_FLAG.is_set():
        drain_and_print(current_id=None)

        # 首先检查 INPUT_Q 中是否有来自 WebSocket 的命令
        try:
            # 设置较小的超时时间，避免阻塞太久
            line = INPUT_Q.get(timeout=0.1)
            if line is not None:
                cmd = str(line).strip().lower()
                print(f"[WebSocket] 收到命令: {cmd}")

                if cmd == "esc":
                    return None
                elif cmd == "q":
                    print("（当前在菜单，不需要回菜单）", flush=True)
                    continue
                elif cmd in ("1", "2", "3"):
                    return int(cmd)
        except queue.Empty:
            pass  # 没有来自 WebSocket 的命令，继续等待用户输入

        # 如果没有来自 WebSocket 的命令，则等待用户输入
        line = prompt_wait_line("\n输入 1/2/3 选择脚本；输入 q 回菜单；输入 esc 结束全部：", timeout=0.1)
        if line is None:
            return None

        cmd = line.strip().lower()

        if cmd == "esc":
            return None

        if cmd == "q":
            print("（当前在菜单，不需要回菜单）", flush=True)
            continue

        if cmd in ("1", "2", "3"):
            return int(cmd)

        print("无效输入，请输入 1/2/3 或 esc。", flush=True)

    return None
def running_loop(current_id: int) -> bool:
    # 运行态：暂停预览
    pause_preview()

    # 确保子进程是最新启动的（实现“重新进去还能进去”）
    print(f"正在准备游戏 {current_id}...", flush=True)
    start_child(current_id)
    time.sleep(0.5)  # 给一点启动时间

    print(f"选择了 {current_id}：通知前端切换到游戏 {current_id}", flush=True)

    # 发送指令给前端切换端口
    try:
        ws_send_queue.put_nowait(json.dumps({
            "cmd": "switch_game",
            "game_id": current_id,
            "port": 8765 + current_id  # 8766, 8767, 8768
        }))
    except Exception:
        pass

    while not STOP_FLAG.is_set():
        # 先处理并打印子进程的日志，这样如果子进程挂了，我们能看到报错
        back_to_menu_event = drain_and_print(current_id=current_id)

        # 检查子进程是否还在运行
        p = PROCS.get(current_id)
        if p is None or p.poll() is not None:
            # 进程退出后，最后尝试再读一次残留的日志
            time.sleep(0.1)
            drain_and_print(current_id=current_id)
            print(f"\n❌ 子进程 {current_id} 已退出，回到主页面。", flush=True)
            return True

        if back_to_menu_event:
            print("\n收到子进程 back_to_menu：回到主页面。", flush=True)
            stop_child(current_id)
            return True

        # 检查 INPUT_Q 中是否有来自 WebSocket 的命令
        try:
            line = INPUT_Q.get(timeout=0.1)
            if line is not None:
                cmd = str(line).strip().lower()
                if cmd == "esc":
                    return False
                elif cmd == "q":
                    print("收到 q 命令，正在退出游戏...", flush=True)
                    stop_child(current_id)
                    # 通知前端返回主页
                    try:
                        ws_send_queue.put_nowait(json.dumps({
                            "cmd": "return_to_main",
                            "from_game": current_id
                        }))
                    except Exception:
                        pass
                    return True
        except queue.Empty:
            pass

        # 终端输入兜底
        line = prompt_wait_line("运行中：输入 q 回菜单；输入 esc 结束全部；", timeout=0.1)
        if line:
            cmd = line.strip().lower()
            if cmd == "esc": return False
            if cmd == "q":
                stop_child(current_id)
                return True

    return False
def main():
    global yolo_model

    # ✅ 启动后台输入/ESC监听（关键）
    threading.Thread(target=stdin_reader, daemon=True).start()

    # 1) 启动器加载 YOLO，并打印
    yolo_model = dg.load_model(**aaa)
    print(f"YOLO 已加载: {MODEL_PATH}", flush=True)

    # 4) 主状态机：菜单态 <-> 运行态
    while True:
        if STOP_FLAG.is_set():
            print("收到 ESC：结束全部脚本...", flush=True)
            shutdown_all()
            break

        flush_queue()
        pid = menu_loop()
        if pid is None:
            print("收到 esc/ESC：结束全部脚本...", flush=True)
            shutdown_all()
            break

        flush_queue()
        go_menu = running_loop(pid)
        if not go_menu:
            print("收到 esc/ESC：结束全部脚本...", flush=True)
            shutdown_all()
            break


if __name__ == "__main__":
    main()
