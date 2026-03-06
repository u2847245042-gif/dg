# -*- coding: utf-8 -*-
"""
高抬腿计数脚本 dg_gtt.py
逻辑：检测膝盖是否抬过髋部（knee_y < hip_y），
      左腿或右腿各抬一次算一个完整计数（左→右 或 右→左 均可）。
框架结构与 dg_kht.py 保持一致。
"""
import cv2, threading, queue, time, json, os, base64, math, copy, sys,datetime
import numpy as np
import degirum as dg

# 添加父目录到 sys.path 以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    Rect, distance, get_rect_center,
    save_rects_to_json, load_rects_from_json,
    _init_logger, skeleton_map,
    aaa, OneEuroFilter, OneEuro2D, CentroidTracker,
    video_path, if_show_gtt
)

from websockets.sync.client import connect
ip_id = 8769
# ======================== WebSocket 发送线程 ==========================
ws_send_queue = queue.Queue()
def ws_sender_thread():
    """后台线程：维持一个 WebSocket 连接，持续发送队列里的消息"""
    uri = f"ws://127.0.0.1:{ip_id}"
    while True:
        try:
            print("尝试连接 WebSocket 服务器:", uri)
            with connect(uri) as websocket:
                print("dg_gtt 已连接 WebSocket 服务器")
                while True:
                    msg = ws_send_queue.get()
                    if msg is None:
                        break
                    try:
                        websocket.send(msg)
                    except Exception as e:
                        print("发送失败:", e)
                        break
        except Exception as e:
            print("连接 WebSocket 失败，将在 2 秒后重试:", e)
            time.sleep(2)
# ======================== WebSocket 接收线程 ==========================
ws_receiver_queue = queue.Queue()
def ws_receiver_thread():
    global rects, persons, marked_ids
    while True:
        try:
            uri = f"ws://127.0.0.1:{ip_id}"
            with connect(uri) as ws:
                print("[gtt] 接收通道已连接")
                for msg in ws:
                    cmd = json.loads(msg).get("cmd")
                    if cmd in ("clear_marked1", "clear_marked2"):
                        # 矩形计数清零
                        for rect in rects:
                            rect.count = 0
                            rect.count_if = False
                            if hasattr(rect, "saved_avatar"):
                                rect.saved_avatar = False
                        # 人员标记清零
                        for p in persons:
                            p["is_marked"] = False
                        # 状态机清零
                        pose_history.clear()
                        kpt_filters_bank.clear()
                        cy_filter_bank.clear()
                        marked_ids.clear()
                        person_action_state.clear()
                        person_knee_count.clear()
                        print("[gtt] 已清零 rects.count & persons.is_marked & marked_ids")
                    elif json.loads(msg).get("cmd") == "gtt_window":
                        # 切换深蹲窗口显示/隐藏
                        if toggle_window_flag.is_set():
                            toggle_window_flag.clear()
                        else:
                            toggle_window_flag.set()
                        print(f"[dg_kht] dss_window 切换，当前={'显示' if toggle_window_flag.is_set() else '隐藏'}")
        except Exception as e:
            print("[gtt] 接收通道断开，2 秒后重连：", e)
            time.sleep(2)
threading.Thread(target=ws_sender_thread, daemon=True).start()
threading.Thread(target=ws_receiver_thread, daemon=True).start()

# ======================== 鼠标回调（同 dg_kht.py）===================
def mouse_callback(event, x, y, flags, param):
    global rects, drawing_new, temp_start, temp_end
    for rect in rects:
        if rect.locked:
            rect.selected_point = -1
            rect.selected_whole = False
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = False
        for rect in rects:
            if rect.locked:
                continue
            for i, (px, py) in enumerate(rect.points):
                if distance((x, y), (px, py)) < 10:
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
            for rect in rects:
                if rect.locked:
                    continue
                if cv2.pointPolygonTest(np.array(rect.points), (x, y), False) >= 0:
                    for r in rects:
                        if r != rect:
                            r.selected_point = -1
                            r.selected_whole = False
                    rect.selected_whole = True
                    center = get_rect_center(rect.points)
                    rect.drag_offset = (center[0] - x, center[1] - y)
                    clicked = True
                    break
        if not clicked:
            drawing_new = True
            temp_start = (x, y)
            temp_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        for rect in rects:
            if rect.locked or rect.selected_point == -1:
                continue
            rect.points[rect.selected_point] = (x, y)
        for rect in rects:
            if rect.locked or not rect.selected_whole:
                continue
            new_center = (x + rect.drag_offset[0], y + rect.drag_offset[1])
            old_center = get_rect_center(rect.points)
            dx = new_center[0] - old_center[0]
            dy = new_center[1] - old_center[1]
            for i in range(4):
                rect.points[i] = (rect.points[i][0] + dx, rect.points[i][1] + dy)
        if drawing_new:
            temp_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        for rect in rects:
            rect.selected_point = -1
            rect.selected_whole = False
        if drawing_new:
            drawing_new = False
            new_rect_points = [
                temp_start, (temp_end[0], temp_start[1]),
                temp_end, (temp_start[0], temp_end[1])
            ]
            rects.append(Rect(new_rect_points, locked=False))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if rects:
            for i in reversed(range(len(rects))):
                if not rects[i].locked:
                    del rects[i]
                    break
# ======================== 工具函数 ====================================
def is_box_bottom_in_rect(bbox, rect: Rect):
    x1, y1, x2, y2 = bbox
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)
    bottom_center = ((x1 + x2) / 2.0, y2)
    pts_rect = np.array(rect.points, np.int32)
    for px, py in [bottom_left, bottom_center, bottom_right]:
        if cv2.pointPolygonTest(pts_rect, (float(px), float(py)), False) >= 0:
            return True
    return False
def compute_hist(frame, bbox):
    x1, y1, x2, y2 = bbox
    H, W = frame.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W - 1, int(x2)); y2 = min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32).tolist()
def crop_head_avatar(frame, kpts, lr_ratio=0.35, top_ratio=0.8, bottom_ratio=0.2):
    try:
        xs = [kpts[3][0], kpts[4][0], kpts[5][0], kpts[6][0]]
        ys = [kpts[3][1], kpts[4][1], kpts[5][1], kpts[6][1]]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))
        w = xmax - xmin; h = ymax - ymin
        xmin = int(xmin - w * lr_ratio)
        xmax = int(xmax + w * lr_ratio)
        ymin = int(ymin - h * top_ratio)
        ymax = int(ymax + h * bottom_ratio)
        H, W, _ = frame.shape
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(W, xmax); ymax = min(H, ymax)
        if xmax > xmin and ymax > ymin:
            return frame[ymin:ymax, xmin:xmax].copy()
        return None
    except:
        return None
def img_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buffer).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'
def smooth_kpts_one_euro(person, dt, key_name="track_id",min_cutoff=1.2, beta=0.006, d_cutoff=1.0,conf_thresh=0.3, max_jump=60):
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
        kpt_filters_bank[pid] = [
            OneEuro2D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(N)
        ]
    prev = pose_history[pid]
    smooth = prev.copy()
    for i in range(N):
        x_new, y_new = kpts[i]
        x_old, y_old = prev[i]
        if scores[i] < conf_thresh:
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

# ======================== 高抬腿核心计数逻辑 =========================
"""
COCO 关键点索引（17 点）：
  0: 鼻子   1: 左眼   2: 右眼   3: 左耳   4: 右耳
  5: 左肩   6: 右肩   7: 左肘   8: 右肘
  9: 左腕  10: 右腕  11: 左胯  12: 右胯
 13: 左膝  14: 右膝  15: 左踝  16: 右踝

高抬腿判定：
  - 膝盖 Y 坐标 < 髋部 Y 坐标（图像坐标系，Y 越小越靠上）
  - 同时膝盖高度超过髋部与肩部中点的一定比例（可调）

计数策略（双腿交替，每两步计 1 次）：
  状态机 per box_id：
    idle  -> left_up  : 左膝抬起
    left_up -> idle   : 左膝放下（完成左腿，等右腿）
    idle  -> right_up : 右膝抬起（前提：左腿已完成过一次）
    right_up -> idle  : 右膝放下 → count+1，重置左腿标记
  或反过来先右后左同样计数。
  简化版：任意一腿抬起放下算 1 步，2 步 = 1 次完整高抬腿。
"""
def detect_knee_raise(smoothed_kpts, box_id, dt):
    """
    返回 (left_raised, right_raised, hip_threshold_y)
    left_raised / right_raised: 该腿膝盖当前是否抬过髋部
    """
    # 关键点索引
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_SHO, R_SHO = 5, 6

    lhip_x, lhip_y   = smoothed_kpts[L_HIP]
    rhip_x, rhip_y   = smoothed_kpts[R_HIP]
    lkne_x, lkne_y   = smoothed_kpts[L_KNEE]
    rkne_x, rkne_y   = smoothed_kpts[R_KNEE]
    lsho_x, lsho_y   = smoothed_kpts[L_SHO]
    rsho_x, rsho_y   = smoothed_kpts[R_SHO]

    # 有效性检查（坐标为 0 视为无效）
    l_valid = lhip_y > 0 and lkne_y > 0
    r_valid = rhip_y > 0 and rkne_y > 0

    # 计算髋部 + 肩部中线，用于动态阈值
    hip_y_avg = (lhip_y + rhip_y) / 2.0 if (lhip_y > 0 and rhip_y > 0) else max(lhip_y, rhip_y)
    sho_y_avg = (lsho_y + rsho_y) / 2.0 if (lsho_y > 0 and rsho_y > 0) else min(lsho_y, rsho_y)

    # 动态阈值：膝盖必须抬到髋部以上（负数 = 辅助线在髋部上方）
    # -0.10 表示膝盖需超过髋部 10% 躯干高度，可根据实际调整
    torso_h = abs(hip_y_avg - sho_y_avg) if sho_y_avg > 0 else 50
    delta = int(torso_h * 0.20)

    # 判断：膝盖 Y < 髋部 Y + delta（Y 越小越高）
    left_raised  = l_valid and (lkne_y < lhip_y + delta)
    right_raised = r_valid and (rkne_y < rhip_y + delta)

    # 返回辅助线所需信息：各髋部坐标 + delta 阈值线 Y
    extra = {
        "lhip_x": lhip_x, "lhip_y": lhip_y,
        "rhip_x": rhip_x, "rhip_y": rhip_y,
        "lkne_x": lkne_x, "lkne_y": lkne_y,
        "rkne_x": rkne_x, "rkne_y": rkne_y,
        "threshold_y_left":  lhip_y + delta,   # 左腿：膝盖需低于此 Y 值才算抬起
        "threshold_y_right": rhip_y + delta,   # 右腿：同上
        "delta": delta,
        "l_valid": l_valid, "r_valid": r_valid,
    }
    return left_raised, right_raised, extra

# ======================== 初始化 ======================================
MY_ID = 4
log_dir = "dg_logs"
log = _init_logger(log_dir)
model = dg.load_model(**aaa)
json_path = "dg_gtt.json"          # 复用同一个区域配置文件
rects = load_rects_from_json(json_path)
drawing_new = False
temp_start = (-1, -1)
temp_end = (-1, -1)
conf_number = 0.5

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
time.sleep(0.5)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("警告：无法获取视频的FPS，将使用默认值 30。")
    fps = 30
if fps > 30:
    print(f"检测到 FPS={fps} 较高，强制限制逻辑 FPS 为 30 以适配处理速度")
    fps = 30

wait_time_ms = int(1000 / fps)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ======================== 全局状态 ====================================
pose_history = {}
kpt_filters_bank = {}
cy_filter_bank = {}
prev_ts = None
marked_ids = set()
tracker = CentroidTracker(max_dist=600, max_missed=30)
persons = []

# 高抬腿专用状态
# person_action_state[box_id] = {
#   "left_was_up":  bool,   左腿上一帧是否抬起
#   "right_was_up": bool,   右腿上一帧是否抬起
#   "left_done":    bool,   当前轮左腿已完成一次抬起-放下
#   "right_done":   bool,   当前轮右腿已完成一次抬起-放下
# }
person_action_state = {}
person_knee_count = {}    # box_id -> 高抬腿总计数

# ============= 录制视频设置 =============
recording = False
video_writer = None
now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
output_path = f"{formatted_time}_gtt.avi"
# ============= 窗口显示控制 =============
show_window = if_show_gtt
toggle_window_flag = threading.Event()
if show_window:
    toggle_window_flag.set()
while True:
    # 检查是否需要切换窗口显示状态
    want_show = toggle_window_flag.is_set()
    if want_show != show_window:
        show_window = want_show
        if show_window:
            cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Pose Estimation", 1920, 1200)
            cv2.setMouseCallback("Pose Estimation", mouse_callback, param=(frame_h, frame_w))
            print("[dg_kht] 显示窗口")
        else:
            cv2.destroyWindow("Pose Estimation")
            cv2.waitKey(1)  # ← 必须调用一次，窗口才真正关闭
            print("[dg_kht] 隐藏窗口")
    ret, frame = cap.read()
    if not ret:
        break
    ts = time.time()
    dt = ts - prev_ts if prev_ts is not None else 1.0 / fps
    prev_ts = ts
    # 推理
    inference_result = model(frame)
    persons = []
    for p in inference_result.results:
        score = p.get("score")
        label = p.get("label")
        if label == "person":
            bbox = p.get("bbox")
            l, s = [], []
            for lm in p.get("landmarks"):
                l.append(lm.get("landmark"))
                s.append(lm.get("score"))
            if score > conf_number:
                persons.append({
                    "bbox": bbox,
                    "kpts": l,
                    "scores": s,
                    "score": score,
                    "low_conf": False,
                    "in_if": False
                })
    persons.sort(key=lambda person: person["bbox"][0])
    for person in persons:
        bbox = person["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        in_rect_id = None
        for rid, rect in enumerate(rects):
            if is_box_bottom_in_rect(bbox, rect):
                person["in_rect_id"] = rid
                person["in_if"] = True
                in_rect_id = rid
                break

        if in_rect_id is None:
            continue

        # ---- 标记逻辑（同 dg_kht.py）----
        if in_rect_id in marked_ids:
            person["is_marked"] = True
        else:
            person["is_marked"] = False

        kpts_raw = person["kpts"]
        if len(kpts_raw) > 10:
            l_sh = kpts_raw[5]; r_sh = kpts_raw[6]
            l_wr = kpts_raw[9]; r_wr = kpts_raw[10]
            l_raised_hand = (l_wr[1] < l_sh[1]) and (l_wr[1] > 0) and (l_sh[1] > 0)
            r_raised_hand = (r_wr[1] < r_sh[1]) and (r_wr[1] > 0) and (r_sh[1] > 0)
            if l_raised_hand or r_raised_hand:
                marked_ids.add(in_rect_id)
                person["is_marked"] = True

        rect = rects[in_rect_id]
        if not rect.name:
            rect.name = f"box{in_rect_id + 1}"
        name = rect.name
        person["name"] = name

        # ---- 关键点平滑 ----
        smoothed_kpts = smooth_kpts_one_euro(
            person, dt,
            key_name="in_rect_id",
            min_cutoff=1.2, beta=0.1, d_cutoff=1.0,
            conf_thresh=0.2, max_jump=150
        )

        box_id = in_rect_id

        # ---- 初始化状态 ----
        if box_id not in person_action_state:
            person_action_state[box_id] = {
                "left_was_up":      False,
                "right_was_up":     False,
                "left_up_pending":  False,  # 左腿已抬起，等待放下
                "right_up_pending": False,  # 右腿已抬起，等待放下
                "left_done":        False,  # 左腿完成一次完整抬放
                "right_done":       False,  # 右腿完成一次完整抬放
            }
        if box_id not in person_knee_count:
            person_knee_count[box_id] = 0

        rect.count_if = False
        state = person_action_state[box_id]

        # ---- 高抬腿检测 ----
        left_raised, right_raised, knee_extra = detect_knee_raise(smoothed_kpts, box_id, dt)

        # 左腿：「抬起」只有在 left_done 未消费时才接受，防止单腿重复累积
        if left_raised and not state["left_was_up"] and not state["left_done"]:
            state["left_up_pending"] = True
        if not left_raised and state["left_was_up"] and state["left_up_pending"]:
            state["left_done"]       = True   # 放下才算完成
            state["left_up_pending"] = False

        # 右腿：同理
        if right_raised and not state["right_was_up"] and not state["right_done"]:
            state["right_up_pending"] = True
        if not right_raised and state["right_was_up"] and state["right_up_pending"]:
            state["right_done"]       = True  # 放下才算完成
            state["right_up_pending"] = False

        # 左右腿都完成完整「抬起→放下」→ 计数 +1，全部重置
        if state["left_done"] and state["right_done"]:
            person_knee_count[box_id] += 1
            rect.count += 1
            rect.count_if = True
            state["left_done"]        = False
            state["right_done"]       = False
            state["left_up_pending"]  = False
            state["right_up_pending"] = False
            # print(f"[{name}] 高抬腿 +1，总数: {person_knee_count[box_id]}")

        # 更新上一帧状态
        state["left_was_up"]  = left_raised
        state["right_was_up"] = right_raised

        log.info(f"person_knee_count:{person_knee_count}")
        log.info(f"left_raised:{left_raised}, right_raised:{right_raised}, state:{state}")
        log.info(f"box_id:{box_id}, count={rect.count}, count_if:{rect.count_if}\n")

        # ---- 头像截取 ----
        head_img = crop_head_avatar(frame, smoothed_kpts)
        if head_img is not None:
            person["head_img"] = head_img
            person["head_base64"] = img_to_base64(head_img)
        if rect.in_if and not getattr(rect, "saved_avatar", False):
            rect.saved_avatar = True
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("avatars", exist_ok=True)
            save_path = f"avatars/{name}_{ts_str}.jpg"
            if head_img is not None:
                cv2.imwrite(save_path, head_img)
                print(f"✅ 已保存头像: {save_path}")

        # ---- 画框 + 文字 ----
        count_text = f"Knee: {person_knee_count[box_id]}"
        cv2.putText(frame, count_text, (x1, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        l_color = (0, 0, 255) if left_raised  else (200, 200, 200)
        r_color = (0, 0, 255) if right_raised else (200, 200, 200)
        cv2.putText(frame, f"L:{int(left_raised)} R:{int(right_raised)}",
                    (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---- 画骨架 ----
        for j in skeleton_map:
            s_id = j.get('srt_kpt_id')
            d_id = j.get('dst_kpt_id')
            sx, sy = int(smoothed_kpts[s_id][0]), int(smoothed_kpts[s_id][1])
            dx, dy = int(smoothed_kpts[d_id][0]), int(smoothed_kpts[d_id][1])
            if sx == 0 or sy == 0 or dx == 0 or dy == 0:
                continue
            cv2.line(frame, (sx, sy), (dx, dy), (0, 100, 255), 2)

        # ---- 高亮膝盖关键点 ----
        for kid, color in [(13, l_color), (14, r_color)]:
            kx, ky = int(smoothed_kpts[kid][0]), int(smoothed_kpts[kid][1])
            if kx > 0 and ky > 0:
                cv2.circle(frame, (kx, ky), 8, color, -1)

        # ---- 辅助线：显示膝盖需要抬到的目标高度 ----
        # 以 bbox 左右边界为线段范围，画在髋部 + delta 的 Y 位置
        thr_y_l = int(knee_extra["threshold_y_left"])
        thr_y_r = int(knee_extra["threshold_y_right"])
        lhip_x_i = int(knee_extra["lhip_x"])
        rhip_x_i = int(knee_extra["rhip_x"])

        # 左腿辅助线：绿色=已抬到位，红色=未到位，灰色=关键点无效
        if knee_extra["l_valid"] and lhip_x_i > 0:
            line_x1 = x1
            line_x2 = (x1 + x2) // 2
            lline_color = (0, 255, 0) if left_raised else (0, 80, 255)
            cv2.line(frame, (line_x1, thr_y_l), (line_x2, thr_y_l), lline_color, 2, cv2.LINE_AA)
            # 左侧小箭头标注
            cv2.putText(frame, "L", (line_x1 - 18, thr_y_l + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, lline_color, 2)

        # 右腿辅助线
        if knee_extra["r_valid"] and rhip_x_i > 0:
            line_x1 = (x1 + x2) // 2
            line_x2 = x2
            rline_color = (0, 255, 0) if right_raised else (0, 80, 255)
            cv2.line(frame, (line_x1, thr_y_r), (line_x2, thr_y_r), rline_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "R", (line_x2 + 4, thr_y_r + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, rline_color, 2)

        # 髋部实际位置（虚线圆点标记）
        lhip_yi = int(knee_extra["lhip_y"])
        rhip_yi = int(knee_extra["rhip_y"])
        if lhip_x_i > 0 and lhip_yi > 0:
            cv2.circle(frame, (lhip_x_i, lhip_yi), 5, (255, 200, 0), -1)
        if rhip_x_i > 0 and rhip_yi > 0:
            cv2.circle(frame, (rhip_x_i, rhip_yi), 5, (255, 200, 0), -1)

        # 膝盖当前位置到辅助线的距离（调试用，可注释掉）
        lkne_yi = int(knee_extra["lkne_y"])
        rkne_yi = int(knee_extra["rkne_y"])
        if knee_extra["l_valid"] and lkne_yi > 0:
            gap_l = lkne_yi - thr_y_l   # 正值=还差多少，负值=已超过
            cv2.putText(frame, f"{gap_l:+d}px", (x1, thr_y_l - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if knee_extra["r_valid"] and rkne_yi > 0:
            gap_r = rkne_yi - thr_y_r
            cv2.putText(frame, f"{gap_r:+d}px", ((x1 + x2) // 2, thr_y_r - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    # ---- 绘制所有矩形 ----
    for idx, rect in enumerate(rects):
        if rect.locked:
            border_color = (128, 128, 128)
        elif rect.selected_point != -1 or rect.selected_whole:
            border_color = (255, 0, 0)
        else:
            border_color = (0, 255, 0)
        cv2.polylines(frame, [np.array(rect.points, np.int32)],
                      isClosed=True, color=border_color, thickness=2)
        for i, (px, py) in enumerate(rect.points):
            if rect.locked:
                pt_color = (128, 128, 128)
            elif rect.selected_point == i:
                pt_color = (0, 255, 255)
            else:
                pt_color = (0, 0, 255)
            cv2.circle(frame, (px, py), 5, pt_color, -1)
        px, py = rect.points[0]
        label = rect.name if rect.name else f"box{idx + 1}"
        cv2.putText(frame, f"{label}-{rect.count}",
                    (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # ---- WebSocket 广播 ----
    try:
        payload = {
            "boxes": [
                {
                    "id": idx,
                    "name": rect.name if rect.name else f"box{idx + 1}",
                    "count": rect.count,
                    "count_event": rect.count_if,
                    "in_if": any(
                        p.get("in_if", False)
                        for p in persons
                        if p.get("in_rect_id") == idx
                    ),
                    "head_img": next((
                        p.get("head_base64")
                        for p in persons
                        if p.get("in_rect_id") == idx and "head_base64" in p
                    ), None),
                    "is_marked": any(
                        p.get("is_marked", False)
                        for p in persons
                        if p.get("in_rect_id") == idx
                    ),
                }
                for idx, rect in enumerate(rects)
            ]
        }
        ws_send_queue.put_nowait(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
    # ---- 临时矩形 ----
    if drawing_new and temp_start != (-1, -1):
        temp_points = [
            temp_start, (temp_end[0], temp_start[1]),
            temp_end, (temp_start[0], temp_end[1])
        ]
        cv2.polylines(frame, [np.array(temp_points, np.int32)],
                      isClosed=True, color=(255, 0, 0), thickness=1,
                      lineType=cv2.LINE_AA)
    if recording and video_writer is not None:
        video_writer.write(frame)
    if show_window:
        cv2.imshow("Pose Estimation", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        try:
            ws_send_queue.put_nowait(json.dumps({"cmd": "exit"}))
        except:
            pass
        break
    elif key == ord('r'):
        rects = []
        print("已清空所有矩形")
    elif key == ord('c'):
        for rect in rects:
            rect.count = 0
            rect.count_if = False
        person_action_state.clear()
        person_knee_count.clear()
        print("已清零计数")
    elif key == ord('s'):
        save_rects_to_json(rects, json_path)
        print("已保存矩形配置")
    elif key == ord('v'):
        recording = not recording
        if recording:
            print("🎬 开始录制视频...")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(
                output_path, fourcc, 5,
                (frame.shape[1], frame.shape[0])
            )
        else:
            print("🛑 停止录制")
            if video_writer is not None:
                video_writer.release()
                video_writer = None
if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()