# -*- coding: utf-8 -*-
import cv2, threading, queue, time, json, os, base64, math, copy, sys
import numpy as np
import degirum as dg

# 添加父目录到 sys.path 以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Rect, distance, get_rect_center, save_rects_to_json, load_rects_from_json, _init_logger, skeleton_map, \
aaa, OneEuroFilter, OneEuro2D, CentroidTracker,video_path,if_show_kht

from websockets.sync.client import connect
ip_id = 8768
ws_send_queue = queue.Queue()
def ws_sender_thread():
    """后台线程：维持一个 WebSocket 连接，持续发送队列里的消息"""
    uri = f"ws://127.0.0.1:{ip_id}"
    while True:
        try:
            print("尝试连接 WebSocket 服务器:", uri)
            with connect(uri) as websocket:
                print("dg_dds 已连接 WebSocket 服务器")

                # ✅ 新增：连接成功后先清空积压的旧消息，避免前端重连时收到旧状态
                cleared = 0
                while not ws_send_queue.empty():
                    try:
                        ws_send_queue.get_nowait()
                        cleared += 1
                    except:
                        break
                if cleared > 0:
                    print(f"[dg_dds] 清空积压消息 {cleared} 条，避免旧状态误触")

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
ws_receiver_queue = queue.Queue()
def ws_receiver_thread():
    global rects, persons, marked_ids  # 声明要改的全局变量
    while True:
        try:
            uri = f"ws://127.0.0.1:{ip_id}"
            with connect(uri) as ws:
                print("[test] 接收通道已连接")
                for msg in ws:
                    if json.loads(msg).get("cmd") == "clear_marked1" or json.loads(msg).get("cmd") == "clear_marked2":
                        # 1. 矩形计数清零
                        for rect in rects:
                            rect.count = 0
                            rect.count_if = False
                            # 如果还想重新截头像，把 saved_avatar 也复位
                            if hasattr(rect, "saved_avatar"):
                                rect.saved_avatar = False

                        # 2. 人员标记清零
                        for p in persons:
                            p["is_marked"] = False

                        # 状态机 & 阈值也清（可选）
                        pose_history.clear()
                        kpt_filters_bank.clear()
                        cy_filter_bank.clear()
                        marked_ids.clear()
                        person_action_state.clear()
                        person_jump_count.clear()

                        print("[test] 已清零 rects.count & persons.is_marked & marked_ids")
        except Exception as e:
            print("[test] 接收通道断开，2 秒后重连：", e)
            time.sleep(2)
threading.Thread(target=ws_sender_thread, daemon=True).start()
threading.Thread(target=ws_receiver_thread, daemon=True).start()

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
def is_box_bottom_in_rect(bbox, rect: Rect):
    """
    判断 bbox 的下边 3 个关键点 是否在矩形区域内：
    1. 下边中心点
    2. 左下角
    3. 右下角
    满足任意一个即可
    """
    x1, y1, x2, y2 = bbox

    # 下边 3 个点
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)
    bottom_center = ((x1 + x2) / 2.0, y2)

    check_points = [bottom_left, bottom_center, bottom_right]

    # 矩形点
    pts_rect = np.array(rect.points, np.int32)

    # 只要任意一个点在矩形内部（>=0）则返回 True
    for px, py in check_points:
        inside = cv2.pointPolygonTest(
            pts_rect, (float(px), float(py)), False
        )
        if inside >= 0:
            return True

    return False
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
def crop_head_avatar(frame, kpts, lr_ratio=0.35, top_ratio=0.8, bottom_ratio=0.2):
    """
    基于关键点裁剪头像（耳朵+肩膀）
    返回：head_img 或 None
    """
    try:
        # 左右耳 + 左右肩
        xs = [
            kpts[3][0], kpts[4][0],  # 耳朵
            kpts[5][0], kpts[6][0]  # 肩膀
        ]
        ys = [
            kpts[3][1], kpts[4][1],
            kpts[5][1], kpts[6][1]
        ]

        xmin = int(min(xs))
        xmax = int(max(xs))
        ymin = int(min(ys))  # 更靠上（耳朵）
        ymax = int(max(ys))  # 更靠下（肩膀）

        w = xmax - xmin
        h = ymax - ymin

        # 左右扩一点
        xmin = int(xmin - w * lr_ratio)
        xmax = int(xmax + w * lr_ratio)

        # 上下不对称扩展：上边多截，下面少截
        ymin = int(ymin - h * top_ratio)
        ymax = int(ymax + h * bottom_ratio)

        H, W, _ = frame.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W, xmax)
        ymax = min(H, ymax)

        if xmax > xmin and ymax > ymin:
            return frame[ymin:ymax, xmin:xmax].copy()
        else:
            return None
    except Exception as e:
        # 可以临时打印一下看看有没有异常
        # print("crop_head_avatar error:", e)
        return None
def img_to_base64(img):
    """OpenCV 图片 -> base64 字符串"""
    _, buffer = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buffer).decode('ascii')
    data_uri = f'data:image/jpeg;base64,{b64}'
    return data_uri
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
MY_ID = 3
log_dir = f"dg_logs"
log = _init_logger(log_dir)
model = dg.load_model(**aaa)
json_path = "dg_fj.json"
rects = load_rects_from_json(json_path)
drawing_new = False
temp_start = (-1, -1)
temp_end = (-1, -1)
conf_number = 0.5
show_window = if_show_dds

# ==========================================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
time.sleep(0.5)
# 1. 获取视频的原始FPS
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("警告：无法获取视频的FPS，将使用默认值 30。")
    fps = 30
# 强制限制 fps 用于逻辑计算的最大值，避免因相机fps过高导致hold帧数过大
if fps > 30:
    print(f"检测到 FPS={fps} 较高，强制限制逻辑 FPS 为 30 以适配处理速度")
    fps = 30

# 2. 计算每帧应该等待的时间（单位：毫秒）
wait_time_ms = int(1000 / fps)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if if_show_kht == True:
    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Estimation", 1920, 1200)
    cv2.setMouseCallback("Pose Estimation", mouse_callback, param=(frame_h, frame_w))

# ==========================================================
pose_history = {}
kpt_filters_bank = {}
cy_filter_bank = {}
prev_ts = None
marked_ids = set()
tracker = CentroidTracker(max_dist=600, max_missed=30)
# 用于存储每个人的动作状态 ("closed" 或 "open")
person_action_state = {}
# 用于存储每个人的计数值
person_jump_count = {}

# ============= 录制视频设置 =============
recording = False
video_writer = None
output_path = "output_record.avi"
while True:
    key = 0xFF
    ret, frame = cap.read()
    if not ret:
        break
    ts = time.time()
    dt = ts - prev_ts if prev_ts is not None else 1.0 / fps
    prev_ts = ts
    # 推理
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
                    "low_conf": False,
                    "in_if": False
                })
    # 按 bbox 左上角 x1 从小到大排序
    persons.sort(key=lambda person: person["bbox"][0])
    # ==========================================================
    # 画框 + 写名字 + 画骨架：一人一框一处理
    for person in persons:
        bbox = person["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # 判断属于哪个矩形框
        in_rect_id = None
        for rid, rect in enumerate(rects):
            if is_box_bottom_in_rect(bbox, rect):
                person["in_rect_id"] = rid
                person["in_if"] = True
                in_rect_id = rid
                break
        if in_rect_id is not None:
            # Check existing mark
            if in_rect_id in marked_ids:
                person["is_marked"] = True
            else:
                person["is_marked"] = False
            all_in_box = (len(persons) > 0)
            # If all in box, check for new hand raises
            if all_in_box:
                kpts = person["kpts"]
                if len(kpts) > 10:
                    # 5:LSh, 6:RSh, 9:LWri, 10:RWri
                    l_sh = kpts[5]
                    r_sh = kpts[6]
                    l_wr = kpts[9]
                    r_wr = kpts[10]

                    # Check left hand (wrist above shoulder -> smaller y)
                    l_raised = (l_wr[1] < l_sh[1]) and (l_wr[1] > 0) and (l_sh[1] > 0)
                    r_raised = (r_wr[1] < r_sh[1]) and (r_wr[1] > 0) and (r_sh[1] > 0)

                    if l_raised or r_raised:
                        marked_ids.add(in_rect_id)
                        person["is_marked"] = True

            rect = rects[person["in_rect_id"]]
            # 如果这个矩形还没有名字，就按矩形索引生成一次
            if not rect.name:
                rect.name = f"box{in_rect_id + 1}"
            name = rect.name  # 人的名字 = 所属矩形的名字
            person["name"] = name  # 如果后面还需要用到，可以存回去

            smoothed_kpts = smooth_kpts_one_euro(person,dt,key_name="in_rect_id",min_cutoff=1.2,beta=0.1,d_cutoff=1.0,conf_thresh=0.2,max_jump=150)
            # ========== ① 求 bbox 中心点 ==========
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # ========== ② 画横线（基于 bbox 宽度）==========
            if in_rect_id not in cy_filter_bank:
                cy_filter_bank[in_rect_id] = OneEuroFilter(min_cutoff=0.5, beta=0.02, d_cutoff=1.0)
            cx_smooth = int(cy_filter_bank[in_rect_id].filter(float(cx), dt))
            cy_smooth = int(cy_filter_bank[in_rect_id].filter(float(cy), dt))

            box_id = in_rect_id
            # 左右手腕
            lw_x, lw_y = smoothed_kpts[9]
            rw_x, rw_y = smoothed_kpts[10]
            # 左右肩膀
            ls_x, ls_y = smoothed_kpts[5]
            rs_x, rs_y = smoothed_kpts[6]
            # 左右脚踝
            la_x, la_y = smoothed_kpts[15]
            ra_x, ra_y = smoothed_kpts[16]

            # 定义坐标
            A = smoothed_kpts[15]
            B = smoothed_kpts[16]
            C = smoothed_kpts[12]

            # 计算向量
            CA = (A[0] - C[0], A[1] - C[1])
            CB = (B[0] - C[0], B[1] - C[1])

            # 点积
            dot_product = CA[0] * CB[0] + CA[1] * CB[1]

            # 模长
            len_CA = math.hypot(CA[0], CA[1])
            len_CB = math.hypot(CB[0], CB[1])

            # 计算余弦值（避免浮点误差，限制在[-1,1]）
            cos_theta = dot_product / (len_CA * len_CB)
            cos_theta = max(min(cos_theta, 1.0), -1.0)

            # 弧度转角度
            theta_rad = math.acos(cos_theta)
            theta_deg = math.degrees(theta_rad)  # 最终角度值
            # print(f"∠ACB的角度：{theta_deg:.2f}度")  # 输出：20.79度

            # 动态计算手臂容差 (基于头肩距离)
            head_candidates = []
            for hid in [0, 1, 2, 3, 4]:
                hx, hy = smoothed_kpts[hid]
                if hx > 0 and hy > 0:
                    head_candidates.append(hy)
            # 头部Y坐标 (取最高点，即最小Y)
            head_y = min(head_candidates) if head_candidates else (ls_y + rs_y) / 2.0
            shoulder_y = (ls_y + rs_y) / 2.0
            # delta容差：允许手臂稍微低于肩膀也算"抬起"
            # 增加容差系数到 0.2 (20% 头肩距)
            delta = max(6, int(abs(shoulder_y - head_y) * 0.30))
            # 判断手臂抬起 / 放下状态
            # w_y_s (Start/Up): 手腕 < 肩膀 + 容差 (Y越小越靠上)
            w_y_s = (lw_y < ls_y + delta) and (rw_y < rs_y + delta)
            # w_y_x (End/Down): 手腕 > 肩膀 - 容差
            w_y_x = (lw_y > ls_y - delta) or (rw_y > rs_y - delta)
            # print(f"w_y_s：{w_y_s},w_y_x：{w_y_x}")

            # 1. 计算基准肩宽 (ref)
            if ls_x > 0 and rs_x > 0:
                shoulder_width = abs(rs_x - ls_x)
            else:
                shoulder_width = (x2 - x1) * 0.5  # 如果没有肩膀，估算为bbox宽度的一半

            # 2. 计算脚踝间距
            if la_x > 0 and ra_x > 0:
                ankle_dist = abs(ra_x - la_x)
            else:
                ankle_dist = 0

            # 3. 根据间距与肩宽的比例，判断腿部状态
            is_leg_open = ankle_dist > shoulder_width * 0.9  # 脚踝间距 > 90% 肩宽，视为打开
            is_leg_close = ankle_dist < shoulder_width * 0.4  # 脚踝间距 < 40% 肩宽，视为闭合

            # 确保每个人都有初始状态和计数值
            if box_id not in person_action_state:
                person_action_state[box_id] = "closed"  # 初始状态默认为闭合
            if box_id not in person_jump_count:
                person_jump_count[box_id] = 0
            rect.count_if = False
            # 获取当前状态
            current_state = person_action_state[box_id]
            # ----- 核心计数逻辑 -----
            # 状态：从 "closed" 转换到 "open"
            # 条件：当前是闭合状态，且检测到腿打开、手抬起
            if current_state == "closed" and is_leg_open  and w_y_s:
                person_action_state[box_id] = "open"
                # print(f"[{name}] 状态: closed -> open") # 调试用

            # 状态：从 "open" 转换到 "closed"
            # 条件：当前是打开状态，且检测到腿闭合、手放下
            elif current_state == "open" and is_leg_close and w_y_x:
                person_action_state[box_id] = "closed"
                # 完成了一次完整的开合跳，计数+1
                person_jump_count[box_id] += 1
                rect.count += 1
                rect.count_if = True
                # print(f"[{name}] 状态: open -> closed, 计数 +1, 总数: {person_jump_count[box_id]}") # 调试用

            log.info(f"person_jump_count:{person_jump_count}")
            log.info(f"w_y_s:{w_y_s},w_y_x:{w_y_x},ankle_dist:{ankle_dist},theta_deg:{theta_deg},person_action_state:{person_action_state}")
            log.info(f"box_id:{box_id},box{box_id + 1},count = {rect.count}, count_if:{rect.count_if}\n")

            # ===== ✅ 自动截图头像（进入框时可截）=====
            head_img = crop_head_avatar(frame, smoothed_kpts)
            if head_img is not None:
                person["head_img"] = head_img
                person["head_base64"] = img_to_base64(head_img)
            # ===== ✅ 进入框第一次才截图，防止狂存 =====
            # 只在第一次进入这个框时保存头像
            if rect.in_if and not getattr(rect, "saved_avatar", False):
                rect.saved_avatar = True  # ✅ 标记这个框已经保存过

                ts = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs("avatars", exist_ok=True)
                save_path = f"avatars/{name}_{ts}.jpg"

                if head_img is not None:
                    cv2.imwrite(save_path, head_img)
                    print(f"✅ 已保存头像: {save_path}")

            # ----- 在图像上显示计数 -----
            count_text = f"Count: {person_jump_count[box_id]}"
            # 将计数值显示在名字旁边
            cv2.putText(frame, count_text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ----- 在图像上显示当前动作状态（可选，用于调试）-----
            state_text = f"Action: {person_action_state[box_id]}"
            color = (0, 255, 0) if person_action_state[box_id] == "closed" else (0, 0, 255)
            cv2.putText(frame, state_text, (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # 画这个人的骨架
            for j in skeleton_map:
                s_id = j.get('srt_kpt_id')
                d_id = j.get('dst_kpt_id')
                sx = int(smoothed_kpts[s_id][0])
                sy = int(smoothed_kpts[s_id][1])
                dx = int(smoothed_kpts[d_id][0])
                dy = int(smoothed_kpts[d_id][1])
                if sx == 0 or sy == 0 or dx == 0 or dy == 0:
                    continue
                cv2.line(frame, (sx, sy), (dx, dy), (0, 100, 255), 2)
        else:
            pass

    # 绘制所有矩形
    for idx, rect in enumerate(rects):
        # 边框颜色：锁定(灰色)、未锁定(绿色)、选中(蓝色)
        if rect.locked:
            border_color = (128, 128, 128)
        elif rect.selected_point != -1 or rect.selected_whole:
            border_color = (255, 0, 0)  # 选中的矩形边框变蓝
        else:
            border_color = (0, 255, 0)
        # 绘制边框
        cv2.polylines(frame, [np.array(rect.points, np.int32)], isClosed=True, color=border_color, thickness=2)
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
        cv2.putText(frame, f"{label}-{rect.count}", (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ========= 在 while True 循环里、画完所有矩形之后 =========
    try:
        payload = {
            "boxes": [
                {
                    "id": idx,
                    "name": rect.name if rect.name else f"box{idx + 1}",
                    "count": rect.count,
                    "count_event": rect.count_if,
                    "in_if": any(p.get("in_if", False)
                                     for p in persons
                                     if p.get("in_rect_id") == idx),
                    "head_img": next((
                        p.get("head_base64")
                        for p in persons
                        if p.get("in_rect_id") == idx and "head_base64" in p
                    ), None),
                    "is_marked": any(p.get("is_marked", False)
                                     for p in persons
                                     if p.get("in_rect_id") == idx),
                }
                for idx, rect in enumerate(rects)
            ]
        }
        ws_send_queue.put_nowait(json.dumps(payload, ensure_ascii=False))
        # log.info(f"payload:{payload}")
    except Exception as e:
        # 队列满了或者其它异常，先忽略，不影响主循环
        pass
    # ==========================================================

    # 绘制临时矩形(蓝色虚线)
    if drawing_new and temp_start != (-1, -1):
        temp_points = [temp_start, (temp_end[0], temp_start[1]), temp_end, (temp_start[0], temp_end[1])]
        cv2.polylines(frame, [np.array(temp_points, np.int32)], isClosed=True, color=(255, 0, 0), thickness=1,
                      lineType=cv2.LINE_AA)
    # ========== 如果正在录制，则写入当前帧 ==========
    if recording and video_writer is not None:
        video_writer.write(frame)

    if if_show_kht == True:
        cv2.imshow("Pose Estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 发送退出消息通知 ws_server.py
            try:
                ws_send_queue.put_nowait(json.dumps({"cmd": "exit"}))
            except:
                pass
            break
        elif key == ord('r'):  # R：清空所有矩形
            rects = []
            print("已清空所有矩形")
        elif key == ord('c'):  # C：清空数据
            for rect in rects:
                rect.count = 0
                rect.count_if = False
            person_action_state.clear()
            person_jump_count.clear()
        elif key == ord('s'):  # S：手动保存数据
            save_rects_to_json(rects, json_path)
        elif key == ord('v'):  # 按 v 开始/停止录制
            recording = not recording
            if recording:
                print("🎬 开始录制视频...")

                # 创建 VideoWriter（注意要用 MJPG，否则容易打不开）
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = VideoWriter = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    5,  # fps
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
