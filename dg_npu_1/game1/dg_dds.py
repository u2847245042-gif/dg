import cv2, threading, queue, time, json, os, base64, math, copy, sys, datetime
import numpy as np
import degirum as dg

# 添加父目录到 sys.path 以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Rect, distance, get_rect_center, save_rects_to_json, load_rects_from_json, _init_logger, skeleton_map, \
aaa, OneEuroFilter, OneEuro2D, CentroidTracker,video_path,if_show_dds

from websockets.sync.client import connect
ws_send_queue = queue.Queue()
ip_id = 8766
def ws_sender_thread():
    """后台线程：维持一个 WebSocket 连接，持续发送队列里的消息"""
    uri = f"ws://127.0.0.1:{ip_id}"   # 如果前端在别的机器，这里改成 RK3588 的 IP
    while True:
        try:
            print("尝试连接 WebSocket 服务器:", uri)
            with connect(uri) as websocket:
                print("dg_dds 已连接 WebSocket 服务器")

                while True:
                    # 阻塞到有消息要发送
                    msg = ws_send_queue.get()
                    if msg is None:
                        break  # 预留退出用
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
    global rects, persons, marked_ids          # 声明要改的全局变量
    while True:
        try:
            uri = f"ws://127.0.0.1:{ip_id}"
            with connect(uri) as ws:
                print("[test] 接收通道已连接")
                for msg in ws:
                    if json.loads(msg).get("cmd") == "clear_marked":
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

                        # 3. 全局举手集合清零
                        marked_ids.clear()

                        # 4. 状态机 & 阈值也清（可选）
                        pose_history.clear()
                        kpt_filters_bank.clear()
                        cy_filter_bank.clear()
                        track_filter_bank.clear()
                        cy_body_filter_bank.clear()
                        prev_cy_body_dict.clear()
                        motion_info.clear()
                        person_squat_state.clear()
                        box_thresholds.clear()

                        print("[test] 已清零 rects.count & persons.is_marked & marked_ids")
                    elif json.loads(msg).get("cmd") == "dss_window":
                        # 切换深蹲窗口显示/隐藏
                        if toggle_window_flag.is_set():
                            toggle_window_flag.clear()
                        else:
                            toggle_window_flag.set()
                        print(f"[dg_dds] dss_window 切换，当前={'显示' if toggle_window_flag.is_set() else '隐藏'}")
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
def are_feet_in_rect(keypoints, rect: Rect):
    """
    判断人的脚部关键点 (15号和16号) 是否在指定的矩形区域 (rect) 内。

    Args:
        keypoints (list): 包含所有身体关键点的列表。
                          假设索引15是左脚踝，16是右脚踝。
                          每个关键点格式为 (x, y) 或 (x, y, confidence)。
        rect (Rect): 一个表示任意四边形区域的对象，拥有 points 属性。

    Returns:
        bool: 如果任意一个脚部关键点在矩形内，则返回 True，否则返回 False。
    """
    # 检查关键点列表是否足够长，以避免索引越界
    if len(keypoints) < 17:
        # 如果关键点数量不足，可以根据你的业务逻辑返回 False 或抛出异常
        return False

    # 提取15号和16号关键点 (脚踝)
    # 我们只取 x, y 坐标，忽略可能存在的置信度分数
    left_ankle = (keypoints[15][0], keypoints[15][1])
    right_ankle = (keypoints[16][0], keypoints[16][1])

    check_points = [left_ankle, right_ankle]

    # 将矩形的顶点转换为OpenCV需要的格式 (int32)
    pts_rect = np.array(rect.points, np.int32)

    # 遍历要检查的脚部关键点
    for px, py in check_points:
        # 使用 cv2.pointPolygonTest 判断点是否在多边形内部
        # measureDist=False 表示我们不关心距离，只关心点是否在内部、外部或边上
        # 返回值: > 0 (内部), == 0 (边上), < 0 (外部)
        is_inside = cv2.pointPolygonTest(
            contour=pts_rect,
            pt=(float(px), float(py)),
            measureDist=False
        )

        # 如果点在矩形内部或边上 (>= 0)，则立即返回 True
        if is_inside >= 0:
            return True

    # 如果所有检查点都不在矩形内，则返回 False
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
log_dir = f"dg_logs"
log = _init_logger(log_dir)
model = dg.load_model(**aaa)
json_path = "dds.json"
rects = load_rects_from_json(json_path)
drawing_new = False
temp_start = (-1, -1)
temp_end = (-1, -1)
conf_number = 0.4
# ==========================================================
cap = cv2.VideoCapture(video_path)
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
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# ==========================================================
pose_history = {}
kpt_filters_bank = {}
cy_filter_bank = {}
track_filter_bank = {}
cy_body_filter_bank = {}
prev_cy_body_dict = {}
motion_info = {}
prev_ts = None
marked_ids = set()
tracker = CentroidTracker(max_dist=600, max_missed=30)
prev_cy = None
person_squat_state = {}      # 存每个 box 的状态（"up"/"down"）
box_thresholds = {}          # 存每个 box 的 cy_up/cy_down
# ============= 录制视频设置 =============
recording = False
video_writer = None
now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
output_path = f"{formatted_time}_dds.avi"
# ============= 窗口显示控制 =============
show_window = if_show_dds
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
            print("[dg_dds] 显示窗口")
        else:
            cv2.destroyWindow("Pose Estimation")
            cv2.waitKey(1)  # ← 必须调用一次，窗口才真正关闭
            print("[dg_dds] 隐藏窗口")
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
    for rect in rects:
        rect.in_if = False
    # 画框 + 写名字 + 画骨架：一人一框一处理
    for person in persons:
        bbox = person["bbox"]
        kpts = person["kpts"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # 判断属于哪个矩形框
        in_rect_id = None
        for rid, rect in enumerate(rects):
            if are_feet_in_rect(kpts, rect):
                person["in_rect_id"] = rid
                person["in_if"] = True
                in_rect_id = rid
                rect.in_if = True
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
                    scores = person["scores"]
                    CONF = 0.5
                    l_raised = (l_wr[1] < l_sh[1]) and scores[9] > CONF and scores[5] > CONF
                    r_raised = (r_wr[1] < r_sh[1]) and scores[10] > CONF and scores[6] > CONF

                    if l_raised or r_raised:
                        marked_ids.add(in_rect_id)
                        person["is_marked"] = True

            person["in_rect_id"] = in_rect_id  # ✅ 补上这个
            rect = rects[person["in_rect_id"]]
            # 如果这个矩形还没有名字，就按矩形索引生成一次
            if not rect.name:
                rect.name = f"box{in_rect_id + 1}"
            name = rect.name  # 人的名字 = 所属矩形的名字
            person["name"] = name  # 如果后面还需要用到，可以存回去
            smoothed_kpts = smooth_kpts_one_euro(
                person,
                dt,
                key_name="in_rect_id",
                min_cutoff=1.0,
                beta=0.05,
                d_cutoff=1.0,
                conf_thresh=0.4,
                max_jump=100
            )

            # ========== ① 求 bbox 中心点 ==========
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # ========== ② 画横线（基于 bbox 宽度）==========
            if in_rect_id not in cy_filter_bank:
                cy_filter_bank[in_rect_id] = OneEuroFilter(min_cutoff=0.5, beta=0.02, d_cutoff=1.0)
            cx_smooth = int(cy_filter_bank[in_rect_id].filter(float(cx), dt))
            cy_smooth = int(cy_filter_bank[in_rect_id].filter(float(cy), dt))

            H = y2 - y1
            s_l_y = smoothed_kpts[5][1]
            s_r_y = smoothed_kpts[6][1]
            h_l_y = smoothed_kpts[11][1]
            h_r_y = smoothed_kpts[12][1]
            candidates = []
            if s_l_y > 1 and s_r_y > 1:
                candidates.append((s_l_y + s_r_y) / 2.0)
            if h_l_y > 1 and h_r_y > 1:
                candidates.append((h_l_y + h_r_y) / 2.0)
            candidates.append(y1 + 0.25 * H)
            candidates.sort()
            track_y = candidates[len(candidates)//2]
            if in_rect_id not in track_filter_bank:
                track_filter_bank[in_rect_id] = OneEuroFilter(min_cutoff=0.6, beta=0.02, d_cutoff=1.0)
            track_y = track_filter_bank[in_rect_id].filter(float(track_y), dt)

            box_id = in_rect_id
            if box_id not in person_squat_state:
                person_squat_state[box_id] = "up"

            # 固定基准线 + 迟滞（仅在更直立时更新基准线）
            if box_id not in box_thresholds:
                box_thresholds[box_id] = {
                    "min_track_y": track_y,
                    "stand_h": H,
                    "cy_down": track_y + 0.18 * H,
                    "cy_up": track_y + 0.14 * H,
                }
                person_squat_state[box_id] = "up"
            else:
                curr_min = box_thresholds[box_id]["min_track_y"]
                if track_y < curr_min - 3:
                    box_thresholds[box_id]["min_track_y"] = track_y
                    box_thresholds[box_id]["stand_h"] = H
                    base = track_y
                    box_thresholds[box_id]["cy_down"] = base + 0.18 * H
                    box_thresholds[box_id]["cy_up"] = base + 0.14 * H
            thr = box_thresholds[box_id]
            cy_down = thr["cy_down"]
            cy_up = thr["cy_up"]

            if not hasattr(rect, "in_frames"):
                rect.in_frames = 0
                rect.out_frames = 0
            rect.in_frames += 1
            state = person_squat_state[box_id]
            rect.count_if = False
            # ===== 深蹲状态机（需连续2帧） =====
            if "hold" not in rect.__dict__:
                rect.hold = {"down": 0, "up": 0}
            if state == "up":
                rect.hold["up"] = 0
                if rect.in_frames >= 1 and track_y > cy_down:
                    rect.hold["down"] += 1
                    if rect.hold["down"] >= 1:
                        person_squat_state[box_id] = "down"
                        rect.hold["down"] = 0
                else:
                    rect.hold["down"] = 0
            elif state == "down":
                rect.hold["down"] = 0
                if rect.in_frames >= 1 and track_y < cy_up:
                    rect.hold["up"] += 1
                    if rect.hold["up"] >= 1:
                        person_squat_state[box_id] = "up"
                        rect.count += 1
                        rect.count_if = True
                        rect.hold["up"] = 0
                else:
                    rect.hold["up"] = 0
            log.info(f"box_thresholds:{box_thresholds}")
            log.info(f"box_id:{box_id},box{box_id + 1},count = {rect.count}, cy_up:{cy_up}, cy_down:{cy_down}, count_if:{rect.count_if}, track_y:{track_y}")

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (x1, int(cy_up)), (x2, int(cy_up)), (255, 255, 0), 2)
            cv2.line(frame, (x1, int(cy_down)), (x2, int(cy_down)), (0, 0, 255), 2)
            cv2.line(frame, (x1, int(track_y)), (x2, int(track_y)), (255, 0, 255), 2)
            cv2.putText(frame, f"{name}-{person_squat_state[box_id]}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
        if not hasattr(rect, "in_frames"):
            rect.in_frames = 0
            rect.out_frames = 0
        if rect.in_if:
            rect.in_frames += 1
            rect.out_frames = 0
        else:
            rect.out_frames += 1
            rect.in_frames = 0
            if rect.out_frames >= 10:
                if idx in person_squat_state:
                    del person_squat_state[idx]
                if idx in box_thresholds:
                    del box_thresholds[idx]
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
        cv2.putText(frame, f"{label}-{rect.count}", (px, py - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # ========= 在 while True 循环里、画完所有矩形之后 =========
    try:
        payload = {
            "boxes": [
                {
                    "id": idx,
                    "name": rect.name if rect.name else f"box{idx + 1}",
                    "count_event": rect.count_if,
                    "in_if": rect.in_if,
                    "head_img": next((
                            p.get("head_base64")
                            for p in persons
                            if p.get("in_rect_id") == idx and "head_base64" in p
                        ),None),
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
        cv2.polylines(frame, [np.array(temp_points, np.int32)], isClosed=True, color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
    # ========== 如果正在录制，则写入当前帧 ==========
    if recording and video_writer is not None:
        video_writer.write(frame)
    if show_window:
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
    elif key == ord('c'):
        for rect in rects:
            rect.count = 0
            rect.count_if = False
            if hasattr(rect, "saved_avatar"):
                rect.saved_avatar = False  # 允许重新截一次头像
        person_squat_state.clear()
        box_thresholds.clear()
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

