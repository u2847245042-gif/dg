# utils.py
import json
import math
import os
import numpy as np
from datetime import datetime
import logging
import sys


video_path = 41
# True 显示 ,False 不显示
if_show_dds = False
if_show_123 = False
if_show_kht = False
if_show_gtt = True
if_show_server = False

# 矩形类：保持原有属性
class Rect:
    def __init__(self, points, locked=False, name=""):
        self.points = points  # [左上,右上,右下,左下]
        self.selected_point = -1  # 选中的顶点索引(-1=未选)
        self.selected_whole = False  # 是否整体选中矩形
        self.locked = locked  # 是否锁定
        self.name = name  # 名称
        self.drag_offset = (0, 0)  # 整体拖动偏移量
        self.count = 0
        self.count_if = False
        self.in_if  = False

    def to_dict(self):
        """将矩形对象转为字典（用于JSON序列化）"""
        return {
            "points": self.points,
            "locked": self.locked,
            "name": self.name,
        }
    @staticmethod
    def from_dict(data):
        """从字典创建矩形对象（用于JSON反序列化）"""
        return Rect(points=data["points"], locked=data["locked"], name=data["name"])
def distance(p1, p2):
    """计算两点距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def get_rect_center(points):
    """计算矩形中心点"""
    x = (points[0][0] + points[2][0]) // 2
    y = (points[0][1] + points[2][1]) // 2
    return (x, y)
def save_rects_to_json(rects, json_path):
    """将矩形列表保存为JSON文件"""
    rects_dict = [rect.to_dict() for rect in rects]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rects_dict, f, indent=2)
    print(f"矩形数据已保存到: {json_path}")
def load_rects_from_json(json_path):
    """从JSON文件加载矩形列表，无文件则返回空列表"""
    if not os.path.exists(json_path):
        print(f"未找到JSON文件: {json_path}，将新建矩形")
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            rects_dict = json.load(f)
        rects = [Rect.from_dict(data) for data in rects_dict]
        print(f"从{json_path}加载了 {len(rects)} 个矩形")
        return rects
    except Exception as e:
        print(f"加载JSON失败: {e}，将新建矩形")
        return []
# 日志记录
def _init_logger(log_dir):
    """初始化日志：终端 + 文件，按天切分，线程安全"""
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名：logs/20251023.log
    log_file = os.path.join(log_dir, f"{datetime.now():%Y%m%d}.log")

    # 格式：1023-14:26:55 [PID] 消息
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(process)d] %(message)s",
        datefmt="%m%d-%H:%M:%S"
    )

    # 同时写终端 + 文件
    h1 = logging.StreamHandler(sys.stderr)
    h2 = logging.FileHandler(log_file, encoding="utf-8")

    for h in (h1, h2):
        h.setFormatter(fmt)

    logger = logging.getLogger("smp")
    logger.setLevel(logging.INFO)
    logger.addHandler(h1)
    logger.addHandler(h2)
    return logger

# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 2},  # 右侧脚踝-右侧膝盖
    {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 2},  # 右侧膝盖-右侧胯
    {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 2},  # 左侧脚踝-左侧膝盖
    {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 2},  # 左侧膝盖-左侧胯
    {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 2},  # 右侧胯-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 2},  # 右边肩膀-右侧胯
    {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 2},  # 左边肩膀-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 2},  # 右边肩膀-左边肩膀
    {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 2},  # 右边肩膀-右侧胳膊肘
    {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 2},  # 左边肩膀-左侧胳膊肘
    {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 2},  # 右侧胳膊肘-右侧手腕
    {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 2},  # 左侧胳膊肘-左侧手腕
    {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 2},  # 右边眼睛-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 2},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 2},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 2},  # 右边眼睛-右边耳朵
    {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 2},  # 左边眼睛-左边耳朵
    {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 2},  # 右边耳朵-右边肩膀
    {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 2}  # 左边耳朵-左边肩膀
]
base_dir = os.path.dirname(os.path.abspath(__file__))
aaa = {
"model_name":'yolov8n_relu6_coco_pose--640x640_quant_rknn_rk3588_1',
"inference_host_address":'@local',
"zoo_url":base_dir,
}
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x = None
        self.dx = 0.0

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * float(cutoff))
        return float(dt) / (float(dt) + tau)

    def filter(self, x, dt):
        x = float(x)
        dt = float(dt) if dt > 1e-8 else 1e-3  # 保证 dt 是一个有效的小正数

        if self.x is None:
            self.x = x
            self.dx = 0.0  # 初始化 dx
            return x

        # -------- 核心修正：计算速度时要除以 dt --------
        # 原始错误代码: dx = x - self.x
        # 修正后代码:
        dx_raw = (x - self.x) / dt
        # -------------------------------------------

        # 使用修正后的速度来更新滤波后的速度
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx_raw + (1.0 - a_d) * self.dx

        # 使用滤波后的速度来计算截止频率
        cutoff = self.min_cutoff + self.beta * abs(self.dx)

        # 使用新的截止频率来计算平滑因子
        a = self._alpha(cutoff, dt)

        # 更新滤波后的位置
        self.x = a * x + (1.0 - a) * self.x

        return self.x
class OneEuro2D:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def filter(self, pt, dt):
        x = self.fx.filter(pt[0], dt)
        y = self.fy.filter(pt[1], dt)
        return [x, y]
class CentroidTracker:
    def __init__(self, max_dist=160, max_missed=15):
        self.tracks = {}
        self.archive = {}  # Store lost tracks for re-identification
        self.next_id = 1
        self.max_dist = float(max_dist)
        self.max_missed = int(max_missed)
        self.max_ids = 5  # Limit to 5 people as per user request

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _compute_similarity(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0.0
        h1 = np.array(hist1, dtype=np.float32)
        h2 = np.array(hist2, dtype=np.float32)
        return np.sum(np.minimum(h1, h2))

    def update(self, persons):
        # 1. Manage existing tracks (mark missed)
        if not persons:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    # Move to archive instead of deleting
                    self.archive[tid] = self.tracks[tid]
                    del self.tracks[tid]
            return persons

        det_ctrs = [self._centroid(p["bbox"]) for p in persons]

        # We will perform matching in two steps:
        # Step 1: Match with ACTIVE tracks (missed == 0) using spatial + appearance.
        # Step 2: Match remaining detections with INACTIVE/ARCHIVED tracks using appearance mainly.

        active_track_ids = [tid for tid in self.tracks if self.tracks[tid]["missed"] == 0]
        drift_track_ids = [tid for tid in self.tracks if self.tracks[tid]["missed"] > 0]

        # --- STAGE 1: Active Tracks Matching ---
        assigned_det_indices = set()
        assigned_track_ids = set()

        if active_track_ids:
            # Create Cost Matrix for Active Tracks
            D_active = np.full((len(active_track_ids), len(persons)), 1000.0, dtype=np.float32)

            for i, tid in enumerate(active_track_ids):
                track = self.tracks[tid]

                # Predict next position
                c = track["centroid"]
                v = track.get("velocity", (0, 0))
                pred = (c[0] + v[0], c[1] + v[1])

                t_avg_area = track.get("avg_area", 0)
                t_hist = track.get("hist")

                for j, person in enumerate(persons):
                    if j in assigned_det_indices: continue

                    dc = det_ctrs[j]
                    d_bbox = person["bbox"]
                    d_area = (d_bbox[2] - d_bbox[0]) * (d_bbox[3] - d_bbox[1])
                    d_hist = person.get("hist")

                    # 1. Spatial Distance
                    dist = math.hypot(pred[0] - dc[0], pred[1] - dc[1])
                    cost_dist = dist / 100.0

                    # 2. IoU
                    t_bbox = track["bbox"]
                    xA = max(t_bbox[0], d_bbox[0])
                    yA = max(t_bbox[1], d_bbox[1])
                    xB = min(t_bbox[2], d_bbox[2])
                    yB = min(t_bbox[3], d_bbox[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (t_bbox[2] - t_bbox[0]) * (t_bbox[3] - t_bbox[1])
                    boxBArea = d_area
                    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
                    cost_iou = (1.0 - iou) * 5.0

                    # 3. Appearance (Critical to prevent swap)
                    cost_app = 0.0
                    sim = self._compute_similarity(t_hist, d_hist)
                    if sim < 0.75:  # Hard constraint for active tracks too
                        cost_app = 100.0  # Penalty
                    else:
                        cost_app = (1.0 - sim) * 5.0

                    # Total Cost
                    D_active[i, j] = cost_dist + cost_iou + cost_app

            # Solve assignment for active
            # Simple greedy
            matches = []
            for i in range(len(active_track_ids)):
                for j in range(len(persons)):
                    matches.append((D_active[i, j], i, j))
            matches.sort(key=lambda x: x[0])

            for cost, r, c in matches:
                if active_track_ids[r] in assigned_track_ids or c in assigned_det_indices:
                    continue
                if cost > 20.0: continue  # Threshold

                tid = active_track_ids[r]
                self._update_track(tid, persons[c], det_ctrs[c])
                assigned_track_ids.add(tid)
                assigned_det_indices.add(c)

        # --- STAGE 2: Re-Identification (Drifting + Archived) ---
        # Try to match remaining detections to Drifting tracks OR Archived tracks
        # Priority: Drifting (recently lost) > Archived (long lost)
        # BUT: Must strictly follow Appearance.

        unassigned_dets = [j for j in range(len(persons)) if j not in assigned_det_indices]

        if unassigned_dets:
            # Combine drift and archive candidates
            candidates = []
            # Add drifting
            for tid in drift_track_ids:
                if tid not in assigned_track_ids:
                    candidates.append((tid, self.tracks[tid], "drift"))
            # Add archive
            for tid, track in self.archive.items():
                candidates.append((tid, track, "archive"))

            if candidates:
                # Greedy match based on APPEARANCE ONLY
                # We do NOT use distance for archived tracks (they can re-enter anywhere)
                # We use loose distance for drifting tracks? No, user said "4 leaves, 3 enters -> don't rely on distance"
                # So we rely on APPEARANCE.

                reid_matches = []
                for cand_idx, (tid, track, status) in enumerate(candidates):
                    t_hist = track.get("hist")
                    for j in unassigned_dets:
                        d_hist = persons[j].get("hist")
                        sim = self._compute_similarity(t_hist, d_hist)

                        # STRICT THRESHOLD for ReID
                        # User wants "ID bound to death", so we need high confidence to re-bind.
                        # But we also need to distinguish between 3 and 4.
                        # If sim > 0.85, it's a strong candidate.
                        if sim > 0.85:
                            reid_matches.append((sim, cand_idx, j))

                # Sort by similarity DESCENDING
                reid_matches.sort(key=lambda x: x[0], reverse=True)

                used_cand_indices = set()

                for sim, cand_idx, det_idx in reid_matches:
                    if cand_idx in used_cand_indices or det_idx in assigned_det_indices:
                        continue

                    tid, track, status = candidates[cand_idx]

                    # Resurrect
                    if status == "archive":
                        self.tracks[tid] = track  # Move back to tracks
                        del self.archive[tid]

                    self._update_track(tid, persons[det_idx], det_ctrs[det_idx])
                    assigned_track_ids.add(tid)
                    assigned_det_indices.add(det_idx)
                    used_cand_indices.add(cand_idx)

        # --- STAGE 3: Create New Tracks ---
        for j in range(len(persons)):
            if j not in assigned_det_indices:
                # Only create new ID if we haven't reached limit (or if we really have to)
                # User says "Video has 5 people... ID bound to death".
                # If we have < 5 known IDs, create new.
                # If we have 5 known IDs, and this person didn't match any...
                # This implies either:
                # 1. It's a false detection.
                # 2. Appearance changed drastically.
                # 3. Our threshold 0.85 is too high.

                # Strategy: If < 5 total IDs (active + archive), create new.
                # If >= 5, we try to force match to the "closest" available ID in archive/drift based on appearance?
                # Or we just assign a temporary new ID?
                # User said "Cannot produce 5 after ID". (No IDs > 5).

                total_ids = len(self.tracks) + len(self.archive)
                if total_ids < self.max_ids:
                    tid = self.next_id
                    self.next_id += 1
                    self._create_track(tid, persons[j], det_ctrs[j])
                else:
                    # We have 5 IDs already. This person MUST be one of them.
                    # Find the best match among ALL unavailable IDs (drift + archive)
                    # even if similarity is < 0.85 (but still reasonable?)

                    best_tid = -1
                    best_sim = -1.0

                    # Check drift
                    for tid in drift_track_ids:
                        if tid not in assigned_track_ids:
                            sim = self._compute_similarity(self.tracks[tid].get("hist"), persons[j].get("hist"))
                            if sim > best_sim:
                                best_sim = sim
                                best_tid = tid

                    # Check archive
                    for tid in self.archive:
                        sim = self._compute_similarity(self.archive[tid].get("hist"), persons[j].get("hist"))
                        if sim > best_sim:
                            best_sim = sim
                            best_tid = tid

                    # If we found a "reasonable" match (e.g. > 0.6?), take it.
                    # If it's garbage (<0.5), maybe it's a false detection?
                    # But better to assign an ID than nothing if it's a person.
                    if best_tid != -1 and best_sim > 0.6:
                        if best_tid in self.archive:
                            self.tracks[best_tid] = self.archive[best_tid]
                            del self.archive[best_tid]
                        self._update_track(best_tid, persons[j], det_ctrs[j])
                    else:
                        # What to do? Create ID 6? User said NO.
                        # But if we don't, we lose tracking.
                        # Maybe force ID creation but cap at 5? No, next_id is global.
                        # Let's create it for now, but log a warning.
                        # Or maybe re-use the oldest archive ID?
                        # Let's stick to "try best match". If really fail, create new (maybe user count is wrong?)
                        # But strictly following user: "Cannot produce 5 after ID".
                        # So I will NOT create ID > 5. I will force match to best available.
                        if best_tid != -1:
                            if best_tid in self.archive:
                                self.tracks[best_tid] = self.archive[best_tid]
                                del self.archive[best_tid]
                            self._update_track(best_tid, persons[j], det_ctrs[j])
                        else:
                            # Should not happen if we have candidates.
                            # If NO candidates (all 5 are active?), then this is a 6th person?
                            # Ignore or create new.
                            pass

        # Update missed counts for unmatched tracks
        for tid in list(self.tracks.keys()):
            if tid not in assigned_track_ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    self.archive[tid] = self.tracks[tid]
                    del self.tracks[tid]

        # Assign IDs to persons object
        # Reverse mapping: find which person got which tid
        # Actually I updated tracks using `persons[c]`, but didn't set `persons[c]["track_id"]`
        # Wait, I need to set `persons[j]["track_id"] = tid`

        # Filter out persons who did not get an ID (e.g. 6th person when limit is 5)
        assigned_persons = [p for p in persons if "track_id" in p]

        return assigned_persons

    def _create_track(self, tid, person, centroid):
        d_bbox = person["bbox"]
        d_w = d_bbox[2] - d_bbox[0]
        d_h = d_bbox[3] - d_bbox[1]
        self.tracks[tid] = {
            "centroid": centroid,
            "bbox": d_bbox,
            "kpts": person.get("kpts", []),
            "velocity": (0, 0),
            "avg_area": d_w * d_h,
            "avg_ar": d_w / d_h if d_h > 0 else 0,
            "hist": person.get("hist"),
            "missed": 0
        }
        person["track_id"] = tid

    def _update_track(self, tid, person, new_c):
        track = self.tracks[tid]
        old_c = track["centroid"]

        # Update Velocity
        raw_v = (new_c[0] - old_c[0], new_c[1] - old_c[1])
        old_v = track.get("velocity", (0, 0))
        smooth_v = (0.7 * old_v[0] + 0.3 * raw_v[0], 0.7 * old_v[1] + 0.3 * raw_v[1])

        # Update History
        d_bbox = person["bbox"]
        d_w = d_bbox[2] - d_bbox[0]
        d_h = d_bbox[3] - d_bbox[1]
        curr_area = d_w * d_h
        curr_ar = d_w / d_h if d_h > 0 else 0

        old_avg_area = track.get("avg_area", curr_area)
        old_avg_ar = track.get("avg_ar", curr_ar)

        new_avg_area = 0.9 * old_avg_area + 0.1 * curr_area
        new_avg_ar = 0.9 * old_avg_ar + 0.1 * curr_ar

        # Update Histogram (Smooth)
        t_hist = track.get("hist")
        d_hist = person.get("hist")
        new_hist = d_hist
        if t_hist is not None and d_hist is not None:
            h1 = np.array(t_hist, dtype=np.float32)
            h2 = np.array(d_hist, dtype=np.float32)
            new_hist = (0.95 * h1 + 0.05 * h2).tolist()  # Slower update for hist to keep identity stable
        elif t_hist is not None:
            new_hist = t_hist

        self.tracks[tid].update({
            "centroid": new_c,
            "bbox": d_bbox,
            "kpts": person.get("kpts", []),
            "velocity": smooth_v,
            "avg_area": new_avg_area,
            "avg_ar": new_avg_ar,
            "hist": new_hist,
            "missed": 0
        })
        person["track_id"] = tid

