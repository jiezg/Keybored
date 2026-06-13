"""
特征提取器：从原始击键事件流提取5类特征
1. 驻留时间 (Dwell Time)
2. 飞行时间 (Flight Time: DD, UD, DU, UU)
3. 二元组统计 (Digraph Statistics)
4. 打字节奏 (Typing Rhythm)
5. 辅助特征 (Auxiliary Features)
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional


# 键位分类常量
CAT_ALPHA = 1      # 字母键
CAT_DIGIT = 2      # 数字键
CAT_SPACE = 3      # 空格
CAT_ENTER = 4      # 回车
CAT_BACKSPACE = 5  # 退格
CAT_TAB = 6        # Tab
CAT_LSHIFT = 7     # 左Shift
CAT_RSHIFT = 8     # 右Shift
CAT_CTRL = 9       # Ctrl
CAT_ALT = 10       # Alt
CAT_FUNC = 11      # 功能键
CAT_ARROW = 12     # 方向键

# 飞行时间阈值（ms），超过此值视为"中断"（用户思考、喝水等）
FLIGHT_TIME_THRESHOLD = 750


class FeatureExtractor:
    """击键动力学特征提取器"""

    def extract(self, events: List[Dict]) -> Dict[str, np.ndarray]:
        """
        从事件流提取所有特征

        Args:
            events: 事件列表，每个事件包含 type(1=down,2=up), cat(键位分类), ts(时间戳ms)

        Returns:
            特征字典，包含5类特征
        """
        # 解析事件流，配对 KeyDown/KeyUp 得到驻留时间
        dwell_times, key_down_map = self._parse_dwell_times(events)

        # 计算飞行时间
        flight_times = self._compute_flight_times(events)

        # 二元组统计
        digraph_stats = self._compute_digraph_stats(flight_times)

        # 打字节奏
        rhythm = self._compute_typing_rhythm(events, dwell_times, flight_times)

        # 辅助特征
        auxiliary = self._compute_auxiliary_features(events, dwell_times, flight_times)

        return {
            "dwell_times": dwell_times,
            "flight_times": flight_times,
            "digraph_stats": digraph_stats,
            "rhythm": rhythm,
            "auxiliary": auxiliary,
        }

    def extract_feature_vector(self, events: List[Dict]) -> np.ndarray:
        """
        提取固定长度的特征向量（用于模型训练/推理）

        Returns:
            1D特征向量
        """
        features = self.extract(events)
        vector_parts = []

        # 1. 驻留时间统计量（按键位分类）
        dt_stats = self._stats_by_category(features["dwell_times"])
        vector_parts.append(dt_stats)

        # 2. 飞行时间统计量
        for ft_key in ["DD", "UD", "DU", "UU"]:
            ft_values = features["flight_times"].get(ft_key, [])
            ft_stats = self._compute_stats(ft_values)
            vector_parts.append(ft_stats)

        # 3. 二元组统计量（高频组合的延迟均值和标准差）
        dg = features["digraph_stats"]
        for key in sorted(dg.keys())[:20]:  # 取前20个高频二元组
            dg_stats = self._compute_stats(dg[key])
            vector_parts.append(dg_stats)

        # 4. 打字节奏特征
        r = features["rhythm"]
        rhythm_vector = [
            r.get("wpm", 0),
            r.get("avg_dwell", 0),
            r.get("std_dwell", 0),
            r.get("avg_flight_dd", 0),
            r.get("std_flight_dd", 0),
            r.get("pause_ratio", 0),
            r.get("avg_burst_length", 0),
        ]
        vector_parts.append(np.array(rhythm_vector))

        # 5. 辅助特征
        aux = features["auxiliary"]
        aux_vector = [
            aux.get("lshift_ratio", 0),
            aux.get("backspace_ratio", 0),
            aux.get("rollover_ratio", 0),
            aux.get("avg_dwell_alpha", 0),
            aux.get("avg_dwell_space", 0),
            aux.get("speed_variance", 0),
        ]
        vector_parts.append(np.array(aux_vector))

        # 拼接所有特征
        return np.concatenate(vector_parts)

    def _parse_dwell_times(self, events: List[Dict]) -> tuple:
        """
        解析事件流，配对KeyDown/KeyUp计算驻留时间

        Returns:
            (dwell_times_dict, key_down_map)
            dwell_times_dict: {键位分类: [驻留时间列表]}
            key_down_map: 用于飞行时间计算的按下事件序列
        """
        dwell_times = defaultdict(list)
        # 按键按下记录 {键位分类: 时间戳}，用于配对
        pending_down = {}
        # 按下事件序列（按时间排序），用于飞行时间计算
        key_down_seq = []

        for ev in events:
            cat = ev["cat"]
            ts = ev["ts"]

            if ev["type"] == 1:  # KEY_DOWN
                pending_down[cat] = ts
                key_down_seq.append({"cat": cat, "ts": ts})
            elif ev["type"] == 2:  # KEY_UP
                if cat in pending_down:
                    dwell = ts - pending_down[cat]
                    if dwell > 0:  # 过滤异常值
                        dwell_times[cat].append(dwell)
                    del pending_down[cat]

        return dict(dwell_times), key_down_seq

    def _compute_flight_times(self, events: List[Dict]) -> Dict[str, List[float]]:
        """
        计算4种飞行时间：DD, UD, DU, UU

        Returns:
            {"DD": [...], "UD": [...], "DU": [...], "UU": [...]}
        """
        # 将事件按时间排序，分离down和up事件
        down_events = []
        up_events = []

        for ev in events:
            if ev["type"] == 1:
                down_events.append(ev)
            elif ev["type"] == 2:
                up_events.append(ev)

        flight_times = {"DD": [], "UD": [], "DU": [], "UU": []}

        # DD: 第n键按下 → 第n+1键按下
        for i in range(len(down_events) - 1):
            ft = down_events[i + 1]["ts"] - down_events[i]["ts"]
            if 0 < ft < FLIGHT_TIME_THRESHOLD:
                flight_times["DD"].append(ft)

        # UD: 第n键释放 → 第n+1键按下
        for i in range(len(up_events)):
            for j in range(len(down_events)):
                if down_events[j]["ts"] > up_events[i]["ts"]:
                    ft = down_events[j]["ts"] - up_events[i]["ts"]
                    if 0 < ft < FLIGHT_TIME_THRESHOLD:
                        flight_times["UD"].append(ft)
                    break

        # DU: 第n键按下 → 第n+1键释放
        for i in range(len(down_events)):
            for j in range(len(up_events)):
                if up_events[j]["ts"] > down_events[i]["ts"]:
                    ft = up_events[j]["ts"] - down_events[i]["ts"]
                    if 0 < ft < FLIGHT_TIME_THRESHOLD:
                        flight_times["DU"].append(ft)
                    break

        # UU: 第n键释放 → 第n+1键释放
        for i in range(len(up_events) - 1):
            ft = up_events[i + 1]["ts"] - up_events[i]["ts"]
            if 0 < ft < FLIGHT_TIME_THRESHOLD:
                flight_times["UU"].append(ft)

        return flight_times

    def _compute_digraph_stats(self, flight_times: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        计算二元组统计量（按键位组合分组的飞行时间）

        Returns:
            {键位组合: [飞行时间列表]}
        """
        # 简化实现：直接按飞行时间类型分组
        # 完整实现需要记录具体的键位组合
        return flight_times

    def _compute_typing_rhythm(self, events, dwell_times, flight_times) -> Dict:
        """
        计算打字节奏特征

        Returns:
            打字节奏特征字典
        """
        down_events = [ev for ev in events if ev["type"] == 1]

        # 计算WPM（每分钟单词数，假设平均单词5个字符）
        total_duration_ms = 0
        if len(down_events) >= 2:
            total_duration_ms = down_events[-1]["ts"] - down_events[0]["ts"]

        wpm = 0
        if total_duration_ms > 0:
            chars_per_min = len(down_events) / total_duration_ms * 60000
            wpm = chars_per_min / 5

        # 驻留时间全局统计
        all_dwell = []
        for cat_dwell in dwell_times.values():
            all_dwell.extend(cat_dwell)

        avg_dwell = np.mean(all_dwell) if all_dwell else 0
        std_dwell = np.std(all_dwell) if all_dwell else 0

        # DD飞行时间统计
        dd_values = flight_times.get("DD", [])
        avg_flight_dd = np.mean(dd_values) if dd_values else 0
        std_flight_dd = np.std(dd_values) if dd_values else 0

        # 停顿比例（飞行时间>300ms的比例）
        pause_count = sum(1 for ft in dd_values if ft > 300)
        pause_ratio = pause_count / len(dd_values) if dd_values else 0

        # 连续打字爆发段平均长度
        burst_lengths = []
        current_burst = 0
        for ft in dd_values:
            if ft < 300:
                current_burst += 1
            else:
                if current_burst > 0:
                    burst_lengths.append(current_burst)
                current_burst = 0
        if current_burst > 0:
            burst_lengths.append(current_burst)
        avg_burst_length = np.mean(burst_lengths) if burst_lengths else 0

        return {
            "wpm": wpm,
            "avg_dwell": avg_dwell,
            "std_dwell": std_dwell,
            "avg_flight_dd": avg_flight_dd,
            "std_flight_dd": std_flight_dd,
            "pause_ratio": pause_ratio,
            "avg_burst_length": avg_burst_length,
        }

    def _compute_auxiliary_features(self, events, dwell_times, flight_times) -> Dict:
        """
        计算辅助特征

        Returns:
            辅助特征字典
        """
        down_events = [ev for ev in events if ev["type"] == 1]

        # Shift键使用习惯
        lshift_count = sum(1 for ev in down_events if ev["cat"] == CAT_LSHIFT)
        rshift_count = sum(1 for ev in down_events if ev["cat"] == CAT_RSHIFT)
        total_shift = lshift_count + rshift_count
        lshift_ratio = lshift_count / total_shift if total_shift > 0 else 0.5

        # Backspace频率
        backspace_count = sum(1 for ev in down_events if ev["cat"] == CAT_BACKSPACE)
        backspace_ratio = backspace_count / len(down_events) if down_events else 0

        # Rollover比例（DD飞行时间为负值=前一键未释放时按下下一键）
        dd_values = flight_times.get("DD", [])
        # 注意：由于事件流可能不严格按时间排序，负值飞行时间表示rollover
        # 在当前实现中，DD值已过滤为正值，所以rollover需要从原始事件检测
        rollover_ratio = 0  # 简化实现

        # 字母键平均驻留时间
        alpha_dwell = dwell_times.get(CAT_ALPHA, [])
        avg_dwell_alpha = np.mean(alpha_dwell) if alpha_dwell else 0

        # 空格键平均驻留时间
        space_dwell = dwell_times.get(CAT_SPACE, [])
        avg_dwell_space = np.mean(space_dwell) if space_dwell else 0

        # 打字速度方差（将事件分成10段，计算每段WPM的方差）
        if len(down_events) >= 20:
            segment_size = len(down_events) // 10
            segment_wpm = []
            for i in range(0, len(down_events) - segment_size, segment_size):
                seg = down_events[i:i + segment_size + 1]
                duration = seg[-1]["ts"] - seg[0]["ts"]
                if duration > 0:
                    seg_wpm = len(seg) / duration * 60000 / 5
                    segment_wpm.append(seg_wpm)
            speed_variance = np.var(segment_wpm) if segment_wpm else 0
        else:
            speed_variance = 0

        return {
            "lshift_ratio": lshift_ratio,
            "backspace_ratio": backspace_ratio,
            "rollover_ratio": rollover_ratio,
            "avg_dwell_alpha": avg_dwell_alpha,
            "avg_dwell_space": avg_dwell_space,
            "speed_variance": speed_variance,
        }

    def _stats_by_category(self, dwell_times: Dict) -> np.ndarray:
        """按键位分类计算驻留时间统计量"""
        stats = []
        for cat in [CAT_ALPHA, CAT_DIGIT, CAT_SPACE, CAT_ENTER, CAT_BACKSPACE]:
            values = dwell_times.get(cat, [])
            stats.extend(self._compute_stats(values))
        return np.array(stats)

    @staticmethod
    def _compute_stats(values: List[float]) -> np.ndarray:
        """计算统计量：均值、标准差、中位数、最小值、最大值"""
        if not values:
            return np.zeros(5)
        return np.array([
            np.mean(values),
            np.std(values),
            np.median(values),
            np.min(values),
            np.max(values),
        ])
