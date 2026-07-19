"""
键纹模型推理入口

用法:
  # 多模型对比（推荐）
  python infer.py --model-dirs <dir1,dir2,...> --input <实时样本JSON> --output <结果JSON> --threshold 0.6

  # 单模型（向后兼容）
  python infer.py --model <dir> --input <实时样本JSON> --output <结果JSON>

流程:
1. 读取实时样本，提取特征（仅一次）
2. 对每个模型计算相似度
3. 取最佳匹配，超过阈值则 matched_user_id 有值
4. 写入结果文件
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys
import pickle
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.extractor import FeatureExtractor


def load_manhattan_model(model_dir: str) -> dict:
    """加载Manhattan距离基线模型"""
    model_path = os.path.join(model_dir, "manhattan_model.json")
    with open(model_path, "r") as f:
        return json.load(f)


def load_onnx_model(model_dir: str):
    """加载ONNX模型"""
    import onnxruntime as ort
    model_path = os.path.join(model_dir, "model.onnx")
    session = ort.InferenceSession(model_path)
    return session


def load_scaler(model_dir: str):
    """加载归一化参数"""
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def compute_manhattan_similarity(features: np.ndarray, model: dict, scaler) -> float:
    """
    计算Manhattan距离相似度

    Returns:
        相似度分数（0-1，1=完全匹配）
    """
    mean_vector = np.array(model["mean_vector"])
    threshold = model["threshold"]

    scaled = scaler.transform(features)
    distances = np.sum(np.abs(scaled - mean_vector), axis=1)
    avg_distance = np.mean(distances)

    # 将距离映射为相似度（0-1）
    # 距离越小，相似度越高
    similarity = max(0, 1 - avg_distance / (threshold * 2))
    return float(similarity)


def compute_onnx_similarity(features: np.ndarray, session, scaler) -> float:
    """
    使用ONNX模型计算相似度

    Returns:
        相似度分数（0-1）
    """
    scaled = scaler.transform(features).astype(np.float32)

    # ONNX推理
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: scaled})

    # RandomForest ONNX 启用 zipmap 后第二输出为概率
    # 格式可能是 list of dicts（zipmap）、list of lists 或 numpy array
    # 取类别1（匹配）的概率均值，得到连续 similarity（0.0-1.0）
    # 这样 A→B 切换时 A 模型的 similarity 会逐渐降低，而非硬分类的突变
    if len(predictions) > 1:
        probs = predictions[1]
        # zipmap 输出是 list of dicts：[{'0': 0.1, '1': 0.9}, ...]
        if isinstance(probs, list) and len(probs) > 0 and isinstance(probs[0], dict):
            # 提取类别1的概率（dict 的 key 可能是字符串 '1' 或整数 1）
            prob_class1 = [p.get('1', p.get(1, 0.0)) for p in probs]
            similarity = float(np.mean(prob_class1))
        else:
            # 转为 numpy array 统一处理（list of lists 或 numpy array）
            probs_arr = np.array(probs)
            if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                similarity = float(np.mean(probs_arr[:, 1]))
            else:
                # 格式异常，回退到硬分类比例
                similarity = float(np.mean(predictions[0] == 1))
    else:
        # 旧模型无概率输出，回退到硬分类比例
        similarity = float(np.mean(predictions[0] == 1))
    return max(0.0, min(1.0, similarity))


def extract_features_from_sample(sample_path: str):
    """
    从样本文件提取特征向量

    Returns:
        (features, n_windows) 或 (None, 0)
    """
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    events = sample_data.get("events", [])
    if len(events) < 10:
        return None, 0

    extractor = FeatureExtractor()
    feature_vectors = []
    window_size = 50
    step = 25

    for start in range(0, len(events) - window_size + 1, step):
        window = events[start:start + window_size]
        try:
            fv = extractor.extract_feature_vector(window)
            if not np.any(np.isnan(fv)) and not np.any(np.isinf(fv)):
                feature_vectors.append(fv)
        except Exception:
            continue

    if not feature_vectors:
        return None, 0

    return np.array(feature_vectors), len(feature_vectors)


def infer_single_model(features: np.ndarray, model_dir: str) -> dict:
    """
    对单个模型进行推理

    Returns:
        {user_id, similarity, confidence, model_type}
    """
    meta_path = os.path.join(model_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_type = meta["model_type"]
    scaler = load_scaler(model_dir)

    if model_type == "manhattan":
        model = load_manhattan_model(model_dir)
        similarity = compute_manhattan_similarity(features, model, scaler)
    else:
        session = load_onnx_model(model_dir)
        similarity = compute_onnx_similarity(features, session, scaler)

    n_windows = len(features)
    if n_windows >= 5:
        confidence = "high"
    elif n_windows >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "user_id": meta["user_id"],
        "similarity": round(similarity, 4),
        "confidence": confidence,
        "model_type": model_type,
    }


def main():
    parser = argparse.ArgumentParser(description="键纹模型推理")
    parser.add_argument("--model", help="单个模型目录路径（向后兼容）")
    parser.add_argument("--model-dirs", help="多个模型目录路径（逗号分隔）")
    parser.add_argument("--input", required=True, help="实时样本JSON文件路径")
    parser.add_argument("--output", required=True, help="结果输出JSON文件路径")
    parser.add_argument("--threshold", type=float, default=0.6, help="匹配阈值（0-1）")
    args = parser.parse_args()

    # 解析模型目录列表
    if args.model_dirs:
        model_dirs = [d.strip() for d in args.model_dirs.split(",") if d.strip()]
    elif args.model:
        model_dirs = [args.model]
    else:
        print("[键纹推理] 错误: 必须指定 --model 或 --model-dirs")
        sys.exit(1)

    print(f"[键纹推理] 模型数: {len(model_dirs)}, 阈值: {args.threshold}")

    # 1. 从样本提取特征（仅一次）
    features, n_windows = extract_features_from_sample(args.input)
    if features is None:
        result = {
            "matched_user_id": None,
            "best_similarity": 0,
            "best_confidence": "low",
            "is_match": False,
            "n_windows": n_windows,
            "message": "样本不足或无法提取有效特征",
            "all_results": [],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    print(f"[键纹推理] 特征向量: {features.shape}, 窗口数: {n_windows}")

    # 2. 对每个模型推理
    all_results = []
    for model_dir in model_dirs:
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            print(f"[键纹推理] 跳过（无meta.json）: {model_dir}")
            continue
        try:
            r = infer_single_model(features, model_dir)
            all_results.append(r)
            print(f"[键纹推理] {r['user_id']}: similarity={r['similarity']:.4f}")
        except Exception as e:
            print(f"[键纹推理] 模型推理失败 {model_dir}: {e}")
            continue

    # 3. 取最佳匹配
    if not all_results:
        result = {
            "matched_user_id": None,
            "best_similarity": 0,
            "best_confidence": "low",
            "is_match": False,
            "n_windows": n_windows,
            "message": "无可用模型",
            "all_results": [],
        }
    else:
        # 按相似度降序排序
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        best = all_results[0]
        is_match = best["similarity"] >= args.threshold

        result = {
            "matched_user_id": best["user_id"] if is_match else None,
            "best_similarity": best["similarity"],
            "best_confidence": best["confidence"],
            "is_match": is_match,
            "n_windows": n_windows,
            "all_results": all_results,
        }

    # 4. 写入结果
    # dirname 为空（output 是纯文件名）时跳过 makedirs，避免 FileNotFoundError
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    match_name = result["matched_user_id"] or "未识别"
    print(f"[键纹推理] 最佳匹配: {match_name}, 相似度: {result['best_similarity']:.4f}, 匹配: {is_match}")


if __name__ == "__main__":
    main()
    #强制退出，绕过 onnxruntime/sklearn atexit 清理挂起（同 train.py）
    import os as _os, sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
