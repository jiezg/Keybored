"""
键纹模型推理入口

用法: python infer.py --model <模型目录> --input <实时样本JSON> --output <结果输出JSON>

流程:
1. 加载模型（ONNX或Manhattan JSON）
2. 读取实时样本
3. 特征提取 + 归一化
4. 模型推理 → 相似度分数（0-1）
5. 写入结果文件
"""

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

    # One-Class SVM: 1=正常, -1=异常
    # RandomForest: 1=匹配, 0=不匹配
    raw_scores = predictions[0]

    # 计算匹配比例作为相似度
    match_count = np.sum(raw_scores == 1) if len(raw_scores.shape) == 1 else np.sum(raw_scores > 0.5)
    similarity = match_count / len(raw_scores)
    return float(similarity)


def main():
    parser = argparse.ArgumentParser(description="键纹模型推理")
    parser.add_argument("--model", required=True, help="模型目录路径")
    parser.add_argument("--input", required=True, help="实时样本JSON文件路径")
    parser.add_argument("--output", required=True, help="结果输出JSON文件路径")
    args = parser.parse_args()

    print(f"[键纹推理] 模型目录: {args.model}")
    print(f"[键纹推理] 输入样本: {args.input}")

    # 1. 加载模型元信息
    meta_path = os.path.join(args.model, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_type = meta["model_type"]
    print(f"[键纹推理] 模型类型: {model_type}, 用户: {meta['user_id']}")

    # 2. 加载归一化参数
    scaler = load_scaler(args.model)

    # 3. 读取实时样本
    with open(args.input, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    events = sample_data.get("events", [])
    if len(events) < 10:
        result = {
            "user_id": meta["user_id"],
            "similarity": 0,
            "confidence": "low",
            "message": "样本不足，至少需要10个事件",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    # 4. 特征提取（滑动窗口）
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
        result = {
            "user_id": meta["user_id"],
            "similarity": 0,
            "confidence": "low",
            "message": "无法提取有效特征",
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    features = np.array(feature_vectors)
    print(f"[键纹推理] 特征向量: {features.shape}")

    # 5. 模型推理
    if model_type == "manhattan":
        model = load_manhattan_model(args.model)
        similarity = compute_manhattan_similarity(features, model, scaler)
    else:
        session = load_onnx_model(args.model)
        similarity = compute_onnx_similarity(features, session, scaler)

    # 6. 计算置信度
    n_windows = len(feature_vectors)
    if n_windows >= 5:
        confidence = "high"
    elif n_windows >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # 7. 写入结果
    result = {
        "user_id": meta["user_id"],
        "similarity": round(similarity, 4),
        "confidence": confidence,
        "n_windows": n_windows,
        "model_type": model_type,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[键纹推理] 相似度: {similarity:.4f}, 置信度: {confidence}")


if __name__ == "__main__":
    main()
