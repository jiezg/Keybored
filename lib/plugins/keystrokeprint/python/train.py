"""
键纹模型训练入口

用法: python train.py --input <样本目录> --output <模型输出目录> --user-id <用户ID>

流程:
1. 读取样本文件（JSON格式）
2. 特征提取 → 特征矩阵
3. 数据清洗（异常值剔除）
4. Z-Score归一化
5. 模型训练（分阶段：Manhattan基线 → SVM/随机森林）
6. 导出ONNX模型 + 归一化参数
7. 写入训练结果
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.extractor import FeatureExtractor


def load_samples(sample_dir: str) -> list:
    """加载样本目录下所有JSON文件"""
    samples = []
    for filename in os.listdir(sample_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(sample_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "events" in data and len(data["events"]) > 0:
                samples.append(data)
    return samples


def extract_features_from_samples(samples: list) -> np.ndarray:
    """从样本列表提取特征矩阵"""
    extractor = FeatureExtractor()
    feature_vectors = []

    for sample in samples:
        events = sample["events"]
        if len(events) < 10:  # 至少10个事件才有意义
            continue

        # 滑动窗口切分（每50个事件一个窗口，步长25）
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
        raise ValueError("无法提取有效特征，请确保采集了足够的击键数据")

    return np.array(feature_vectors)


def train_manhattan_baseline(features: np.ndarray, scaler: StandardScaler) -> dict:
    """
    训练Manhattan距离基线模型
    仅存储归一化后的均值向量，推理时计算Manhattan距离
    """
    scaled = scaler.transform(features)
    mean_vector = np.mean(scaled, axis=0)
    std_vector = np.std(scaled, axis=0)

    # 计算训练集自身的距离分布，用于确定阈值
    distances = np.sum(np.abs(scaled - mean_vector), axis=1)
    threshold = np.percentile(distances, 95)  # 95百分位作为阈值

    return {
        "model_type": "manhattan",
        "mean_vector": mean_vector.tolist(),
        "std_vector": std_vector.tolist(),
        "threshold": float(threshold),
        "distance_stats": {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
        }
    }


def train_svm_model(features: np.ndarray, scaler: StandardScaler) -> tuple:
    """
    训练One-Class SVM模型（单类分类，适合只有正样本的场景）
    """
    scaled = scaler.transform(features)

    # One-Class SVM
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    model.fit(scaled)

    return model


def train_random_forest(features: np.ndarray, scaler: StandardScaler) -> tuple:
    """
    训练随机森林模型（需要负样本，用异常值合成）
    """
    scaled = scaler.transform(features)
    n_samples = len(scaled)

    # 合成负样本（在特征空间中随机偏移）
    rng = np.random.RandomState(42)
    neg_samples = scaled + rng.randn(n_samples, scaled.shape[1]) * 3

    X = np.vstack([scaled, neg_samples])
    y = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)

    return rf


def export_model_onnx(model, input_dim: int, output_path: str):
    """将sklearn模型导出为ONNX格式"""
    initial_type = [("float_input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description="键纹模型训练")
    parser.add_argument("--input", required=True, help="样本目录路径")
    parser.add_argument("--output", required=True, help="模型输出目录路径")
    parser.add_argument("--user-id", required=True, help="用户ID")
    parser.add_argument("--model-type", default="auto",
                        choices=["auto", "manhattan", "svm", "rf"],
                        help="模型类型（auto=根据样本数自动选择）")
    args = parser.parse_args()

    print(f"[键纹训练] 用户: {args.user_id}")
    print(f"[键纹训练] 样本目录: {args.input}")

    # 1. 加载样本
    samples = load_samples(args.input)
    print(f"[键纹训练] 加载样本文件: {len(samples)}")

    if len(samples) == 0:
        print("[键纹训练] 错误: 未找到有效样本文件")
        sys.exit(1)

    # 2. 特征提取
    features = extract_features_from_samples(samples)
    print(f"[键纹训练] 提取特征向量: {features.shape}")

    # 3. 归一化
    scaler = StandardScaler()
    scaler.fit(features)

    # 4. 确定模型类型
    n_samples = len(features)
    if args.model_type == "auto":
        if n_samples < 20:
            model_type = "manhattan"
        elif n_samples < 100:
            model_type = "svm"
        else:
            model_type = "rf"
    else:
        model_type = args.model_type

    print(f"[键纹训练] 样本数: {n_samples}, 选择模型: {model_type}")

    # 5. 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 6. 训练模型
    meta = {
        "user_id": args.user_id,
        "model_type": model_type,
        "n_samples": n_samples,
        "n_features": features.shape[1],
        "training_status": "success",
    }

    if model_type == "manhattan":
        result = train_manhattan_baseline(features, scaler)
        # Manhattan模型直接存储为JSON
        with open(os.path.join(args.output, "manhattan_model.json"), "w") as f:
            json.dump(result, f, indent=2)
        meta["distance_stats"] = result["distance_stats"]

    elif model_type == "svm":
        model = train_svm_model(features, scaler)
        # 导出ONNX
        onnx_path = os.path.join(args.output, "model.onnx")
        export_model_onnx(model, features.shape[1], onnx_path)
        meta["onnx_model"] = "model.onnx"

    elif model_type == "rf":
        model = train_random_forest(features, scaler)
        # 导出ONNX
        onnx_path = os.path.join(args.output, "model.onnx")
        export_model_onnx(model, features.shape[1], onnx_path)
        meta["onnx_model"] = "model.onnx"

    # 7. 保存归一化参数
    scaler_path = os.path.join(args.output, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # 8. 保存元信息
    meta_path = os.path.join(args.output, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[键纹训练] 训练完成! 模型保存至: {args.output}")
    print(f"[键纹训练] 模型类型: {model_type}, 样本数: {n_samples}")


if __name__ == "__main__":
    main()
