# src/utils.py

import random
import numpy as np
import torch
import time
from thop import profile
import matplotlib.pyplot as plt
from matplotlib import rcParams


def set_seed(seed=42):
    """设置随机种子以确保结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_snr(y_true, y_pred):
    """计算信噪比 (SNR)。"""
    signal, noise = y_true, y_true - y_pred
    signal_power, noise_power = np.var(signal), np.var(noise)
    return 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')


def calculate_cc(y_true, y_pred):
    """计算相关系数 (CC)。"""
    if len(y_true) != len(y_pred):
        raise ValueError("真实值和预测值的长度必须相同。")
    return np.corrcoef(y_true, y_pred)[0, 1]


def setup_plotting_style(font_size=35, font_family='Times New Roman'):
    """设置全局Matplotlib绘图风格。"""
    rcParams['font.sans-serif'] = [font_family]
    rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': font_size})


def plot_predictions(actuals, predictions, title, save_path, scale_factor=1e7, font_size=35):
    """
    绘制并保存真实值与预测值的对比图。
    该函数复现了您多个脚本中的绘图逻辑。
    """
    plt.figure(figsize=(12, 8))
    plt.plot(actuals / scale_factor, label='True Values')
    plt.plot(predictions / scale_factor, label='Predicted Values')
    plt.title(title, fontsize=70)
    plt.xlabel('Timestamps', fontsize=50)
    plt.ylabel("Bandwidth [bit/s]")
    plt.legend()
    ax = plt.gca()
    ax.text(-0.05, 1.02, f'{scale_factor:.0e}'.replace('+0', ''), transform=ax.transAxes, ha='left', va='bottom',
            fontsize=font_size)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.show()
    print(f"绘图已保存至: {save_path}")


def plot_loss_curve(loss_history, title, save_path):
    """绘制并保存损失函数曲线。"""
    plt.figure(figsize=(12, 4))
    plt.plot(loss_history, label='训练集损失')
    plt.title(title, fontsize=14)
    plt.xlabel('迭代次数（Epochs）', fontsize=14)
    plt.ylabel('损失', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"损失曲线图已保存至: {save_path}")


def analyze_efficiency(model, device, lookback, feature_size=1, batch_size=64, *model_fwd_args):
    """
    对给定的PyTorch模型进行全面的计算效率分析 (通用版)。
    该函数直接来自您提供的 utils.py。
    """
    print("\n" + "=" * 50)
    # 尝试获取model.model_type，如果不存在则使用类名
    model_name = getattr(model, 'model_type', model.__class__.__name__)
    print(" " * 15 + f"计算效率分析: {model_name}")
    print("=" * 50)

    model.eval()
    model.to(device)

    # 1. 模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[1] 模型参数量: {num_params / 1e6:.3f} M")

    # 2. 理论计算量 (FLOPs)
    dummy_input_flops = torch.randn(1, lookback, feature_size).to(device)
    profile_inputs = (dummy_input_flops, *model_fwd_args)
    flops, params = profile(model, inputs=profile_inputs, verbose=False)
    print(f"[2] 理论计算量 (FLOPs): {flops / 1e9:.3f} GFLOPs")

    # 3. 推理延迟 & 吞吐量
    with torch.no_grad():
        # 3.1 单样本推理延迟
        dummy_input_single = torch.randn(1, lookback, feature_size).to(device)
        for _ in range(20):
            _ = model(dummy_input_single, *model_fwd_args)

        iterations = 200
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy_input_single, *model_fwd_args)
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_latency_ms = (end_time - start_time) / iterations * 1000
        print(f"[3.1] 单样本推理延迟: {avg_latency_ms:.3f} ms/sample")

        # 3.2 批处理推理延迟 & 吞吐量
        dummy_input_batch = torch.randn(batch_size, lookback, feature_size).to(device)
        for _ in range(20):
            _ = model(dummy_input_batch, *model_fwd_args)

        iterations = 100
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time_batch = time.perf_counter()
        for _ in range(iterations):
            _ = model(dummy_input_batch, *model_fwd_args)
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time_batch = time.perf_counter()

        avg_batch_latency_s = (end_time_batch - start_time_batch) / iterations
        throughput = batch_size / avg_batch_latency_s
        print(f"[3.2] 批处理延迟 ({batch_size}个样本/批): {avg_batch_latency_s * 1000:.3f} ms/batch")
        print(f"[3.3] 模型吞吐量: {throughput:.2f} samples/sec")

    print("=" * 50 + "\n")