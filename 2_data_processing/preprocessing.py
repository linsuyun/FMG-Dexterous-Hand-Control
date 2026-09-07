"""
数据预处理模块
处理原始传感器数据：滤波、归一化、分段
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
import pandas as pd
from typing import Tuple, List


class FMGDataPreprocessor:
    """FMG 传感器数据预处理类"""
    
    def __init__(self, 
                 sampling_rate: int = 1000,
                 num_channels: int = 24,
                 window_size: int = 200):
        """
        初始化预处理器
        
        Args:
            sampling_rate: 采样率 (Hz)
            num_channels: 传感器通道数
            window_size: 分割窗口大小 (采样点)
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.window_size = window_size
        self.stride = window_size // 2  # 50% 重叠
        
    def butterworth_filter(self, 
                          signal_data: np.ndarray,
                          cutoff_freq: float = 20,
                          order: int = 4) -> np.ndarray:
        """
        应用 Butterworth 低通滤波器
        
        Args:
            signal_data: 输入信号 (shape: [time_steps, num_channels])
            cutoff_freq: 截止频率 (Hz)
            order: 滤波器阶数
            
        Returns:
            滤波后的信号
        """
        nyquist_freq = self.sampling_rate / 2
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # 防止 cutoff >= Nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        # 对每个通道应用滤波
        filtered_signal = np.zeros_like(signal_data)
        for ch in range(signal_data.shape[1]):
            filtered_signal[:, ch] = signal.filtfilt(b, a, signal_data[:, ch])
        
        return filtered_signal
    
    def notch_filter(self, 
                    signal_data: np.ndarray,
                    notch_freq: float = 50) -> np.ndarray:
        """
        应用 Notch 滤波器（去除工频噪声）
        
        Args:
            signal_data: 输入信号
            notch_freq: Notch 频率 (通常为 50Hz 或 60Hz)
            
        Returns:
            滤波后的信号
        """
        nyquist_freq = self.sampling_rate / 2
        normalized_notch = notch_freq / nyquist_freq
        
        if normalized_notch >= 1.0 or normalized_notch <= 0:
            return signal_data
        
        # 设置 Q 值（品质因数）
        Q = 30
        b, a = signal.iirnotch(normalized_notch, Q)
        
        filtered_signal = np.zeros_like(signal_data)
        for ch in range(signal_data.shape[1]):
            filtered_signal[:, ch] = signal.filtfilt(b, a, signal_data[:, ch])
        
        return filtered_signal
    
    def normalize(self, 
                 signal_data: np.ndarray,
                 method: str = 'z_score') -> np.ndarray:
        """
        信号归一化
        
        Args:
            signal_data: 输入信号
            method: 归一化方法 ('z_score', 'min_max', 'robust')
            
        Returns:
            归一化后的信号
        """
        if method == 'z_score':
            # Z-score 归一化
            mean = np.mean(signal_data, axis=0, keepdims=True)
            std = np.std(signal_data, axis=0, keepdims=True)
            std[std == 0] = 1  # 防止除以零
            return (signal_data - mean) / std
        
        elif method == 'min_max':
            # Min-Max 归一化 到 [0, 1]
            min_val = np.min(signal_data, axis=0, keepdims=True)
            max_val = np.max(signal_data, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            return (signal_data - min_val) / range_val
        
        elif method == 'robust':
            # 鲁棒归一化（使用中位数和四分位数）
            q1 = np.percentile(signal_data, 25, axis=0, keepdims=True)
            q3 = np.percentile(signal_data, 75, axis=0, keepdims=True)
            median = np.median(signal_data, axis=0, keepdims=True)
            iqr = q3 - q1
            iqr[iqr == 0] = 1
            return (signal_data - median) / iqr
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def segment_signal(self, 
                      signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将信号分段
        
        Args:
            signal_data: 输入信号 (shape: [time_steps, num_channels])
            
        Returns:
            segments: 分段后的信号 (shape: [num_segments, window_size, num_channels])
            segment_indices: 每个分段的起始索引
        """
        segments = []
        segment_indices = []
        
        for start_idx in range(0, len(signal_data) - self.window_size + 1, self.stride):
            segment = signal_data[start_idx:start_idx + self.window_size]
            segments.append(segment)
            segment_indices.append(start_idx)
        
        return np.array(segments), np.array(segment_indices)
    
    def extract_features(self, segment: np.ndarray) -> np.ndarray:
        """
        从单个分段提取特征
        
        Args:
            segment: 单个分段 (shape: [window_size, num_channels])
            
        Returns:
            特征向量
        """
        features = []
        
        for ch in range(segment.shape[1]):
            channel_data = segment[:, ch]
            
            # 时域特征
            features.extend([
                np.mean(channel_data),           # 均值
                np.std(channel_data),            # 标准差
                np.max(channel_data),            # 最大值
                np.min(channel_data),            # 最小值
                np.max(np.abs(channel_data)),    # 绝对值最大
                np.sqrt(np.mean(channel_data**2)), # 均方根 (RMS)
                np.sum(np.abs(np.diff(channel_data))), # 绝对差分和
            ])
        
        return np.array(features)
    
    def preprocess_raw_data(self, 
                           raw_signal: np.ndarray,
                           label: int = None) -> Tuple[np.ndarray, int]:
        """
        完整的预处理流程
        
        Args:
            raw_signal: 原始信号 (shape: [time_steps, num_channels])
            label: 手势标签（可选）
            
        Returns:
            preprocessed_signal: 预处理后的信号
            label: 手势标签
        """
        # 1. 去除工频噪声
        signal_data = self.notch_filter(raw_signal, notch_freq=50)
        
        # 2. 低通滤波
        signal_data = self.butterworth_filter(signal_data, cutoff_freq=20, order=4)
        
        # 3. 归一化
        signal_data = self.normalize(signal_data, method='z_score')
        
        return signal_data, label
    
    def process_batch(self, 
                     raw_signals: List[np.ndarray],
                     labels: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量处理多个信号
        
        Args:
            raw_signals: 原始信号列表
            labels: 标签列表
            
        Returns:
            processed_segments: 处理后的分段 (shape: [total_segments, window_size, num_channels])
            segment_labels: 对应的标签
        """
        all_segments = []
        all_labels = []
        
        for i, raw_signal in enumerate(raw_signals):
            # 预处理
            processed_signal, _ = self.preprocess_raw_data(raw_signal)
            
            # 分段
            segments, _ = self.segment_signal(processed_signal)
            all_segments.extend(segments)
            
            # 标签
            if labels is not None:
                all_labels.extend([labels[i]] * len(segments))
        
        return np.array(all_segments), np.array(all_labels)


def load_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    加载示例数据
    
    Returns:
        signals: 信号数据 (shape: [num_samples, time_steps, num_channels])
        labels: 标签 (shape: [num_samples])
    """
    # 模拟 5 种手势，每种 10 个样本
    num_gestures = 5
    samples_per_gesture = 10
    time_steps = 2000
    num_channels = 24
    
    signals = []
    labels = []
    
    for gesture_id in range(num_gestures):
        for _ in range(samples_per_gesture):
            # 生成模拟信号（随机信号 + 频率特征）
            t = np.linspace(0, 2, time_steps)
            
            # 为不同的手势设置不同的频率特征
            freq = 5 + gesture_id * 2
            signal_data = np.random.randn(time_steps, num_channels) * 0.5
            signal_data += np.sin(2 * np.pi * freq * t)[:, np.newaxis]
            
            signals.append(signal_data)
            labels.append(gesture_id)
    
    return np.array(signals), np.array(labels)


if __name__ == "__main__":
    # 示例：加载和预处理数据
    print("加载示例数据...")
    signals, labels = load_sample_data()
    print(f"信号形状: {signals.shape}")
    print(f"标签: {labels}")
    
    # 初始化预处理器
    preprocessor = FMGDataPreprocessor(
        sampling_rate=1000,
        num_channels=24,
        window_size=200
    )
    
    # 处理第一个信号
    print("\n处理第一个信号...")
    processed_signal, label = preprocessor.preprocess_raw_data(signals[0], labels[0])
    print(f"处理后的信号形状: {processed_signal.shape}")
    
    # 分段
    segments, indices = preprocessor.segment_signal(processed_signal)
    print(f"分段数: {len(segments)}")
    print(f"每个分段形状: {segments[0].shape}")
    
    # 提取特征
    features = preprocessor.extract_features(segments[0])
    print(f"提取的特征数: {len(features)}")
    print(f"特征示例: {features[:10]}")
    
    # 批量处理
    print("\n批量处理所有数据...")
    processed_segments, segment_labels = preprocessor.process_batch(
        raw_signals=list(signals[:5]),
        labels=list(labels[:5])
    )
    print(f"处理后的分段形状: {processed_segments.shape}")
    print(f"分段标签: {segment_labels}")
