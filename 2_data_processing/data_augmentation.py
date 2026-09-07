"""
数据增强模块
通过多种增强技术提高模型泛化性
"""

import numpy as np
from typing import Tuple, List
from scipy.interpolate import interp1d


class FMGDataAugmentation:
    """FMG 数据增强类"""
    
    def __init__(self, window_size: int = 200, num_channels: int = 24):
        """
        初始化数据增强
        
        Args:
            window_size: 时间窗口大小
            num_channels: 传感器通道数
        """
        self.window_size = window_size
        self.num_channels = num_channels
    
    def time_warping(self, 
                    segment: np.ndarray,
                    warp_amount: float = 0.1) -> np.ndarray:
        """
        时间扭曲增强 (Time Warping)
        通过改变时间轴的速度来增强数据
        
        Args:
            segment: 输入分段 (shape: [window_size, num_channels])
            warp_amount: 扭曲程度 (0-1)
            
        Returns:
            扭曲后的分段
        """
        time_steps = np.arange(self.window_size)
        
        # 随机生成扭曲曲线
        warp_curve = np.random.randn(self.window_size) * warp_amount
        warp_curve = np.cumsum(warp_curve) + time_steps
        warp_curve = np.clip(warp_curve, 0, self.window_size - 1)
        
        warped_segment = np.zeros_like(segment)
        
        for ch in range(self.num_channels):
            f = interp1d(time_steps, segment[:, ch], kind='cubic', fill_value='extrapolate')
            warped_segment[:, ch] = f(warp_curve)
        
        return warped_segment
    
    def amplitude_scaling(self, 
                         segment: np.ndarray,
                         scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        幅度缩放增强 (Amplitude Scaling)
        随机缩放信号幅度
        
        Args:
            segment: 输入分段
            scale_range: 缩放范围 (min_scale, max_scale)
            
        Returns:
            缩放后的分段
        """
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return segment * scale_factor
    
    def jittering(self, 
                 segment: np.ndarray,
                 jitter_std: float = 0.01) -> np.ndarray:
        """
        抖动增强 (Jittering)
        添加高斯噪声
        
        Args:
            segment: 输入分段
            jitter_std: 噪声标准差
            
        Returns:
            添加噪声后的分段
        """
        noise = np.random.randn(segment.shape[0], segment.shape[1]) * jitter_std
        return segment + noise
    
    def permutation(self, 
                   segment: np.ndarray,
                   num_permutations: int = 4) -> np.ndarray:
        """
        排列增强 (Permutation)
        将分段分成若干部分并随机排列
        
        Args:
            segment: 输入分段
            num_permutations: 排列数
            
        Returns:
            排列后的分段
        """
        segment_copy = segment.copy()
        partition_size = self.window_size // num_permutations
        
        for ch in range(self.num_channels):
            indices = np.arange(self.window_size)
            
            # 随机打乱分区
            for i in range(num_permutations - 1):
                start = i * partition_size
                end = (i + 1) * partition_size
                part_indices = np.random.permutation(end - start)
                indices[start:end] = start + part_indices
            
            segment_copy[:, ch] = segment[indices, ch]
        
        return segment_copy
    
    def mixup(self, 
             segment1: np.ndarray,
             segment2: np.ndarray,
             alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        混合增强 (Mixup)
        将两个分段按比例混合
        
        Args:
            segment1: 第一个分段
            segment2: 第二个分段
            alpha: 混合权重 (0-1)
            
        Returns:
            混合后的两个分段
        """
        lambda_val = np.random.beta(alpha, alpha)
        
        mixed1 = lambda_val * segment1 + (1 - lambda_val) * segment2
        mixed2 = (1 - lambda_val) * segment1 + lambda_val * segment2
        
        return mixed1, mixed2
    
    def random_rotation(self, 
                       segment: np.ndarray) -> np.ndarray:
        """
        随机旋转增强 (Random Rotation)
        对每个通道应用随机旋转
        
        Args:
            segment: 输入分段
            
        Returns:
            旋转后的分段
        """
        rotated_segment = np.zeros_like(segment)
        
        for ch in range(self.num_channels):
            # 生成随机旋转角度
            theta = np.random.uniform(0, 2 * np.pi)
            
            # 构造旋转矩阵（在复平面上）
            rotation = np.exp(1j * theta)
            
            # 转换为复数并旋转
            complex_signal = segment[:, ch] * rotation
            rotated_segment[:, ch] = np.real(complex_signal)
        
        return rotated_segment
    
    def window_slicing(self, 
                      segment: np.ndarray,
                      slice_ratio: float = 0.9) -> np.ndarray:
        """
        窗口切片增强 (Window Slicing)
        随机删除一部分时间序列
        
        Args:
            segment: 输入分段
            slice_ratio: 保留的比例
            
        Returns:
            切片后的分段
        """
        slice_len = int(self.window_size * slice_ratio)
        start = np.random.randint(0, self.window_size - slice_len + 1)
        
        sliced_segment = segment.copy()
        sliced_segment[start:start + slice_len] = 0
        
        return sliced_segment
    
    def dynamic_time_warping_noise(self, 
                                   segment: np.ndarray,
                                   warp_std: float = 0.05) -> np.ndarray:
        """
        动态时间扭曲噪声增强 (DTW Noise)
        添加时间轴扭曲噪声
        
        Args:
            segment: 输入分段
            warp_std: 扭曲标准差
            
        Returns:
            添加 DTW 噪声后的分段
        """
        # 生成光滑的扭曲曲线
        random_points = np.random.randn(5) * warp_std
        x_original = np.linspace(0, 1, 5)
        x_smooth = np.linspace(0, 1, self.window_size)
        
        warp_curve = np.interp(x_smooth, x_original, random_points)
        time_indices = np.arange(self.window_size)
        warped_indices = time_indices + warp_curve * self.window_size
        warped_indices = np.clip(warped_indices, 0, self.window_size - 1)
        
        warped_segment = np.zeros_like(segment)
        for ch in range(self.num_channels):
            f = interp1d(time_indices, segment[:, ch], kind='linear', fill_value='extrapolate')
            warped_segment[:, ch] = f(warped_indices)
        
        return warped_segment
    
    def augment_batch(self, 
                     segments: np.ndarray,
                     labels: np.ndarray,
                     augmentation_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量增强数据
        
        Args:
            segments: 原始分段 (shape: [num_segments, window_size, num_channels])
            labels: 标签 (shape: [num_segments])
            augmentation_factor: 每个样本增强的倍数
            
        Returns:
            augmented_segments: 增强后的分段
            augmented_labels: 对应的标签
        """
        augmented_segments = list(segments)
        augmented_labels = list(labels)
        
        augmentation_methods = [
            self.time_warping,
            self.amplitude_scaling,
            self.jittering,
            self.random_rotation,
            self.dynamic_time_warping_noise,
        ]
        
        for segment, label in zip(segments, labels):
            for _ in range(augmentation_factor):
                # 随机选择增强方法
                aug_method = np.random.choice(augmentation_methods)
                augmented_segment = aug_method(segment)
                
                augmented_segments.append(augmented_segment)
                augmented_labels.append(label)
        
        return np.array(augmented_segments), np.array(augmented_labels)
    
    def augment_with_mixup(self, 
                          segments: np.ndarray,
                          labels: np.ndarray,
                          mixup_pairs: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Mixup 增强数据
        
        Args:
            segments: 原始分段
            labels: 标签
            mixup_pairs: Mixup 对数
            
        Returns:
            增强后的分段和标签
        """
        augmented_segments = list(segments)
        augmented_labels = list(labels)
        
        for _ in range(mixup_pairs):
            # 随机选择两个样本
            idx1, idx2 = np.random.choice(len(segments), 2, replace=False)
            
            mixed1, mixed2 = self.mixup(segments[idx1], segments[idx2])
            
            augmented_segments.extend([mixed1, mixed2])
            augmented_labels.extend([labels[idx1], labels[idx2]])
        
        return np.array(augmented_segments), np.array(augmented_labels)


def apply_combined_augmentation(segments: np.ndarray,
                               labels: np.ndarray,
                               num_channels: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用综合数据增强
    
    Args:
        segments: 输入分段
        labels: 标签
        num_channels: 通道数
        
    Returns:
        增强后的分段和标签
    """
    window_size = segments.shape[1]
    augmenter = FMGDataAugmentation(window_size, num_channels)
    
    # 第一步：基础增强（3倍）
    aug_segments, aug_labels = augmenter.augment_batch(
        segments, labels, augmentation_factor=3
    )
    
    # 第二步：Mixup 增强
    final_segments, final_labels = augmenter.augment_with_mixup(
        aug_segments, aug_labels, mixup_pairs=10
    )
    
    return final_segments, final_labels


if __name__ == "__main__":
    from preprocessing import load_sample_data, FMGDataPreprocessor
    
    # 加载数据
    print("加载示例数据...")
    signals, labels = load_sample_data()
    
    # 预处理
    preprocessor = FMGDataPreprocessor(window_size=200, num_channels=24)
    segments, segment_labels = preprocessor.process_batch(
        raw_signals=list(signals[:10]),
        labels=list(labels[:10])
    )
    
    print(f"原始分段形状: {segments.shape}")
    print(f"原始标签: {np.unique(segment_labels, return_counts=True)}")
    
    # 应用数据增强
    print("\n应用数据增强...")
    augmented_segments, augmented_labels = apply_combined_augmentation(
        segments, segment_labels, num_channels=24
    )
    
    print(f"增强后分段形状: {augmented_segments.shape}")
    print(f"增强后标签: {np.unique(augmented_labels, return_counts=True)}")
    print(f"增强倍数: {len(augmented_segments) / len(segments):.1f}x")
