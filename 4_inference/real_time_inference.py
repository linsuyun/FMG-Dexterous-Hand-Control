"""
实时推理模块
用于实时手势识别和控制
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Tuple, Optional
import time


class RealTimeInference:
    """
    实时推理引擎
    """
    
    def __init__(self, 
                 model: nn.Module,
                 window_size: int = 200,
                 stride: int = 50,
                 device: str = 'cuda',
                 smoothing_window: int = 3):
        """
        初始化实时推理模块
        
        Args:
            model: 训练正亚的模型
            window_size: 时间窗口大小
            stride: 滑动步上
            device: 计算设备
            smoothing_window: 平滑窗口大小
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.smoothing_window = smoothing_window
        
        # 数据缓存
        self.data_buffer = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=smoothing_window)
        
        # 统计信息
        self.inference_times = deque(maxlen=100)
        self.last_gesture = None
        self.gesture_confidence = 0
    
    def add_data(self, new_data: np.ndarray) -> None:
        """
        添加新的传感器数据
        
        Args:
            new_data: 新数据 (shape: [num_channels] 或 [batch, num_channels])
        """
        if len(new_data.shape) == 1:
            # 单个样本
            self.data_buffer.append(new_data)
        else:
            # 批量样本
            for sample in new_data:
                self.data_buffer.append(sample)
    
    def get_window(self) -> Optional[torch.Tensor]:
        """
        获取当前时间窗口数据
        
        Returns:
            时间窗口张量 (shape: [num_channels, window_size])
        """
        if len(self.data_buffer) < self.window_size:
            return None
        
        # 转换为 numpy 数组
        window_data = np.array(list(self.data_buffer))  # [window_size, num_channels]
        window_data = window_data.T  # [num_channels, window_size]
        
        return torch.from_numpy(window_data.astype(np.float32)).to(self.device)
    
    def predict(self, return_all_probs: bool = False) -> Tuple[int, float, dict]:
        """
        执行一次预测
        
        Args:
            return_all_probs: 是否返回所有污率
            
        Returns:
            (gesture_id, confidence, info_dict)
        """
        window = self.get_window()
        
        if window is None:
            return -1, 0.0, {'status': 'buffering'}
        
        start_time = time.time()
        
        with torch.no_grad():
            # 添加 batch 维度
            input_tensor = window.unsqueeze(0)  # [1, num_channels, window_size]
            
            # 执行推理
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            
            gesture_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0, gesture_id].item()
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 平滑一致性
        self.prediction_history.append(gesture_id)
        smoothed_gesture = max(set(self.prediction_history), key=list(self.prediction_history).count)
        
        info = {
            'status': 'success',
            'gesture_id': gesture_id,
            'confidence': float(confidence),
            'smoothed_gesture': smoothed_gesture,
            'inference_time_ms': inference_time * 1000,
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000
        }
        
        if return_all_probs:
            info['all_probs'] = probs[0].cpu().numpy()
        
        self.last_gesture = smoothed_gesture
        self.gesture_confidence = confidence
        
        return smoothed_gesture, confidence, info
    
    def get_statistics(self) -> dict:
        """
        获取推理统计信息
        
        Returns:
            统计信息字典
        """
        if not self.inference_times:
            return {
                'total_inferences': 0,
                'avg_inference_time_ms': 0,
                'min_inference_time_ms': 0,
                'max_inference_time_ms': 0
            }
        
        inference_times_ms = np.array(self.inference_times) * 1000
        
        return {
            'total_inferences': len(self.inference_times),
            'avg_inference_time_ms': float(np.mean(inference_times_ms)),
            'min_inference_time_ms': float(np.min(inference_times_ms)),
            'max_inference_time_ms': float(np.max(inference_times_ms)),
            'std_inference_time_ms': float(np.std(inference_times_ms)),
            'buffer_size': len(self.data_buffer)
        }


class GestureController:
    """
    手势控制模块
    将预测的手势转换为控制指令
    """
    
    # 不同手势的憧时时间（了会两次同一个手势和前次手势的结果）
    DEBOUNCE_TIME = 0.5  # 秒
    
    # 信心度阈值
    CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self, gesture_names: list = None):
        """
        初始化手势控制模块
        
        Args:
            gesture_names: 手势名称列表
        """
        self.gesture_names = gesture_names or [
            'Rest', 'Pinch', 'Power', 'Tripod', 'Extension'
        ]
        
        self.last_gesture = -1
        self.last_gesture_time = 0
        self.gesture_callbacks = {}
    
    def register_gesture_callback(self, gesture_id: int, callback):
        """
        为找个手势注册回调函数
        
        Args:
            gesture_id: 手势 ID
            callback: 回调函数
        """
        self.gesture_callbacks[gesture_id] = callback
    
    def process_prediction(self, gesture_id: int, confidence: float) -> Optional[str]:
        """
        处理预测结果并执行控制指令
        
        Args:
            gesture_id: 预测的手势 ID
            confidence: 信心度
            
        Returns:
            执行的指令
        """
        current_time = time.time()
        
        # 检查信心度
        if confidence < self.CONFIDENCE_THRESHOLD:
            return None
        
        # 检查憧时时间
        if (gesture_id == self.last_gesture and 
            current_time - self.last_gesture_time < self.DEBOUNCE_TIME):
            return None
        
        # 更新状态
        self.last_gesture = gesture_id
        self.last_gesture_time = current_time
        
        # 执行回调函数
        if gesture_id in self.gesture_callbacks:
            self.gesture_callbacks[gesture_id]()
        
        return self.gesture_names[gesture_id]
    
    def get_gesture_name(self, gesture_id: int) -> str:
        """
        获取手势名称
        
        Args:
            gesture_id: 手势 ID
            
        Returns:
            手势名称
        """
        if 0 <= gesture_id < len(self.gesture_names):
            return self.gesture_names[gesture_id]
        return "Unknown"


class StreamingGestureRecognizer:
    """
    流式手势识别系统
    组合实时推理和控制
    """
    
    def __init__(self, 
                 model: nn.Module,
                 gesture_names: list = None,
                 **inference_kwargs):
        """
        初始化流式识别系统
        
        Args:
            model: 训练正亚的模型
            gesture_names: 手势名称列表
            **inference_kwargs: 实时推理参数
        """
        self.inference = RealTimeInference(model, **inference_kwargs)
        self.controller = GestureController(gesture_names)
    
    def process_stream(self, 
                      data_stream: np.ndarray) -> dict:
        """
        处理数据流
        
        Args:
            data_stream: 数据流 (shape: [num_channels] 或 [batch, num_channels])
            
        Returns:
            处理结果字典
        """
        # 添加数据
        self.inference.add_data(data_stream)
        
        # 执行推理
        gesture_id, confidence, info = self.inference.predict(return_all_probs=True)
        
        # 处理预测结果
        command = self.controller.process_prediction(gesture_id, confidence)
        
        return {
            'gesture_id': gesture_id,
            'gesture_name': self.controller.get_gesture_name(gesture_id),
            'confidence': confidence,
            'command': command,
            'info': info
        }
    
    def get_status(self) -> dict:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        return {
            'inference_stats': self.inference.get_statistics(),
            'last_gesture': self.controller.get_gesture_name(self.inference.last_gesture),
            'gesture_confidence': self.inference.gesture_confidence
        }


if __name__ == "__main__":
    from fast_cnn_model import FastGestureCNN
    
    # 创建模型
    print("创建模型...")
    model = FastGestureCNN(num_channels=24, num_gestures=5)
    
    # 创建实时识别系统
    print("创建实时识别系统...")
    gesture_names = ['Rest', 'Pinch', 'Power', 'Tripod', 'Extension']
    recognizer = StreamingGestureRecognizer(model, gesture_names)
    
    # 模拟传感器数据
    print("模拟传感器数据流...")
    for i in range(300):
        # 隐漏混合数据
        if i < 100:
            data = np.random.randn(24) * 0.5  # Rest 手势
        elif i < 200:
            data = np.random.randn(24) * 1.5 + 1.0  # Pinch 手势
        else:
            data = np.random.randn(24) * 2.0 - 1.0  # 不同手势
        
        result = recognizer.process_stream(data)
        
        if result['command']:
            print(f"Step {i}: {result['gesture_name']} "
                  f"(confidence: {result['confidence']:.3f}) "
                  f"- Command: {result['command']}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("统计信息:")
    status = recognizer.get_status()
    for key, value in status['inference_stats'].items():
        print(f"  {key}: {value}")
