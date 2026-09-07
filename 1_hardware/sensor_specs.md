# MLX90393 磁传感器规格

## 概述

MLX90393 是一款 MEMS 磁传感器，用于检测磁场强度和方向。在 FMG 系统中，它用于检测肌肉收缩产生的磁场变化。

## 主要规格

### 传感范围

```
轴向 (Z轴，垂直于芯片平面):
  范围: ±16 mT (毫特斯拉)
  分辨率: 0.00163 mT/LSB
  
平面 (X/Y轴):
  范围: ±16 mT
  分辨率: 0.00163 mT/LSB
```

### 电气特性

| 参数 | 最小值 | 典型值 | 最大值 | 单位 |
|------|-------|-------|-------|------|
| 电源电压 (VDD) | 3.0 | 3.3 | 3.6 | V |
| 功耗 (工作) | - | 3 | 5 | mA |
| 功耗 (待机) | - | 0.1 | 0.5 | µA |
| 工作温度 | -40 | 25 | 85 | °C |
| I2C 速率 | - | - | 400 | kHz |

### 输出

```
分辨率: 16 位
数据格式: 有符号整数 (Two's complement)
输出频率: 可配置
  - 0.8 Hz
  - 1.6 Hz
  - 3.2 Hz
  - 6.4 Hz
  - 12.8 Hz
  - 25.6 Hz
  - 51.2 Hz
  - 102.4 Hz (推荐用于 FMG 系统)
```

## 引脚定义

```
             ┌────────────┐
         1 ─┤ VDD (3.3V) ├─ 8
         2 ─┤ SDA (I2C) ├─ 7 (不使用)
         3 ─┤ SCL (I2C)├─ 6 (不使用)
         4 ─┤ GND       ├─ 5 (不使用)
             └────────────┘
```

## I2C 通信

### 地址配置

```
I2C 基础地址: 0x0C

通过 ADDR0/ADDR1 引脚配置不同的地址:

ADDR1  ADDR0  I2C Address  Hex
─────────────────────────────
  0      0      0x0C        0x0C
  0      1      0x0D        0x0D
  1      0      0x0E        0x0E
  1      1      0x0F        0x0F
```

### 寄存器映射

| 地址 | 寄存器名 | 功能 |
|------|---------|------|
| 0x00 | STATUS | 传感器状态标志 |
| 0x01-0x06 | XOUT, YOUT, ZOUT | 磁场输出值 (X, Y, Z 轴) |
| 0x07-0x0C | XXOUT, XYOUT, XZOUT, YXOUT, YYOUT, YZOUT | 额外输出值 |
| 0x0D-0x13 | CONF1-CONF3 | 配置寄存器 |
| 0x14 | CTRL | 控制寄存器 |

## Python 驱动示例

```python
import smbus
import time

class MLX90393:
    def __init__(self, i2c_bus=1, address=0x0C):
        self.bus = smbus.SMBus(i2c_bus)
        self.address = address
        self.initialize()
    
    def initialize(self):
        """初始化传感器"""
        # 设置配置寄存器
        # 102.4 Hz 采样率
        self.write_register(0x0D, 0x70)
    
    def read_registers(self, reg, length=6):
        """读取多个寄存器"""
        return self.bus.read_i2c_block_data(self.address, reg, length)
    
    def write_register(self, reg, data):
        """写入寄存器"""
        self.bus.write_i2c_block_data(self.address, reg, [data])
    
    def read_magnetic_field(self):
        """读取磁场值 (X, Y, Z)"""
        # 读取磁场输出寄存器 (0x01-0x06)
        data = self.read_registers(0x01, 6)
        
        # 转换为有���号整数
        x = self._bytes_to_int(data[0:2])
        y = self._bytes_to_int(data[2:4])
        z = self._bytes_to_int(data[4:6])
        
        # 转换为毫特斯拉
        x_mT = x * 0.00163
        y_mT = y * 0.00163
        z_mT = z * 0.00163
        
        return {
            'x': x_mT,
            'y': y_mT,
            'z': z_mT,
            'magnitude': (x_mT**2 + y_mT**2 + z_mT**2) ** 0.5
        }
    
    @staticmethod
    def _bytes_to_int(byte_array):
        """将两个字节转换为有符号整数"""
        value = (byte_array[0] << 8) | byte_array[1]
        if value >= 0x8000:
            value -= 0x10000
        return value

# 使用示例
if __name__ == "__main__":
    sensor = MLX90393(address=0x0C)
    
    for i in range(100):
        data = sensor.read_magnetic_field()
        print(f"X: {data['x']:.3f} mT, "
              f"Y: {data['y']:.3f} mT, "
              f"Z: {data['z']:.3f} mT, "
              f"Mag: {data['magnitude']:.3f} mT")
        time.sleep(0.01)  # 100 Hz 采样率
```

## 应用注意事项

### 磁铁位置

在 FMG 系统中，应在传感器下方（靠近皮肤）放置小型永磁体：

```
皮肤表面
   ↓
┌────────────────┐
│  柔性皮肤层     │
├────────────────┤
│ 磁传感器 (MLX)  │  ← 感应磁场
├────────────────┤
│ [磁铁]         │  ← 产生磁场
├────────────────┤
│ PCB / 绑带      │
└────────────────┘

磁铁规格：
- 大小：5×5×1 mm
- 材料：钕铁硼 (NdFeB)
- 磁感应强度：约 0.3 T @ 表面
```

### 磁场干扰

避免以下干扰源：
- 手机和平板电脑（2-10 cm 范围内）
- 电磁炉和微波炉
- 高压线和变压器
- 其他磁性物体（手表、眼镜等）

### 温度补偿

MLX90393 具有温度漂移。为获得最佳准确性：

```python
def temperature_compensated_reading(sensor, calibration_temp=25):
    """温度补偿读取"""
    # 典型温度系数：约 0.1% / °C
    current_temp = get_temperature()  # 需要额外的温度传感器
    temp_drift = (current_temp - calibration_temp) * 0.001
    
    raw_data = sensor.read_magnetic_field()
    
    # 应用补偿
    compensated_x = raw_data['x'] / (1 + temp_drift)
    compensated_y = raw_data['y'] / (1 + temp_drift)
    compensated_z = raw_data['z'] / (1 + temp_drift)
    
    return compensated_x, compensated_y, compensated_z
```

## 相关文档

- [MLX90393 数据手册](https://www.melexis.com/en/product/mlx90393/)
- [应用笔记](https://www.melexis.com/en/documents/documentation/application-notes/)
