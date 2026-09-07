# FMG 灵巧手控制系统

## 项目概述

这是一个完整的灵巧手控制系统，集成了：
- **硬件部分**：柔性三层磁传感阵列（PCB + 磁传感器 + 柔性皮肤）
- **模型部分**：快速泛化性深度学习模型（训练时间 < 30秒）

### 系统架构

```
┌─────────────────────────────────────────────┐
│          护膝式柔性绑带                       │
├─────────────────────────────────────────────┤
│  ┌──────┐  ┌──────┐  ┌──────┐              │
│  │ 传感器 │  │ 传感器 │  │ 传感器 │  ... (10+个) │
│  │ 模块1  │  │ 模块2  │  │ 模块3  │              │
│  └──────┘  └──────┘  └──────┘              │
│    ↓        ↓        ↓                      │
│  ┌────────────────────────────┐           │
│  │   数据采集 & 预处理          │           │
│  └────────────────────────────┘           │
│    ↓                                       │
│  ┌────────────────────────────┐           │
│  │   深度学习模型 (1D-CNN)      │           │
│  │   (30秒快速训练)            │           │
│  └────────────────────────────┘           │
│    ↓                                       │
│  ┌────────────────────────────┐           │
│  │   手势识别结果              │           │
│  │   (灵巧手控制指令)           │           │
│  └────────────────────────────┘           │
└─────────────────────────────────────────────┘
```

## 项目结构

```
FMG-Dexterous-Hand-Control/
├── README.md                          # 项目文档
├── requirements.txt                   # Python 依赖
├── 
├── 1_hardware/                        # 硬件设计
│   ├── CAD_files/                    # CAD 设计文件
│   ├── BOM.csv                       # 物料清单
│   ├── assembly_guide.md             # 组装指南
│   └── sensor_specs.md               # 传感器规格
│
├── 2_data_processing/                # 数据处理
│   ├── data_collection.py            # 数据采集脚本
│   ├── preprocessing.py              # 数据预处理
│   ├── data_augmentation.py          # 数据增强
│   └── datasets/                     # 数据集目录
│       ├── sample_data.csv           # 示例数据
│       └── NinaPro_loader.py         # NinaPro 数据集加载器
│
├── 3_models/                         # 深度学习模型
│   ├── fast_cnn_model.py             # 1D-CNN 快速模型
│   ├── transfer_learning.py          # 迁移学习实现
│   ├── model_training.py             # 训练脚本 (<30秒)
│   ├── model_evaluation.py           # 模型评估
│   └── pretrained_weights/           # 预训练权重
│
├── 4_inference/                      # 实时推理
│   ├── real_time_inference.py        # 实时手势识别
│   ├── gesture_controller.py         # 手势控制命令
│   └── embedded_integration.py       # 嵌入式系统集成
│
├── 5_examples/                       # 使用示例
│   ├── quick_start.py               # 快速开始
│   ├── train_custom_model.py        # 训练自定义模型
│   └── real_time_demo.py            # 实时演示
│
└── docs/                             # 详细文档
    ├── hardware_guide.md            # 硬件指南
    ├── model_architecture.md        # 模型架构详解
    └── troubleshooting.md           # 常见问题
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 数据采集与预处理
```bash
python 2_data_processing/data_collection.py
python 2_data_processing/preprocessing.py
```

### 3. 快速训练模型（< 30秒）
```bash
python 3_models/model_training.py
```

### 4. 实时推理
```bash
python 4_inference/real_time_inference.py
```

## 核心特性

✅ **硬件集成**
- 三层柔性结构设计（PCB + 磁传感器 + 柔性皮肤）
- 10+ 磁传感器阵列
- 护膝式魔术贴绑带集成

✅ **快速训练**
- 训练时间 < 30秒
- 轻量级 1D-CNN 架构
- 迁移学习加速

✅ **强泛化性**
- 数据增强管道
- 交叉验证
- 多使用者适配

✅ **实时控制**
- 低延迟推理
- 嵌入式系统支持
- ROS 集成

## 参考资源

### 开源项目
- [Compliant-Magnetic-FMG-Array-for-Amputees](https://github.com/MartyWYCheng/Compliant-Magnetic-FMG-Array-for-Amputees) - 硬件设计参考
- [FMGinterface](https://github.com/newdexterity/FMGinterface) - 绑带集成参考

### 论文数据集
- **NinaPro** - 大型 EMG 数据集（50个使用者，52种手势）
- **Physionet** - 医疗生理信号数据集
- **Papers with Code** - 手势识别相关实现

## 许可证

MIT License - 见 LICENSE 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请提交 GitHub Issue。
