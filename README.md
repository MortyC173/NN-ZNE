# Quantum Ising Model with Nerual Network Zero Noise Extrapolation (NN-ZNE)

## 项目概述

本项目实现了基于4量子比特横向场伊辛模型(TFIM)的量子计算仿真，具有以下特色：

- **零噪声外推(ZNE)**：使用时间缩放而非门折叠的方法进行噪声外推
- **现实噪声模型**：实现了T1(振幅阻尼)和T2(相位阻尼)噪声
- **多种外推方法**：支持线性、二次、三次多项式、指数、Richardson和有理函数拟合
- **机器学习优化**：使用多层感知机(MLP)优化ZNE外推结果
- **完整的仿真流程**：从基态到时间演化的完整物理仿真

## 系统要求

### Python环境
- Python 3.8 或更高版本
- 推荐使用 Python 3.9-3.11

### 必需依赖包
```
numpy >= 1.20.0
matplotlib >= 3.3.0
scipy >= 1.7.0
cqlib >= 0.1.0
```

### 可选依赖包(用于机器学习功能)
```
tensorflow >= 2.8.0
keras >= 2.8.0
```

## 安装指南

### 方法1：使用pip安装依赖

```bash
# 安装基础依赖
pip install numpy matplotlib scipy

# 安装cqlib量子计算库
pip install cqlib

# 可选：安装TensorFlow用于机器学习功能
pip install tensorflow
```

### 方法2：使用conda环境

```bash
# 创建新的conda环境
conda create -n quantum_ising python=3.9
conda activate quantum_ising

# 安装基础科学计算包
conda install numpy matplotlib scipy

# 使用pip安装专门的量子计算包
pip install cqlib

# 可选：安装TensorFlow
conda install tensorflow
```

## 项目结构

```
quantum_ising_zne/
├── src/                        # 核心源代码
│   ├── __init__.py             # 包初始化
│   ├── noise_models.py         # 噪声模型实现
│   ├── quantum_gates.py        # 量子门实现
│   ├── circuits.py             # 量子电路构建
│   ├── zne_methods.py          # ZNE方法实现
│   ├── analytics.py            # 解析解计算
│   └── simulations.py          # 主要仿真功能
├── configs/                    # 配置文件
│   └── config.py               # 参数配置
├── examples/                   # 示例脚本
│   ├── basic_example.py        # 基础示例
│   └── complete_simulation.py  # 完整仿真示例
├── output/                     # 输出目录(运行时创建)
└── README.md                   # 本文件
```

## 使用方法

### 快速开始

1. **基础仿真示例**：
```bash
cd examples
python basic_example.py
```

2. **完整仿真流程**：
```bash
cd examples  
python complete_simulation.py
```

### 核心功能使用

#### 1. 基态磁化仿真
```python
from src.simulations import simulate_ground_state
import numpy as np

# 定义lambda参数范围
lambdas = np.linspace(0.1, 2.5, 20)

# 运行基态仿真
lambdas_result, magnetizations = simulate_ground_state(
    lambdas=lambdas,
    shots=8192
)
```

#### 2. 时间演化仿真
```python
from src.simulations import simulate_time_evolution

# 无噪声时间演化
noiseless_results, times = simulate_time_evolution(
    lambdas_to_sim=[0.5, 0.9, 1.8],
    times=np.linspace(0, 2, 51),
    shots=8192
)
```

#### 3. ZNE噪声校正
```python
from src.simulations import simulate_time_evolution_with_zne

# ZNE仿真参数
noise_params = {
    'T1': 10.0,          # T1相干时间(微秒)
    'T2': 15.0,          # T2相干时间(微秒)  
    'dt_1q': 0.05,       # 单量子比特门时间(微秒)
    'dt_2q': 0.3,        # 双量子比特门时间(微秒)
    'shots': 8192,
    'noise_scales': [1.0, 2.0, 3.0, 4.0, 5.0]
}

# 运行ZNE仿真
zne_results, zne_times = simulate_time_evolution_with_zne(
    lambdas_to_sim=[0.5, 0.9, 1.8],
    times=np.linspace(0, 2, 31),
    noise_params=noise_params
)
```

## 参数配置

主要参数可以在 `configs/config.py` 中修改：

### 物理参数
- `T1_COHERENCE_TIME`: T1相干时间(默认10.0微秒)
- `T2_COHERENCE_TIME`: T2相干时间(默认15.0微秒)
- `SINGLE_QUBIT_GATE_TIME`: 单量子比特门时间(默认0.05微秒)
- `TWO_QUBIT_GATE_TIME`: 双量子比特门时间(默认0.3微秒)

### 仿真参数
- `SHOTS`: 测量次数(默认8192)
- `N_QUBITS`: 量子比特数(固定为4)
- `NOISE_SCALES`: ZNE噪声缩放因子(默认[1.0, 2.0, 3.0, 4.0, 5.0])

### 机器学习参数
- `MLP_CONFIG`: MLP模型配置
  - `epochs`: 训练轮数(默认500)
  - `batch_size`: 批处理大小(默认64)
  - `learning_rate`: 学习率(默认0.002)

## 输出文件说明

运行仿真后，会在 `output/` 目录下生成以下文件：

### 图像文件
- `ground_state_exact_vs_lambda_cqlib_corrected.png`: 基态磁化vs lambda
- `Time_evolution_cqlib_reproduction_corrected.png`: 无噪声时间演化
- `Time_evolution_ZNE_comprehensive_comparison.png`: ZNE方法对比
- `phase_diagram_reproduced_cqlib_corrected.png`: 相图
- `mlp_zne_training_history.png`: MLP训练历史(如果使用)

### 数据文件
- `zne_methods_error_stats.csv`: ZNE方法误差统计
- `zne_comparison_full_data.npz`: 完整比较数据
- `mlp_zne_training_data.npz`: MLP训练数据(如果使用)
- `zne_mlp_optimization_model.keras`: 训练好的MLP模型(如果使用)

## 技术特色

### 1. 时间缩放ZNE
- 保持电路深度不变，通过延长门操作时间实现噪声放大
- 避免了传统门折叠方法的电路深度增加问题

### 2. 现实噪声模型
- T1噪声：振幅阻尼(能量衰减)
- T2噪声：相位阻尼(相干性丢失)
- 使用Kraus算子实现完整的密度矩阵演化

### 3. 多种ZNE方法
- 线性外推
- 多项式拟合(二次、三次)
- 指数函数拟合
- Richardson外推
- 有理函数拟合
- MLP机器学习优化

### 4. 量子门分解
- 精确实现复合量子门(CH, FSWAP, Fourier, CRX, Bogoliubov)
- 基于原始量子门的分解，确保物理实现可能性

## 性能优化建议

### 计算资源
- **内存需求**: 基础仿真约需1-2GB RAM
- **CPU**: 多核处理器有助于加速计算
- **GPU**: 如果使用TensorFlow MLP功能，GPU可显著加速训练

### 参数调优
1. **快速测试**: 减少shots数量和时间点数
2. **高精度仿真**: 增加shots到16384或更高
3. **MLP训练**: 根据数据量调整训练参数

## 故障排除

### 常见问题

1. **ImportError: No module named 'cqlib'**
   ```bash
   pip install cqlib
   ```

2. **TensorFlow相关错误**
   ```bash
   # 对于CPU版本
   pip install tensorflow-cpu
   
   # 对于GPU版本
   pip install tensorflow-gpu
   ```

3. **内存不足错误**
   - 减少shots数量
   - 减少时间点数量
   - 减少lambda参数点数

4. **数值精度问题**
   - 检查噪声参数是否在合理范围内
   - 确保T1和T2时间不为零或无穷大

### 调试模式
可以在代码中添加调试输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```


## 贡献指南

欢迎提交问题报告和改进建议！

## 许可证

本项目使用MIT许可证。详见LICENSE文件。

## 联系信息

- 项目维护者: NUAA 量子信息与量子调控研究团队
- 创建日期: 2025年8月
- 版本: 1.0.0

