# 快速启动指南

## 1. 测试安装

首先确保所有依赖都正确安装：

```bash
cd quantum_ising_zne
python test_installation.py
```

## 2. 安装依赖

如果测试失败，安装必需的依赖：

```bash
# 基础依赖
pip install numpy matplotlib scipy cqlib

# 可选：机器学习功能
pip install tensorflow
```

## 3. 运行示例

### 基础示例（快速测试）
```bash
python examples/basic_example.py
```

### 完整仿真（包含ZNE和MLP）
```bash
python examples/complete_simulation.py
```

### 使用命令行界面
```bash
# 查看帮助
python main.py --help

# 运行基态仿真
python main.py ground-state --shots 4096

# 运行时间演化
python main.py time-evolution --lambdas "0.5,1.0" --time-points 21

# 运行ZNE仿真
python main.py zne --t1 15.0 --t2 20.0 --shots 4096

# 运行完整仿真
python main.py full --shots 8192
```

## 4. 输出文件

所有结果将保存在 `output/` 目录中：

- **图像文件**: `.png` 格式的仿真结果图
- **数据文件**: `.csv` 和 `.npz` 格式的数据
- **模型文件**: `.keras` 格式的训练好的机器学习模型

## 5. 自定义参数

编辑 `configs/config.py` 来修改默认参数：

- 噪声参数 (T1, T2时间)
- 仿真参数 (shots数量, 时间点)
- 机器学习参数 (训练轮数, 学习率)

## 6. 故障排除

如果遇到问题：

1. **导入错误**: 确保在正确的目录中运行
2. **内存不足**: 减少shots数量或时间点数
3. **TensorFlow错误**: 这是可选功能，可以跳过

## 7. 功能特色

- ✅ 零噪声外推 (ZNE) 
- ✅ 时间缩放噪声模型
- ✅ 多种拟合方法对比
- ✅ 机器学习优化 (可选)
- ✅ 完整的误差分析
- ✅ 可视化结果输出
