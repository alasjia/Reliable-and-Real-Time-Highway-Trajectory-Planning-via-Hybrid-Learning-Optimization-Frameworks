# velocity_predicition 目录说明文档

本目录实现了基于图神经网络（GNN）的高速公路车辆轨迹预测，核心流程包括数据预处理、模型训练与评估。主要依赖 PyTorch、PyTorch Geometric 及自定义模块。

> 请注意，如果你对将 "dataset_after_dp" 中的数据转换为 ".pt" 文件的过程不感兴趣，可以忽略 [HighD_datapre.py](HighD_datapre.py) 的使用，直接利用 [PyG_DataSet](PyG_DataSet)。

---

## 1. 核心文件说明

### 1.1 HighD_datapre.py

**功能**：  
负责将原始 HighD 数据集预处理为适用于 GNN 的图结构数据，并保存为 `.pt` 文件，便于后续高效加载。

**主要流程**：
- 读取原始 CSV（车辆、车道线信息）。
- 滑窗扩增：每个场景沿时间轴滑动，生成多个轨迹样本，极大丰富数据。
- 坐标归一化：目标车最后观测点归为原点，便于模型学习。
- 特征提取与缺失值处理，统一车辆与车道线的特征格式。
- 构建 PyTorch Geometric 的 Data 对象（节点特征、边、标签等）。
- 合并所有样本，按固定随机种子划分训练/验证/测试集，并生成对应 DataLoader。

**适用场景**：  
首次使用或需自定义数据预处理时运行。若已有 `.pt` 文件，可直接跳过。

**运行方式**：
```bash
python HighD_datapre.py
```

---

### 1.2 train_myway.py

**功能**：  
主训练脚本，基于预处理数据训练 VectorNet 轨迹预测模型。

**主要流程**：
- 加载 `.pt` 数据，构建训练/验证/测试集。
- 初始化模型（VectorNet，见 vectornet.py）、优化器、学习率调度器等。
- 训练循环：  
  - train_epoch：每轮遍历训练集，前向传播、计算损失、反向传播与参数更新。
  - eval_epoch：在验证集上评估模型性能，不更新参数。
- 日志与可视化：记录损失曲线，支持高质量绘图。
- 支持 GPU 自动切换、超参数灵活配置。

**运行方式**：
```bash
python train_myway.py
```

---

### 1.3 test_for_de.py

**功能**：  
用于评估已训练好的模型，计算轨迹预测的各类指标，并可视化预测效果。

**主要流程**：
- 加载训练好的模型参数。
- 加载测试集数据，批量推理。
- 计算平均位移误差（ADE）、最终位移误差（FDE）等指标。
- 可选：将预测轨迹与真实轨迹可视化，便于直观分析。

**运行方式**：
```bash
python test_for_de.py
```

---

## 2. 其他重要模块

- **basic_module.py**  
  定义基础神经网络模块（如 MLP），为主模型提供构建块。

- **vectornet.py**  
  实现 VectorNet 主体结构，包括子图聚合、全局图聚合等核心 GNN 逻辑。

- **global_graph.py、subgraph.py**  
  分别实现全局图和子图的聚合操作，服务于 VectorNet。

- **config_nw.py**  
  配置文件，集中管理数据路径、模型输入输出维度、超参数等全局常量。

- **utils/**  
  工具函数集合，包括损失函数、可视化、数据处理等辅助功能。

- **PyG_DataSet/**  
  存放已处理好的 PyTorch Geometric 数据集文件，可直接用于训练和评估。

- **results_vp/**  
  存放模型预测结果、可视化图片等输出。

vectornet的搭建体现在velocity_predicition/basic_module.py、velocity_predicition/subgraph.py、velocity_predicition/global_graph.py和velocity_predicition/vectornet.py。搭建过程参考了xxx.github.com

---

## 3. 使用流程建议

1. **数据预处理**  
   若无现成 `.pt` 文件，先运行 HighD_datapre.py 生成数据。

2. **模型训练**  
   配置好参数后，运行 train_myway.py 进行训练。

3. **模型评估**  
   训练完成后，运行 test_for_de.py 评估模型效果。

---

## 4. 备注

- 所有核心流程均支持 GPU 加速。
- 数据与模型参数路径、超参数等均可在 config_nw.py 或主脚本顶部灵活配置。
- 若仅需复现训练/测试流程，可直接使用 PyG_DataSet/ 下的现成数据，无需重新预处理。

---

