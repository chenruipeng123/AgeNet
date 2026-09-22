下面是一份可直接保存为 README.md 的内容，基于你仓库里的代码整理而成：

```md
# AgeNet

AgeNet 是一个面向海洋/水体目标分割的深度学习项目，主要用于从遥感影像或水体图像中识别和分割赤潮/藻华区域。该项目结合了多种语义分割模型结构，并提供了训练、验证和推理流程。

该仓库中的核心代码包括：
- `train.py`：训练入口
- `predict.py`：推理/评估入口
- `dataset/TT_Dataset.py`：数据集加载与样本读取
- `model/`：模型实现目录
- `mdoel/`：额外模型实现目录
- `utils/`：工具函数与辅助模块

---

## 项目概述

本项目的目标是对图像中的目标区域（如赤潮、藻华、海藻覆盖区域）进行二分类语义分割，典型任务是：
- 输入：RGB 图像
- 输出：二值分割掩码
- 任务：区分“目标区域”和“背景区域”

代码中采用了多个常见语义分割网络，包括：
- UNet
- DeepLabV3+
- FCN
- HRNet
- UPerNet
- SegFormer
- PSPNet
- ABCNet
- AgeNet（自定义模型）
- AlgaeNet
- WDSR

其中，`model/AgeNet.py` 中定义的 `densehrnet` 是当前仓库中最核心的自定义模型之一，具备密集连接与 deformable / attention 机制，适合复杂边界分割场景。

---

## 仓库结构

```text
AgeNet/
├── README.md
├── train.py
├── predict.py
├── data/                  # 示例数据或可视化资源
├── dataset/
│   └── TT_Dataset.py     # 数据加载与标签读取
├── model/
│   ├── AgeNet.py
│   ├── ABCNet.py
│   ├── AlgaeNet.py
│   ├── WDSR.py
│   └── ...
├── mdoel/
│   ├── unet.py
│   ├── DeeplabV3Plus.py
│   ├── FCN_ResNet.py
│   ├── vit_model.py
│   ├── HRNet.py
│   ├── Upernet.py
│   ├── segformer/
│   └── ...
├── utils/
│   ├── Logger.py
│   └── flops_counter/
└── ...
```

---

## 主要功能

1. 数据读取
   - `dataset/TT_Dataset.py` 中定义了 `MyDataset`
   - 自动扫描图像目录下的 `.jpg` 文件和标签目录下的 `.png` 文件
   - 将图像转换为 tensor，标签归一化到 `[0, 1]`

2. 模型训练
   - `train.py` 中实现了训练逻辑
   - 支持多种模型切换
   - 使用 `CrossEntropyLoss`
   - 训练时使用 AdamW 优化器
   - 具备保存模型权重的能力

3. 模型推理
   - `predict.py` 中实现了批量预测
   - 读取测试图像并生成预测结果
   - 对预测结果进行评估（Accuracy, IoU, Recall, Precision, F1）
   - 将错误区域可视化为不同颜色

4. 可视化结果
   - 预测图像会按像素类别绘制颜色标注
   - 绿色：正确检测区域
   - 白色：误检区域
   - 红色：漏检区域

---

## 环境依赖

建议使用 Python 3.8+，并安装以下依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python pillow imageio
pip install torchsummary
```

若需要在 GPU 上运行，建议使用 CUDA 版本的 PyTorch。

---

## 数据集格式

项目中的数据读取逻辑默认要求：

- 图像目录：`.jpg`
- 标签目录：`.png`
- 图像与标签应一一对应
- 标签通常使用二值掩码，像素值为：
  - 0：背景
  - 255：目标区域（在代码中会除以 255 归一化）

例如：

```text
imagePath = /your/path/imgs
labelPath = /your/path/masks
```

代码中也提供了注释示例：
```python
imagePath = r"/chenruipeng/GreenTide/new_data/imgs"
labelPath = r"/chenruipeng/GreenTide/new_data/masks"
```

你需要把这些路径替换为自己的数据集目录。

---

## 训练

进入项目目录后，修改 `train.py` 中的路径和模型名称，然后运行：

```bash
python train.py
```

关键参数说明：
- `num_classes = 2`：二分类
- `batch_size = 8`
- `end_ep = 200`：训练轮数
- `model_name = "abcnet"`：默认模型
- `save_path`：模型保存目录

你也可以手动切换模型，例如：
- `"unet"`
- `"deeplabv3p"`
- `"fcn_resnet"`
- `"hrnet"`
- `"segformer"`
- `"abcnet"`
- `"densehrnet"`
- `"algaenet"`

---

## 推理/预测

修改 `predict.py` 中的路径后执行：

```bash
python predict.py
```

推理脚本会：
- 加载指定模型权重
- 读取测试图像
- 逐块预测
- 计算评估指标
- 保存可视化结果图

关键参数：
```python
image_size = 256
modelname = "unet"
imagedir = r"/your/path/test_img"
labeldir = r"/your/path/test_mask"
modelPath = r"save_model//"+modelname
savePath = r"/your/path/save_res_new_2024/"+modelname
```

---

## 模型说明

### 1. AgeNet / densehrnet
`model/AgeNet.py` 中定义的 `densehrnet` 是本项目的核心网络之一，整体思路类似 DenseNet + HRNet 的组合，重点在于：
- 密集连接
- 多尺度特征融合
- deformable conv / attention 机制
- 边缘感知特征强化

适合于目标边界复杂、分割任务要求较高的场景。

### 2. ABCNet
`model/ABCNet.py` 中的 `ABCNet` 是基于 attention + context path + spatial path 的网络结构，适合增强特征表达和语义边界恢复。

### 3. 其他模型
其他模型如 UNet、DeepLabV3+、FCN、SegFormer 等，是常见主干网络，方便在不同任务场景中做实验对比。

---

## 评估指标

推理脚本中计算了以下指标：

- Accuracy
- IoU
- Recall
- Precision
- F1-score

核心实现见 `predict.py` 中的 `estimate()` 函数。

---

## 注意事项

1. 代码中有许多绝对路径，如：
   ```python
   /chenruipeng/GreenTide/...
   ```
   这些路径是本地环境路径，实际使用时需要替换成你的数据目录。

2. 仓库中存在 `mdoel` 与 `model` 两个目录，命名中有 `mdoel`（拼写错误），但功能上对应多种模型实现。
   - 需要确保导入路径与文件名一致
   - 如果有 import 错误，可根据实际文件名修正

3. 代码中有一些历史实验性的参数和注释，可能仍保留了旧数据集/旧任务配置，使用前请根据你的数据集进行调试。

4. 如果使用 GPU，需要确认 CUDA 版本和 PyTorch 版本兼容。

---

## 参考用途

本项目适用于：
- 海洋/水体目标检测
- 赤潮检测
- 藻华区域分割
- 遥感影像语义分割
- 复杂背景下的二值分割研究

---

## 许可证

本仓库未显式声明许可证信息，若用于公开发布或商业使用，请先确认代码原始作者的授权要求。

---

## 结论

AgeNet 是一个以深度学习语义分割为核心的水体目标检测项目，覆盖了训练、验证、推理和结果可视化的完整流程。它适合用于藻华/赤潮分割等场景，并且支持多种分割模型进行实验对比。

如果你愿意，我还可以继续帮你做两件事中的任意一个：
1. 把这份 README 再优化成更正式的 GitHub 风格版本
2. 直接生成一份适合你仓库的中文/英文双语 README，并按你当前代码结构精修
``` 

如果你想，我也可以继续直接给你一版“更像正式 GitHub 展示用”的 README，适合直接提交到仓库里。
