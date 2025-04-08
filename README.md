# InferIntervene(推理干预)

基于论文[《Effectively Controlling Reasoning Models through Thinking Intervention》](https://arxiv.org/abs/2503.24370)实现的思维干预技术。

## 项目介绍

思维干预是一种在大型语言模型(LLM)推理过程中插入或修改思考步骤的方法，用于引导模型的推理过程。该方法不需要修改模型参数，只需要在模型生成过程中的中间步骤进行干预，因此可以轻松应用于各种开源模型。

该项目实现了论文中提出的思维干预方法，并提供了一个简单的演示应用，展示思维干预在以下方面的效果：

- **指令遵循**：提高模型按照用户指令行事的能力
- **指令层次性**：帮助模型处理多层次指令和优先级
- **安全对齐**：确保模型拒绝不安全的请求并提供合规的回答
- **数学推理**：提高模型在数学问题上的推理能力
- **反事实思维**：帮助模型避免生成错误信息和幻觉

## 工作原理

思维干预的核心思想是在模型的推理过程中插入特定的指导内容，这些内容可以引导模型沿着更加安全、准确、有效的推理路径前进。与传统的输入提示工程（Prompt Engineering）不同，思维干预直接操作模型的推理过程，而不仅仅是修改输入。

![思维干预工作原理](https://i.imgur.com/xxxxxxxxx.png)

思维干预的主要步骤：

1. 在用户输入后，引导模型进入显式的"思考模式"
2. 在思考模式开始时（或中间某个位置）插入特定的干预文本
3. 让模型继续生成思考过程
4. 基于生成的思考过程，引导模型生成最终答案

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- gradio (用于演示界面)

### 安装步骤

1. 克隆本仓库：

```bash
git clone https://github.com/yourusername/thinking-intervention.git
cd thinking-intervention
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行工具

可以通过命令行直接使用思维干预功能：

```bash
python thinking_intervention.py --prompt "你的问题或指令" --intervention "你的干预文本" --model "模型名称" --position "beginning"
```

参数说明：
- `--prompt`: 输入提示
- `--intervention`: 干预文本
- `--model`: 使用的HuggingFace模型名称 (默认: Qwen/Qwen-1_8B)
- `--position`: 干预位置，可以是"beginning"（开始）或"middle"（中间）
- `--compare`: 添加此标志可以同时运行不干预的版本进行比较
- `--device`: 运行设备 (默认: cuda 如果可用，否则 cpu)

### Web界面演示

运行Web演示界面：

```bash
python thinking_intervention_demo.py
```

这将启动一个Gradio界面，您可以在浏览器中进行交互式的思维干预演示。

## 支持的模型

该项目支持任何基于transformers库的开源大型语言模型，已测试的模型包括：

- Qwen/Qwen-1_8B
- Qwen/Qwen-7B
- meta-llama/Llama-2-7b-hf
- baichuan-inc/Baichuan2-7B-Base

## 自定义干预策略

您可以创建自己的干预策略，方法是直接在Web界面中选择"自定义"策略并输入您的干预文本，或者在`thinking_intervention_demo.py`文件中的`INTERVENTION_STRATEGIES`字典中添加新的策略。

## 引用

如果您在研究中使用了这个项目，请引用原论文：

```
@article{wu2024effectively,
  title={Effectively Controlling Reasoning Models through Thinking Intervention},
  author={Wu, Tong and Xiang, Chong and Wang, Jiachen T. and Mittal, Prateek},
  journal={arXiv preprint arXiv:2503.24370},
  year={2024}
}
```

## 贡献

欢迎提交问题和拉取请求。对于重大更改，请先打开一个问题来讨论您想要更改的内容。

## 许可证

[MIT License](LICENSE) 