[**🇨🇳中文**](https://github.com/shibing624/thinking-intervention/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/thinking-intervention/blob/main/README_EN.md)

<div align="center">
  <a href="https://github.com/shibing624/thinking-intervention">
    <img src="https://raw.githubusercontent.com/shibing624/thinking-intervention/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# 思维干预 (Thinking Intervention)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/thinking-intervention.svg)](https://github.com/shibing624/thinking-intervention/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


基于论文 [《Effectively Controlling Reasoning Models through Thinking Intervention》](https://arxiv.org/abs/2503.24370) 实现的思维干预技术，用于引导大语言模型的推理过程。

## 🌟 功能特点

- **直接干预**：在提示后直接添加干预文本，引导模型输出
- **流式生成**：支持流式文本生成，实时返回生成内容
- **JSON响应**：支持以JSON格式返回响应，便于集成到应用中
- **简单易用**：提供简洁的API，易于在各种应用中集成

## 🔍 思维干预简介

思维干预是一种在LLM推理过程中插入或修改思考步骤的方法，用于引导模型的推理过程。通过在模型生成思考的关键位置（一般是开头）注入干预文本，可以显著改变模型的输出方向和质量。

## 🚀 快速开始

### 安装依赖

```bash
pip install torch transformers loguru
```

### 命令行使用

```bash
# 使用干预文本生成JSON格式回答，演示了流式和非流式生成
python thinking_intervention.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --prompt "一句话介绍北京" --intervention "我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。"
```

### Python代码使用

```python
from thinking_intervention import ThinkingIntervention
ti = ThinkingIntervention(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    system_prompt="你是一个有用的助手。中文回答"
)

# 非流式生成
response = ti.intervene_generate(
    prompt="一句话介绍北京",
    intervention_text="我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。"
)
print(response)

# 流式生成
for chunk in ti.stream_intervene_generate(
    prompt="一句话介绍北京",
    intervention_text="我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。"
):
    print(chunk.decode())
```

output:
![](https://github.com/shibing624/thinking-intervention/blob/main/docs/show_img.png)

#### 高级思维干预示例

```shell
python demo.py
```

output:
![](https://github.com/shibing624/thinking-intervention/blob/main/docs/sample.png)

## 🌈 干预策略示例

以下是一些常用的干预策略：

### JSON格式化输出

```
我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。
```

### 指令遵循增强

```
我需要仔细理解用户的指令，并确保我的回答符合所有要求。
1. 首先，我会分析用户指令中的所有关键要求和约束条件
2. 我会确保我的回答满足所有这些要求，不遗漏任何关键点
3. 如果指令中有矛盾或不明确的地方，我会以用户明确表达的要求为准
4. 我不会添加用户未要求的额外信息或功能
5. 我会特别注意指令中的格式、长度和风格要求
```

### 安全内容引导

```
我必须确保我的回答是安全、合规且有益的。
1. 我会拒绝生成任何可能导致伤害的内容，包括但不限于：违法活动、有害建议、歧视性言论等
2. 对于边界情况，我会倾向于保守和安全的解释
3. 如果需要拒绝请求，我会解释原因并尝试提供合适的替代建议
4. 我会确保我的回答符合道德准则和社会规范
5. 我不会提供可能被误用或滥用的信息或方法
```

## 📊 使用场景

- **Web应用集成**：在网站或应用中使用流式生成API
- **内容格式控制**：强制模型输出特定格式（如JSON、Markdown）
- **安全内容过滤**：引导模型生成符合安全标准的内容
- **特定领域优化**：针对数学、编程等领域优化模型输出


## ☎️ Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/thinking-intervention.svg)](https://github.com/shibing624/thinking-intervention/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/thinking-intervention/blob/main/docs/wechat.jpeg" width="200" />

## 😇 Citation

如果你在研究中使用了`thinking-intervention`，请按如下格式引用：

APA:

```
Xu, M. thinking-intervention: Effectively Controlling Reasoning Models through Thinking Intervention (Version 0.0.1) [Computer software]. https://github.com/shibing624/thinking-intervention
```

BibTeX:

```
@misc{Xu_thinking-intervention,
  title={thinking-intervention: Effectively Controlling Reasoning Models through Thinking Intervention},
  author={Xu Ming},
  year={2025},
  howpublished={\url{https://github.com/shibing624/thinking-intervention}},
}
```

## ⚠️ License

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加`thinking-intervention`的链接和授权协议。
## 😍 Contribute

项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

- 在`tests`添加相应的单元测试
- 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

## 💕 Acknowledgements

- [《Effectively Controlling Reasoning Models through Thinking Intervention》](https://arxiv.org/abs/2503.24370)

Thanks for their great work!