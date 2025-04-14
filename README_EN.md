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

An implementation of thinking intervention techniques based on the paper "Effectively Controlling Reasoning Models through Thinking Intervention" to guide the reasoning process of large language models.

## 🌟 Features

- **Direct Intervention**: Add intervention text after prompts to guide model outputs
- **Streaming Generation**: Support for streaming text generation with real-time responses
- **JSON Response**: Return responses in JSON format for easy application integration
- **Simple API**: Provide a clean, simple API for easy integration in various applications

## 🔍 What is Thinking Intervention?

Thinking intervention is a method of inserting or modifying reasoning steps in the LLM inference process to guide the model's reasoning. By injecting intervention text at critical points during model generation, you can significantly change the direction and quality of the model's output.

## 🚀 Quick Start

### Install Dependencies

```bash
pip install torch transformers loguru
```

### Command Line Usage

```bash
# Generate JSON formatted answer with intervention text
python thinking_intervention.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --prompt "Describe Beijing in one sentence" --intervention "I need to respond in JSON format and only return the content inside the JSON, without any other information."

# Streaming generation
python thinking_intervention.py --prompt "Describe Beijing in one sentence" --intervention "I need to respond in JSON format and only return the content inside the JSON, without any other information."
```

### Python Code Usage

```python
from thinking_intervention import ThinkingIntervention

# Initialize
ti = ThinkingIntervention(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    system_prompt="You are a helpful assistant."
)

# Non-streaming generation
response = ti.intervene_generate(
    prompt="Describe Beijing in one sentence",
    intervention_text="I need to respond in JSON format and only return the content inside the JSON, without any other information."
)
print(response)

# Streaming generation
for chunk in ti.stream_intervene_generate(
    prompt="Describe Beijing in one sentence",
    intervention_text="I need to respond in JSON format and only return the content inside the JSON, without any other information."
):
    print(chunk.decode())
```
output:
![](https://github.com/shibing624/thinking-intervention/blob/main/docs/show_img.png)

#### Advanced Thinking Intervention Example

```shell
python demo.py
```

output:
![](https://github.com/shibing624/thinking-intervention/blob/main/docs/sample.png)

## 🌈 Intervention Strategy Examples

Here are some common intervention strategies:

### JSON Formatting

```
I need to respond in JSON format and only return the content inside the JSON, without any other information.
```

### Instruction Following Enhancement

```
I need to carefully understand the user's instructions and ensure my response meets all requirements.
1. First, I will analyze all key requirements and constraints in the user's instructions
2. I will ensure my answer meets all these requirements, without missing any key points
3. If there are contradictions or ambiguities in the instructions, I will prioritize the explicitly stated requirements
4. I will not add extra information or functionality not requested by the user
5. I will pay special attention to the format, length, and style requirements in the instructions
```

### Safety Guidance

```
I must ensure my answer is safe, compliant, and beneficial.
1. I will refuse to generate any content that could cause harm, including but not limited to: illegal activities, harmful advice, discriminatory statements, etc.
2. For boundary cases, I will lean towards conservative and safe interpretations
3. If I need to reject a request, I will explain why and try to provide appropriate alternatives
4. I will ensure my answer complies with ethical guidelines and social norms
5. I will not provide information or methods that could be misused or abused
```

## 📊 Use Cases

- **Web Application Integration**: Use the streaming generation API in websites or applications
- **Content Format Control**: Force models to output specific formats (e.g., JSON, Markdown)
- **Safe Content Filtering**: Guide models to generate content that meets safety standards
- **Domain-Specific Optimization**: Optimize model outputs for domains like mathematics, programming, etc.

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