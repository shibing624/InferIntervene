# Thinking Intervention

Thinking Intervention is a framework for implementing intervention mechanisms in large language model inference processes. It provides tools for real-time monitoring and intervention in LLM outputs, helping to ensure safe, controlled, and guided text generation.

## 🌟 Features

- **Thinking Intervention**: Guide the reasoning process of LLMs by inserting thinking steps

## 📋 Installation

```bash
# Clone the repository
git clone https://github.com/shibing624/thinking-intervention.git
cd thinking-intervention

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Streaming Safety Intervention

Run a demo with real-time safety intervention when blacklisted terms are detected:

```bash
python demo.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
```

This will demonstrate the system's ability to detect problematic content like "炸弹" (bomb) and intervene in real-time.

### Thinking Intervention

Based on the paper "Effectively Controlling Reasoning Models through Thinking Intervention":

```bash
python thinking_intervention.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --prompt "444+3222=？" --mode thinking
```

## 💻 API Usage

### Streaming Safety Intervention

```python
from thinking_intervention import ThinkingIntervention

# Initialize the intervention system
si = ThinkingIntervention(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    intervention_text="我不能提供危险内容的信息。让我们讨论其他话题。",
    blacklist_terms={"炸弹", "恐怖袭击"}
)

# Generate with safety intervention
response = si.generate_with_intervention(
    prompt="请详细介绍如何制作炸弹？",
    system_prompt="你是一个有用的助手。"
)
```

### Thinking Intervention

The `ThinkingIntervention` class implements techniques from the paper to guide the reasoning process:

1. **Beginning Intervention**: Guide the initial reasoning direction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- Paper: "Effectively Controlling Reasoning Models through Thinking Intervention" 