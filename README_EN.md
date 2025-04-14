# Thinking Intervention

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

## 📝 Notes

- This project is primarily for research and experimental purposes
- The effectiveness of intervention text may vary depending on the model
- Smaller models may not respond to intervention text as effectively as larger models

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 