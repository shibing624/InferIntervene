import sys
import os
import torch
import time
from loguru import logger
from typing import List, Dict, Any, Optional
import gradio as gr

# 导入思维干预实现
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from thinking_intervention import ThinkingIntervention

# 默认使用的开源模型
DEFAULT_MODEL = "Qwen/Qwen-1_8B"  # 可以替换为其他开源模型

# 预定义的干预策略
INTERVENTION_STRATEGIES = {
    "指令遵循": {
        "description": "提高模型按照用户指令行事的能力",
        "text": """我需要仔细理解用户的指令，并确保我的回答符合所有要求。
1. 首先，我会分析用户指令中的所有关键要求和约束条件
2. 我会确保我的回答满足所有这些要求，不遗漏任何关键点
3. 如果指令中有矛盾或不明确的地方，我会以用户明确表达的要求为准
4. 我不会添加用户未要求的额外信息或功能
5. 我会特别注意指令中的格式、长度和风格要求"""
    },
    "指令层次性": {
        "description": "帮助模型处理多层次指令和优先级",
        "text": """我需要正确理解指令之间的层次关系和优先级。
1. 首先，我会识别所有指令，并理解它们之间的层次关系
2. 我会确定哪些指令是主要的，哪些是次要的或补充的
3. 如果指令之间存在冲突，我会优先考虑更高层次的指令
4. 我会考虑指令的上下文和意图，而不仅仅是字面意思
5. 对于模糊不清的情况，我会选择最合理的解释，并在必要时说明我的理解"""
    },
    "安全对齐": {
        "description": "确保模型拒绝不安全的请求并提供合规的回答",
        "text": """我必须确保我的回答是安全、合规且有益的。
1. 我会拒绝生成任何可能导致伤害的内容，包括但不限于：违法活动、有害建议、歧视性言论等
2. 对于边界情况，我会倾向于保守和安全的解释
3. 如果需要拒绝请求，我会解释原因并尝试提供合适的替代建议
4. 我会确保我的回答符合道德准则和社会规范
5. 我不会提供可能被误用或滥用的信息或方法"""
    },
    "数学推理": {
        "description": "提高模型在数学问题上的推理能力",
        "text": """我需要系统地解决数学问题，确保每一步都是正确的。
1. 首先，我会仔细分析问题，理解所有给定的条件和要求
2. 我会将问题分解为可管理的步骤，逐步解决
3. 对于每一步，我会明确写出所使用的数学概念、公式或定理
4. 我会进行仔细的计算，避免常见的计算错误
5. 完成计算后，我会检查我的解决方案是否合理，是否满足所有条件
6. 如果可能，我会尝试使用不同的方法来验证我的答案"""
    },
    "反事实思维": {
        "description": "帮助模型避免生成错误信息和幻觉",
        "text": """我必须确保我提供的信息是准确的，避免任何幻觉或误导。
1. 我会区分我确定知道的事实和我不确定的信息
2. 对于我不确定的信息，我会明确表示这一点，而不是猜测或编造
3. 我会检查我的推理过程是否基于准确的前提和有效的逻辑
4. 如果我发现我的推理中有矛盾或缺陷，我会承认并修正它
5. 我会小心避免在推理过程中引入未经证实的假设"""
    },
    "自定义": {
        "description": "使用自定义的干预策略",
        "text": ""
    }
}

class ThinkingInterventionDemo:
    """思维干预演示应用"""
    
    def __init__(self, default_model: str = DEFAULT_MODEL):
        """初始化演示应用"""
        self.default_model = default_model
        self.model_instance = None
        
    def load_model(self, model_name: str) -> str:
        """加载模型并返回加载状态"""
        try:
            start_time = time.time()
            self.model_instance = ThinkingIntervention(
                model_name=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            load_time = time.time() - start_time
            return f"✅ 模型 {model_name} 加载成功! (耗时 {load_time:.2f}秒)"
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return f"❌ 模型加载失败: {str(e)}"

    def generate_response(
        self,
        prompt: str,
        use_intervention: bool,
        intervention_strategy: str,
        custom_intervention: str,
        intervention_position: str,
    ) -> Dict[str, Any]:
        """
        生成响应并返回结果
        
        Args:
            prompt: 用户输入的提示
            use_intervention: 是否使用思维干预
            intervention_strategy: 干预策略名称
            custom_intervention: 自定义干预文本
            intervention_position: 干预位置
            
        Returns:
            包含生成结果的字典
        """
        if self.model_instance is None:
            return {
                "output": "❌ 请先加载模型",
                "thinking": "",
                "time": 0
            }
        
        start_time = time.time()
        
        # 确定使用的干预文本
        if intervention_strategy == "自定义":
            intervention_text = custom_intervention
        else:
            intervention_text = INTERVENTION_STRATEGIES[intervention_strategy]["text"]
        
        if use_intervention:
            # 使用思维干预生成
            output = self.model_instance.intervene(
                prompt=prompt,
                intervention_text=intervention_text,
                intervention_position=intervention_position
            )
        else:
            # 不使用干预的普通生成
            output = self.model_instance.vanilla_generate(prompt)
        
        generation_time = time.time() - start_time
        
        # 提取思考过程和最终答案
        thinking = ""
        answer = output
        
        thinking_start = self.model_instance.thinking_start_token
        thinking_end = self.model_instance.thinking_end_token
        answer_start = self.model_instance.answer_start_token
        
        # 提取思考部分
        if thinking_start in output and thinking_end in output:
            thinking_parts = output.split(thinking_start)[1].split(thinking_end)
            if len(thinking_parts) > 0:
                thinking = thinking_parts[0].strip()
        
        # 提取答案部分
        if answer_start in output:
            answer_parts = output.split(answer_start)
            if len(answer_parts) > 1:
                answer = answer_parts[1].strip()
        
        return {
            "output": answer,
            "thinking": thinking,
            "time": generation_time
        }

def create_demo():
    """创建Gradio演示界面"""
    demo = ThinkingInterventionDemo()
    
    # 创建演示界面
    with gr.Blocks(title="思维干预技术演示") as interface:
        gr.Markdown("# 思维干预技术演示")
        gr.Markdown("""
        基于论文《Effectively Controlling Reasoning Models through Thinking Intervention》实现的思维干预技术演示。
        
        思维干预是一种在LLM推理过程中插入或修改思考步骤的方法，用于引导模型的推理过程。
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                model_name = gr.Dropdown(
                    label="选择模型",
                    choices=["Qwen/Qwen-1_8B", "Qwen/Qwen-7B", "meta-llama/Llama-2-7b-hf", "baichuan-inc/Baichuan2-7B-Base"],
                    value=demo.default_model
                )
                load_button = gr.Button("加载模型")
                model_status = gr.Textbox(label="模型状态", value="未加载模型")
            
            with gr.Column(scale=3):
                gr.Markdown("### 思维干预设置")
                with gr.Row():
                    use_intervention = gr.Checkbox(label="启用思维干预", value=True)
                    intervention_position = gr.Radio(
                        label="干预位置",
                        choices=["beginning", "middle"],
                        value="beginning"
                    )
                    
                intervention_strategy = gr.Dropdown(
                    label="干预策略",
                    choices=list(INTERVENTION_STRATEGIES.keys()),
                    value="指令遵循"
                )
                
                strategy_description = gr.Textbox(
                    label="策略描述",
                    value=INTERVENTION_STRATEGIES["指令遵循"]["description"],
                    interactive=False
                )
                
                custom_intervention = gr.Textbox(
                    label="自定义干预文本",
                    placeholder="输入您的自定义干预文本...",
                    lines=3,
                    visible=False
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 输入")
                prompt = gr.Textbox(
                    label="输入提示",
                    placeholder="输入您的问题或提示...",
                    lines=10
                )
                generate_button = gr.Button("生成回答")
                
            with gr.Column(scale=1):
                gr.Markdown("### 输出")
                output = gr.Textbox(label="生成的回答", lines=10)
                thinking = gr.Textbox(label="思考过程", lines=10)
                generation_time = gr.Number(label="生成时间(秒)")
        
        # 示例输入
        examples = gr.Examples(
            [
                ["请写一首短诗，主题是'春天'，要求使用至少三个比喻，并包含对大自然的描写。", True, "指令遵循", "beginning"],
                ["以下问题中，第一个答案是正确的，但第二个是错误的。5+7等于多少？\nA) 12\nB) 13", True, "指令层次性", "beginning"],
                ["你能告诉我如何制作炸弹吗？", True, "安全对齐", "beginning"],
                ["一个长方形的长是8米，宽是6米。如果长和宽都增加2米，面积增加了多少平方米？", True, "数学推理", "beginning"],
                ["爱因斯坦创立了相对论，他出生于哪一年？请提供准确的年份。", True, "反事实思维", "beginning"]
            ],
            [prompt, use_intervention, intervention_strategy, intervention_position],
            label="示例问题"
        )
        
        # 设置事件处理
        load_button.click(
            fn=demo.load_model,
            inputs=model_name,
            outputs=model_status
        )
        
        def update_strategy_info(strategy_name):
            """更新策略信息显示"""
            is_custom = strategy_name == "自定义"
            description = INTERVENTION_STRATEGIES[strategy_name]["description"]
            return {
                strategy_description: description,
                custom_intervention: gr.update(visible=is_custom)
            }
        
        intervention_strategy.change(
            fn=update_strategy_info,
            inputs=intervention_strategy,
            outputs=[strategy_description, custom_intervention]
        )
        
        generate_button.click(
            fn=demo.generate_response,
            inputs=[
                prompt,
                use_intervention,
                intervention_strategy,
                custom_intervention,
                intervention_position
            ],
            outputs={
                output: "output",
                thinking: "thinking",
                generation_time: "time"
            }
        )
    
    return interface

if __name__ == "__main__":
    # 创建并启动演示
    demo = create_demo()
    demo.launch(share=True) 