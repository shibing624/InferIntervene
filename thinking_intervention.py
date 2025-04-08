import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger


class ThinkingIntervention:
    """
    实现思维干预技术的类，基于论文《Effectively Controlling Reasoning Models through Thinking Intervention》
    
    思维干预是一种在LLM推理过程中插入或修改思考步骤的方法，用于引导模型的推理过程。
    """

    def __init__(
            self,
            model_name: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            system_prompt: str = "You are a helpful assistant."
    ):
        """
        初始化思维干预类
        
        Args:
            model_name: HuggingFace模型名称
            max_new_tokens: 生成的最大token数
            temperature: 生成的温度
            top_p: 生成的top_p值
            system_prompt: 系统提示
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

        logger.info(f"正在加载模型 {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        # 设置特殊token
        self.thinking_start_token = "<thinking>"
        self.thinking_end_token = "</thinking>"
        self.answer_start_token = "<answer>"
        self.answer_end_token = "</answer>"

        logger.info(
            f"模型和tokenizer加载完成, device: {self.model.device}")

    def intervene(
            self,
            prompt: str,
            intervention_text: str,
            intervention_position: str = "beginning",  # "beginning", "middle", or "custom"
            custom_intervention_point: Optional[int] = None,  # 自定义干预点，表示在生成的文本中插入干预文本的位置
            partial_thinking_length: int = 100  # 中间干预时，第一次生成的长度
    ) -> str:
        """
        对推理过程进行干预
        
        Args:
            prompt: 输入提示
            intervention_text: 干预文本
            intervention_position: 干预位置，可以是"beginning"（开始）,"middle"（中间）或"custom"（自定义）
            custom_intervention_point: 自定义干预点，表示在生成的文本中插入干预文本的位置
            partial_thinking_length: 中间干预时，第一次生成的长度
            
        Returns:
            生成的完整回答
        """
        # 构建包含思考标记和干预的完整提示
        if not prompt.endswith("\n"):
            prompt += "\n"

        # 构建带有思考标记的提示
        thinking_prompt = f"{prompt}{self.thinking_start_token}\n"

        # 基于干预位置插入干预文本
        if intervention_position == "beginning":
            # 在思考开始时进行干预
            thinking_prompt += f"{intervention_text}\n"
            thinking_output = self._generate(thinking_prompt)

        elif intervention_position in ["middle", "custom"]:
            # 生成部分思考内容
            partial_thinking = self._generate(
                thinking_prompt,
                max_new_tokens=partial_thinking_length,
                stop_at_thinking_end=False
            )

            # 如果是自定义干预点，截取到指定位置
            if intervention_position == "custom" and custom_intervention_point is not None:
                # 计算token数量
                tokens = self.tokenizer.encode(partial_thinking)
                if custom_intervention_point < len(tokens):
                    partial_thinking = self.tokenizer.decode(tokens[:custom_intervention_point])

            # 添加干预
            thinking_with_intervention = f"{partial_thinking}\n{intervention_text}\n"

            # 继续生成
            thinking_output = self._generate(thinking_with_intervention)

        else:
            # 默认情况，不进行干预
            thinking_output = self._generate(thinking_prompt)

        # 提取思考部分和答案部分
        thinking_pattern = f"{self.thinking_start_token}(.*?){self.thinking_end_token}"
        thinking_match = re.search(thinking_pattern, thinking_output, re.DOTALL)

        thinking_content = thinking_match.group(1).strip() if thinking_match else ""

        # 继续生成答案
        answer_prompt = f"{prompt}{self.thinking_start_token}\n{thinking_content}\n{self.thinking_end_token}\n{self.answer_start_token}\n"
        full_output = self._generate(answer_prompt)

        return full_output

    def vanilla_generate(self, prompt: str) -> str:
        """不使用思维干预的普通生成"""
        return self._generate(prompt)

    def _generate(
            self,
            prompt: str,
            system_prompt: str = None,
            max_new_tokens: Optional[int] = None,
            stop_at_thinking_end: bool = True
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            system_prompt: 系统提示
            max_new_tokens: 生成的最大token数，如果为None则使用默认值
            stop_at_thinking_end: 是否在思考结束标记处停止生成
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.debug(f"prompt: {text}")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # 进行生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 解码输出
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # 清理输出文本，移除输入部分
        output_text = output_text[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=False)):]

        # 如果需要在思考结束标记处停止，截断输出
        if stop_at_thinking_end and self.thinking_end_token in output_text:
            output_text = output_text.split(self.thinking_end_token)[0] + self.thinking_end_token

        # 返回完整文本
        return prompt + output_text


def main():
    """主函数，用于命令行运行"""
    parser = argparse.ArgumentParser(description="思维干预技术演示")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="使用的HuggingFace模型名称")
    parser.add_argument("--prompt", type=str, default="444+3222=？", help="输入提示")
    parser.add_argument("--system-prompt", type=str, default="你是一个有用的助手,回答格式先思考再回答,思考内容用<thinking>和</thinking>标记,回答内容用<answer>和</answer>标记",
                        help="系统提示")
    parser.add_argument("--intervention", type=str, default="请确保回答准确、客观、全面，如果信息不足应明确说明。",
                        help="干预文本")
    parser.add_argument("--position", type=str, default="beginning",
                        choices=["beginning", "middle", "custom"], help="干预位置")
    parser.add_argument("--intervention-point", type=int, default=None,
                        help="自定义干预点，表示在生成的文本中插入干预文本的位置（token数量）")
    parser.add_argument("--partial-length", type=int, default=500,
                        help="中间干预时，第一次生成的长度")
    parser.add_argument("--compare", action="store_true", help="是否同时运行不干预的版本进行比较")

    args = parser.parse_args()

    # 初始化思维干预
    ti = ThinkingIntervention(
        model_name=args.model,
        system_prompt=args.system_prompt,
    )

    # 运行带干预的生成
    logger.info("正在生成带思维干预的回答...")
    start = time.time()
    output_with_intervention = ti.intervene(
        prompt=args.prompt,
        intervention_text=args.intervention,
        intervention_position=args.position,
        custom_intervention_point=args.intervention_point,
        partial_thinking_length=args.partial_length
    )
    intervention_time = time.time() - start

    print("\n===== 带思维干预的输出 =====")
    print(output_with_intervention)
    print(f"生成时间: {intervention_time:.2f}秒")

    # 如果需要比较，运行不带干预的生成
    if args.compare:
        logger.info("正在生成不带思维干预的回答...")
        start = time.time()
        output_without_intervention = ti.vanilla_generate(args.prompt)
        vanilla_time = time.time() - start

        print("\n===== 不带思维干预的输出 =====")
        print(output_without_intervention)
        print(f"生成时间: {vanilla_time:.2f}秒")


if __name__ == "__main__":
    main()
