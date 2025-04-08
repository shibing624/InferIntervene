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
        device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        初始化思维干预类
        
        Args:
            model_name: HuggingFace模型名称
            device: 运行设备
            max_new_tokens: 生成的最大token数
            temperature: 生成的温度
            top_p: 生成的top_p值
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        logger.info(f"正在加载模型 {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        # 设置特殊token
        self.thinking_start_token = "<thinking>"
        self.thinking_end_token = "</thinking>"
        self.answer_start_token = "<answer>"
        self.answer_end_token = "</answer>"
        
        # 如果不存在这些特殊token，可以添加到tokenizer中，但注意这会改变模型权重的对应关系
        # 此处简单处理，将它们作为普通文本处理
        logger.info("模型和tokenizer加载完成")
    
    def intervene(
        self, 
        prompt: str, 
        intervention_text: str,
        intervention_position: str = "beginning"  # "beginning", "middle", or "custom"
    ) -> str:
        """
        对推理过程进行干预
        
        Args:
            prompt: 输入提示
            intervention_text: 干预文本
            intervention_position: 干预位置，可以是"beginning"（开始）,"middle"（中间）或"custom"（自定义）
            
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
        
        # 生成包含干预的思考过程
        thinking_output = self._generate(thinking_prompt)
        
        # 如果需要在中间进行干预，这里需要额外的逻辑
        if intervention_position == "middle":
            # 生成部分思考内容
            partial_thinking = self._generate(thinking_prompt, max_new_tokens=100, stop_at_thinking_end=False)
            
            # 添加干预
            thinking_with_intervention = f"{partial_thinking}\n{intervention_text}\n"
            
            # 继续生成
            thinking_output = self._generate(thinking_with_intervention)
        
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
        max_new_tokens: Optional[int] = None,
        stop_at_thinking_end: bool = True
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 生成的最大token数，如果为None则使用默认值
            stop_at_thinking_end: 是否在思考结束标记处停止生成
            
        Returns:
            生成的文本
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        # 对输入进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
    parser.add_argument("--prompt", type=str, default="中国的首都是什么？")
    parser.add_argument("--intervention", type=str, default="请确保回答准确、客观、全面，如果信息不足应明确说明。", 
                       help="干预文本")
    parser.add_argument("--position", type=str, default="beginning", 
                       choices=["beginning", "middle"], help="干预位置")
    parser.add_argument("--compare", action="store_true", help="是否同时运行不干预的版本进行比较")
    parser.add_argument("--device", type=str, default="mps", help="运行设备")
    
    args = parser.parse_args()
    
    # 初始化思维干预
    ti = ThinkingIntervention(
        model_name=args.model,
        device=args.device
    )
    
    # 运行带干预的生成
    logger.info("正在生成带思维干预的回答...")
    start = time.time()
    output_with_intervention = ti.intervene(
        prompt=args.prompt,
        intervention_text=args.intervention,
        intervention_position=args.position
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