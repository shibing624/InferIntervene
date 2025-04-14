# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 思维干预完整实现
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import argparse
import datetime
import threading
import json
from typing import Iterator
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
        self.thinking_start_token = "<think>"
        self.thinking_end_token = "</think>"
        logger.debug(f"模型和tokenizer加载完成, device: {self.model.device}")

    def intervene_generate(
            self,
            prompt: str,
            intervention_text: str = "",
            system_prompt: str = None,
    ) -> str:
        """
        直接在提示后附加干预文本，不使用思考标记
        
        Args:
            prompt: 输入提示
            intervention_text: 干预文本
            system_prompt: 系统提示
            
        Returns:
            生成的完整回答
        """
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 直接在提示后附加干预文本
        full_prompt = f"{text} {intervention_text}"
        logger.debug(f"intervention prompt: {full_prompt}")
        model_inputs = self.tokenizer([full_prompt], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = [
            output_ids[i][len(input_ids):] for i, input_ids in enumerate(model_inputs.input_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def stream_intervene_generate(
            self,
            prompt: str,
            intervention_text: str = "",
            system_prompt: str = None,
    ) -> Iterator[bytes]:
        """
        直接在提示后附加干预文本进行流式生成
        
        Args:
            prompt: 输入提示
            intervention_text: 干预文本
            system_prompt: 系统提示

        Yields:
            生成的文本片段，以字节流的形式返回
        """
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 直接在提示后附加干预文本
        full_prompt = f"{text} {intervention_text}"
        logger.debug(f"intervention prompt: {full_prompt}")
        model_inputs = self.tokenizer([full_prompt], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            response = {
                "content": new_text,  # 只发送新增的内容
                "status": 200,
                "time": time_str
            }
            yield json.dumps(response, ensure_ascii=False).encode() + b'\n'
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = {
            "content": generated_text,  # 最后发送完整的内容
            "status": 200,
            "time": time_str
        }
        yield json.dumps(response, ensure_ascii=False).encode()


def main():
    """主函数，用于命令行运行"""
    parser = argparse.ArgumentParser(description="思维干预技术演示")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="使用的HuggingFace模型名称")
    parser.add_argument("--prompt", type=str, default="一句话介绍北京",
                        help="输入提示")
    parser.add_argument("--system-prompt", type=str, default="你是一个有用的助手。中文回答",
                        help="系统提示")
    parser.add_argument("--intervention", type=str,
                        default="我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。",
                        help="干预文本")

    args = parser.parse_args()
    ti = ThinkingIntervention(
        model_name=args.model,
        system_prompt=args.system_prompt,
    )

    print("\n===== 直接干预（非流式）=====")
    # 使用直接干预生成回答
    response = ti.intervene_generate(
        prompt=args.prompt,
        intervention_text=args.intervention,
        system_prompt=args.system_prompt,
    )
    print(response)

    print("\n===== 流式生成（插入干预）=====")
    for chunk in ti.stream_intervene_generate(
            prompt=args.prompt,
            intervention_text=args.intervention,
            system_prompt=args.system_prompt,
    ):
        print(chunk.decode())


if __name__ == "__main__":
    main()
