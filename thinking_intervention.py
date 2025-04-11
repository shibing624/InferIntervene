import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable, Iterator
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
        answer_prompt = f"{prompt}{self.thinking_start_token}\n{thinking_content}\n{self.thinking_end_token}\n"
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


class StreamingSafetyIntervention:
    """
    流式安全干预类。
    
    该类提供在流式生成过程中，对出现在黑名单中的内容进行实时干预的功能。
    可以逐个token进行生成，并在检测到敏感内容时立即插入干预文本。
    """

    def __init__(
            self,
            model_name: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            system_prompt: str = "You are a helpful assistant.",
            intervention_text: str = "我不能提供有关危险或非法内容的信息。让我们讨论其他有益的话题。",
            blacklist_terms: Optional[Set[str]] = None,
            safety_check_function: Optional[Callable[[str], bool]] = None
    ):
        """
        初始化流式安全干预类
        
        Args:
            model_name: HuggingFace模型名称
            max_new_tokens: 生成的最大token数
            temperature: 生成的温度
            top_p: 生成的top_p值
            system_prompt: 系统提示
            intervention_text: 检测到敏感内容时插入的干预文本
            blacklist_terms: 黑名单词汇集合，例如 {"炸弹", "恐怖袭击"}
            safety_check_function: 自定义安全检查函数，接收生成的文本，返回是否触发安全规则
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.intervention_text = intervention_text

        # 初始化黑名单
        self.blacklist_terms = blacklist_terms or {"炸弹", "恐怖袭击", "毒品", "违禁物品", "炸药"}

        # 设置安全检查函数
        self.safety_check_function = safety_check_function or self._default_safety_check

        logger.info(f"正在加载模型 {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        logger.debug(f"模型和tokenizer加载完成, device: {self.model.device}")

    def _default_safety_check(self, text: str) -> bool:
        """
        默认的安全检查函数，检查文本是否包含黑名单词汇
        
        Args:
            text: 要检查的文本
            
        Returns:
            是否触发安全规则
        """
        for term in self.blacklist_terms:
            if term in text:
                logger.warning(f"检测到黑名单词汇: {term}")
                return True
        return False

    def stream_generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            check_window_size: int = 100,  # 安全检查的滑动窗口大小
            callback: Optional[Callable[[str, bool], None]] = None  # 回调函数，用于实时处理生成的token
    ) -> Iterator[Tuple[str, bool]]:
        """
        流式生成文本，并在需要时进行安全干预
        
        Args:
            prompt: 输入提示
            system_prompt: 系统提示，如果为None则使用默认值
            check_window_size: 安全检查的滑动窗口大小，只检查最近生成的n个字符
            callback: 回调函数，用于实时处理生成的token，接收参数(token, is_intervention)
            
        Yields:
            (token, is_intervention): 生成的token和是否是干预文本的标志
        """
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # 存储生成的文本，用于安全检查
        generated_text = ""

        # 是否已经插入干预
        has_intervened = False

        # 流式生成
        streamer = self._stream_generate(input_ids)

        for new_token in streamer:
            # 检查是否需要进行安全干预
            if not has_intervened:
                check_text = generated_text[-check_window_size:] + new_token if generated_text else new_token
                if self.safety_check_function(check_text):
                    has_intervened = True

                    # 生成干预文本的标记
                    for intervention_token in self.intervention_text:
                        if callback:
                            callback(intervention_token, True)
                        yield (intervention_token, True)

            # 将新生成的token添加到文本中
            generated_text += new_token

            # 通过回调函数处理token
            if callback:
                callback(new_token, False)

            yield (new_token, False)

    def _stream_generate(self, input_ids: torch.Tensor) -> Iterator[str]:
        """
        使用huggingface模型进行流式生成
        
        Args:
            input_ids: 输入的token ids
            
        Yields:
            生成的token
        """
        # 使用模型的生成API进行流式生成
        with torch.no_grad():
            # 设置初始past_key_values为None
            past = None

            # 当前生成的token数
            gen_tokens_count = 0

            # 使用当前输入生成第一个token
            cur_input_ids = input_ids

            while gen_tokens_count < self.max_new_tokens:
                # 模型前向传播
                outputs = self.model(
                    input_ids=cur_input_ids,
                    past_key_values=past,
                    use_cache=True
                )

                # 获取下一个token的logits和past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                past = outputs.past_key_values

                # 应用温度和top_p采样
                if self.temperature > 0:
                    next_token_logits = next_token_logits / self.temperature

                # 应用top_p采样
                if 0 < self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除概率较低的tokens
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # 如果生成了EOS token，结束生成
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # 解码token
                next_token_text = self.tokenizer.decode(next_token[0])
                yield next_token_text

                # 更新当前输入为下一个token
                cur_input_ids = next_token
                gen_tokens_count += 1

    def generate_with_intervention(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            print_stream: bool = True
    ) -> str:
        """
        带安全干预的生成，并返回完整结果
        
        Args:
            prompt: 输入提示
            system_prompt: 系统提示
            print_stream: 是否在控制台打印流式输出
            
        Returns:
            生成的完整文本
        """
        full_text = ""

        # 定义回调函数
        def token_callback(token: str, is_intervention: bool):
            nonlocal full_text
            if print_stream:
                # 使用不同颜色打印干预文本
                if is_intervention:
                    print(f"\033[91m{token}\033[0m", end="", flush=True)  # 红色
                else:
                    print(token, end="", flush=True)

            full_text += token

        # 流式生成
        for _, _ in self.stream_generate(
                prompt=prompt,
                system_prompt=system_prompt,
                callback=token_callback
        ):
            pass

        if print_stream:
            print()  # 打印换行

        return full_text


def main():
    """主函数，用于命令行运行"""
    parser = argparse.ArgumentParser(description="思维干预和安全干预技术演示")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="使用的HuggingFace模型名称")
    parser.add_argument("--prompt", type=str, default="化学反应中有哪些危险物质，可以用来做炸药", help="输入提示")
    parser.add_argument("--system-prompt", type=str, default="你是一个有用的助手。", help="系统提示")
    parser.add_argument("--intervention", type=str, default="请确保回答准确、客观、全面，如果信息不足应明确说明。",
                        help="干预文本")
    parser.add_argument("--mode", type=str, default="thinking", choices=["thinking", "streaming"], help="干预模式")
    parser.add_argument("--blacklist", type=str, nargs="+", default=["炸弹", "恐怖袭击"], help="黑名单词汇")

    # 思维干预相关参数
    parser.add_argument("--position", type=str, default="beginning", choices=["beginning", "middle", "custom"],
                        help="干预位置")
    parser.add_argument("--intervention-point", type=int, default=None,
                        help="自定义干预点，表示在生成的文本中插入干预文本的位置（token数量）")
    parser.add_argument("--partial-length", type=int, default=500, help="中间干预时，第一次生成的长度")
    parser.add_argument("--compare", action="store_true", help="是否同时运行不干预的版本进行比较")

    args = parser.parse_args()

    if args.mode == "thinking":
        # 使用思维干预模式
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

    else:
        # 使用流式安全干预模式
        si = StreamingSafetyIntervention(
            model_name=args.model,
            system_prompt=args.system_prompt,
            intervention_text=args.intervention,
            blacklist_terms=set(args.blacklist)
        )

        logger.info("开始流式生成（带安全干预）...")
        print("\n===== 流式输出（带安全干预）=====")
        start = time.time()
        full_output = si.generate_with_intervention(
            prompt=args.prompt,
            system_prompt=args.system_prompt
        )
        streaming_time = time.time() - start
        print(f"生成时间: {streaming_time:.2f}秒")
        # 运行不带干预的生成
        if args.compare:
            logger.info("开始流式生成（不带安全干预）...")
            print("\n===== 流式输出（不带安全干预）=====")

            # 创建一个不进行安全检查的生成器
            si_no_intervention = StreamingSafetyIntervention(
                model_name=args.model,
                system_prompt=args.system_prompt,
                # 使用一个总是返回False的安全检查函数，即不进行干预
                safety_check_function=lambda _: False
            )

            start = time.time()
            vanilla_output = si_no_intervention.generate_with_intervention(
                prompt=args.prompt,
                system_prompt=args.system_prompt
            )
            vanilla_time = time.time() - start
            print(f"生成时间: {vanilla_time:.2f}秒")


if __name__ == "__main__":
    main()
