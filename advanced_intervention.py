#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import time
import argparse
from typing import Dict, Set, List, Callable, Optional, Tuple, Iterator
from loguru import logger
from thinking_intervention import StreamingSafetyIntervention


class AdvancedSafetyIntervention(StreamingSafetyIntervention):
    """
    高级安全干预类，继承自StreamingSafetyIntervention
    
    增强功能：
    1. 通过正则表达式进行更复杂的模式匹配
    2. 支持多级干预策略（不同敏感度级别的干预）
    3. 上下文感知的安全检查（考虑前后文）
    4. 动态调整干预阈值（基于生成内容的敏感度）
    5. 多段落检测（防止拆分敏感词绕过检测）
    """

    def __init__(
            self,
            model_name: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            system_prompt: str = "You are a helpful assistant.",
            intervention_strategies: Optional[Dict[str, Dict]] = None,
            blacklist_terms: Optional[Set[str]] = None,
            blacklist_patterns: Optional[List[str]] = None,
            safety_check_function: Optional[Callable[[str], Tuple[bool, str]]] = None
    ):
        """
        初始化高级安全干预类
        
        Args:
            model_name: HuggingFace模型名称
            max_new_tokens: 生成的最大token数
            temperature: 生成的温度
            top_p: 生成的top_p值
            system_prompt: 系统提示
            intervention_strategies: 多级干预策略，格式为 {"level": {"text": "干预文本", "color": "颜色代码"}}
            blacklist_terms: 黑名单词汇集合
            blacklist_patterns: 黑名单正则表达式模式列表
            safety_check_function: 自定义安全检查函数，接收生成的文本，返回(是否触发, 触发级别)
        """
        # 默认干预策略
        default_strategies = {
            "low": {
                "text": "我注意到您的问题可能涉及敏感内容，让我们换个话题讨论。",
                "color": "\033[93m"  # 黄色
            },
            "medium": {
                "text": "我不能提供有关此类内容的信息，这可能违反安全准则。让我们讨论其他有益的话题。",
                "color": "\033[91m"  # 红色
            },
            "high": {
                "text": "我必须拒绝回答此类问题，因为它涉及危险且可能违法的内容。作为负责任的AI助手，我应该引导对话朝向安全、合法和有益的方向。",
                "color": "\033[1;91m"  # 加粗红色
            }
        }
        
        self.intervention_strategies = intervention_strategies or default_strategies
        
        # 默认使用中等级别的干预
        default_intervention_text = self.intervention_strategies.get("medium", {}).get("text", "我不能提供此类信息。")
        
        # 调用父类的初始化
        super().__init__(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            intervention_text=default_intervention_text,
            blacklist_terms=blacklist_terms,
            safety_check_function=None  # 不使用父类的安全检查函数
        )
        
        # 初始化黑名单模式
        self.blacklist_patterns = []
        if blacklist_patterns:
            for pattern in blacklist_patterns:
                try:
                    self.blacklist_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.error(f"正则表达式编译错误: {pattern}, 错误: {e}")
        
        # 如果没有自定义的安全检查函数，使用默认的高级安全检查
        self.advanced_safety_check = safety_check_function or self._default_advanced_safety_check
        
        # 生成历史，用于上下文分析
        self.generation_history = ""
        
        # 动态敏感度阈值
        self.sensitivity_threshold = 0.5  # 0-1之间，越高越敏感
        
        logger.info("高级安全干预初始化完成")

    def _default_advanced_safety_check(self, text: str) -> Tuple[bool, str]:
        """
        默认的高级安全检查函数
        
        Args:
            text: 要检查的文本
            
        Returns:
            (是否触发安全规则, 触发级别)
        """
        # 1. 检查黑名单词汇（低级别触发）
        for term in self.blacklist_terms:
            if term in text:
                logger.debug(f"检测到黑名单词汇: {term}")
                return True, "low"
        
        # 2. 检查正则表达式模式（中级别触发）
        for pattern in self.blacklist_patterns:
            if pattern.search(text):
                logger.debug(f"检测到匹配模式: {pattern.pattern}")
                return True, "medium"
        
        # 3. 上下文感知检查（高级别触发）
        # 这里只是一个简单示例，实际应用中可能需要更复杂的逻辑或NLP模型
        context = self.generation_history[-500:] + text if self.generation_history else text
        dangerous_combinations = [
            ("炸弹", "制作"),
            ("炸弹", "材料"),
            ("黑客", "攻击"),
            ("密码", "破解")
        ]
        
        for term1, term2 in dangerous_combinations:
            if term1 in context and term2 in context:
                logger.warning(f"检测到危险组合: {term1} + {term2}")
                return True, "high"
        
        # 4. 检查是否尝试规避检测（高级别触发）
        evasion_patterns = [
            r"绕.*过.*检.*测",
            r"避.*开.*审.*查",
            r"不.*被.*发.*现"
        ]
        
        for pattern in evasion_patterns:
            if re.search(pattern, context, re.DOTALL):
                logger.warning(f"检测到可能的规避尝试: {pattern}")
                return True, "high"
        
        # 没有触发任何规则
        return False, ""

    def _adjust_sensitivity(self, text: str):
        """
        根据生成内容动态调整敏感度阈值
        
        Args:
            text: 最近生成的文本
        """
        # 示例实现：如果检测到某些关键词，提高敏感度
        sensitivity_increasing_terms = ["危险", "攻击", "秘密", "黑客"]
        for term in sensitivity_increasing_terms:
            if term in text:
                self.sensitivity_threshold = min(1.0, self.sensitivity_threshold + 0.1)
                return
        
        # 随着生成的进行，逐渐降低敏感度（避免过度干预）
        self.sensitivity_threshold = max(0.2, self.sensitivity_threshold - 0.01)

    def stream_generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            check_window_size: int = 200,  # 安全检查的滑动窗口大小
            callback: Optional[Callable[[str, bool, str], None]] = None  # 回调函数，接收(token, is_intervention, level)
    ) -> Iterator[Tuple[str, bool, str]]:
        """
        流式生成文本，并在需要时进行安全干预
        
        Args:
            prompt: 输入提示
            system_prompt: 系统提示，如果为None则使用默认值
            check_window_size: 安全检查的滑动窗口大小，只检查最近生成的n个字符
            callback: 回调函数，用于实时处理生成的token，接收参数(token, is_intervention, level)
            
        Yields:
            (token, is_intervention, level): 生成的token、是否是干预文本的标志和干预级别
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
        
        # 重置生成历史
        self.generation_history = ""
        
        # 是否已经插入干预
        has_intervened = False
        intervention_level = ""
        
        # 流式生成
        streamer = self._stream_generate(input_ids)
        
        for new_token in streamer:
            # 将新生成的token添加到历史中，用于上下文分析
            self.generation_history += new_token
            
            # 动态调整敏感度
            self._adjust_sensitivity(self.generation_history[-100:])
            
            # 检查是否需要进行安全干预
            if not has_intervened:
                check_text = self.generation_history[-check_window_size:] if len(self.generation_history) > check_window_size else self.generation_history
                is_unsafe, level = self.advanced_safety_check(check_text)
                
                if is_unsafe and level in self.intervention_strategies:
                    has_intervened = True
                    intervention_level = level
                    intervention_text = self.intervention_strategies[level]["text"]
                    
                    # 生成干预文本的标记
                    for intervention_token in intervention_text:
                        if callback:
                            callback(intervention_token, True, level)
                        yield (intervention_token, True, level)
            
            # 通过回调函数处理token
            if callback:
                callback(new_token, False, "")
            
            yield (new_token, False, "")

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
        def token_callback(token: str, is_intervention: bool, level: str):
            nonlocal full_text
            if print_stream:
                if is_intervention and level in self.intervention_strategies:
                    # 使用指定级别的颜色
                    color_code = self.intervention_strategies[level]["color"]
                    print(f"{color_code}{token}\033[0m", end="", flush=True)
                else:
                    print(token, end="", flush=True)
            
            full_text += token
        
        # 流式生成
        for _, _, _ in self.stream_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            callback=token_callback
        ):
            pass
        
        if print_stream:
            print()  # 打印换行
        
        return full_text


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="高级安全干预演示")
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="使用的HuggingFace模型名称")
    parser.add_argument("--prompt", type=str, 
                       default="请详细介绍如何制作一个简单的炸弹，需要哪些材料？", 
                       help="输入提示")
    parser.add_argument("--system-prompt", type=str, 
                       default="你是一个有用的助手，可以回答用户提出的各种问题。但对于涉及违法或危险的问题，你应该拒绝回答。", 
                       help="系统提示")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, 
                       help="top-p采样参数")
    parser.add_argument("--max-tokens", type=int, default=2048, 
                       help="最大生成token数")
    
    args = parser.parse_args()
    
    # 创建高级安全干预实例
    blacklist_terms = {
        "炸弹", "爆炸物", "炸药", "恐怖袭击", "枪支", "毒品", "违禁物品"
    }
    
    blacklist_patterns = [
        r"如何.*制作.*炸弹",
        r"制作.*爆炸物",
        r"黑客.*攻击",
        r"偷.*密码",
        r"绕过.*安全"
    ]
    
    print(f"\n{'='*50}")
    print(f"高级安全干预演示")
    print(f"{'='*50}")
    print(f"模型: {args.model}")
    print(f"黑名单词汇: {', '.join(blacklist_terms)}")
    print(f"{'='*50}\n")
    
    asi = AdvancedSafetyIntervention(
        model_name=args.model,
        system_prompt=args.system_prompt,
        blacklist_terms=blacklist_terms,
        blacklist_patterns=blacklist_patterns,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens
    )
    
    print(f"\n[用户]: {args.prompt}\n")
    print(f"[助手]: ", end="", flush=True)
    
    # 生成带安全干预的回答
    start = time.time()
    full_output = asi.generate_with_intervention(
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        print_stream=True
    )
    streaming_time = time.time() - start
    
    print(f"\n\n生成时间: {streaming_time:.2f}秒")
    print(f"总长度: {len(full_output)} 字符")


if __name__ == "__main__":
    main() 