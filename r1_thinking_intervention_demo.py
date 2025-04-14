# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 最简化实现思维干预技术的代码
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import datetime
from threading import Thread
from transformers import TextIteratorStreamer
from loguru import logger

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "一句话介绍北京."
messages = [
    {"role": "system", "content": "中文回答"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
logger.info(text)
intervention_text = "我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。"
text = text + intervention_text
logger.info(f"new text: {text}")
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


def generate_streaming_text(prompt):
    messages = [
        {"role": "system", "content": "中文回答"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    logger.info(text)
    intervention_text = "我需要用json格式回答，并且只返回json里面的内容，不要返回任何其他信息。"
    text = text + intervention_text
    logger.info(f"new text: {text}")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": new_text,  # 只发送新增的内容
            "status": 200,
            "time": time
        }
        yield json.dumps(answer, ensure_ascii=False).encode() + b'\n'
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    yield json.dumps({"response": generated_text, "status": 200, "time": time}, ensure_ascii=False).encode() + b'\n'


r = generate_streaming_text("一句话介绍北京")
for chunk in r:
    print(chunk.decode())
