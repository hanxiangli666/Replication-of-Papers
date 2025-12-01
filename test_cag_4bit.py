import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
from huggingface_hub import login
import os
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. 环境变量配置与登录
# -----------------------------------------------------------------------------
load_dotenv()  # 加载 .env 文件中的环境变量

# 从 .env 文件中安全读取 HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")

# 检查 Key 是否存在 (必须要有 Key 才能登录)
if not HF_TOKEN:
    # 如果找不到 Key，则抛出错误并停止运行
    raise ValueError("❌ 错误：Hugging Face Token (HF_TOKEN) 未找到，请检查 .env 文件是否配置正确！")

# 使用环境变量中的 Token 登录
print("🔄 正在使用环境变量中的 Token 登录 Hugging Face...")
login(token=HF_TOKEN)

# -----------------------------------------------------------------------------
# 2. 配置 4-bit 量化 (适配 8GB 显存)
# -----------------------------------------------------------------------------
print("🔄 正在配置 4-bit 量化...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print(f"⬇️ 正在加载模型: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 解决 Llama 3.1 的 pad token 问题
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✅ 模型加载成功！")

# -----------------------------------------------------------------------------
# 3. CAG 核心演示: 预加载知识 (Prefill)
# -----------------------------------------------------------------------------
context_text = """
[Context Document]
Running Instructions for Cache-Augmented Generation (CAG):
CAG eliminates the need for real-time retrieval by preloading all relevant documents 
into the LLM's extended context and caching its key-value (KV) states. 
During inference, the model uses these preloaded parameters to answer queries 
without additional retrieval steps. This is faster and reduces errors.
"""

print("\n📚 步骤1: 计算 KV Cache (预加载知识)...")
# 确保所有输入都在 GPU 上
input_ids = tokenizer(context_text, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    # 这一步计算出了“知识”的缓存
    outputs = model(input_ids)
    kv_cache = outputs.past_key_values

print("✅ 知识已存入缓存！(后续不再需要处理这段长文本)")

# -----------------------------------------------------------------------------
# 4. CAG 核心演示: 使用缓存进行推理 (Decoding)
# -----------------------------------------------------------------------------
question = "What is the main benefit of CAG?"
print(f"\n❓ 用户提问: {question}")
print("🤖 正在生成回答 (使用缓存)...")

# 准备问题的 token
curr_input_ids = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
generated_token_ids = []

start_time = time.time()

# --- 手动生成循环 (Manual Generation Loop) ---
# 这段代码完全模拟了 LLM 内部的工作原理：
# 输入当前词 + 历史缓存 -> 预测下一个词 -> 更新缓存 -> 循环
with torch.no_grad():
    for _ in range(50): # 生成 50 个词
        # 1. 模型前向传播 (注意：我们传入了 kv_cache)
        outputs = model(
            curr_input_ids, 
            past_key_values=kv_cache, 
            use_cache=True
        )
        
        # 2. 获取预测结果 (Logits)
        next_token_logits = outputs.logits[:, -1, :]
        
        # 3. 贪婪采样 (选概率最大的词)
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # 4. 收集结果
        generated_token_ids.append(next_token_id.item())
        
        # 5. 遇到结束符就停止
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
        # 6. 【关键】更新变量，准备下一轮
        # 现在的输出缓存包含了 (旧缓存 + 新产生的 token 缓存)
        kv_cache = outputs.past_key_values
        # 下一轮的输入只需要是刚才生成的这个新词 (不需要重复输入整个句子)
        curr_input_ids = next_token_id

end_time = time.time()

# 解码生成的文字
response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

print("-" * 30)
print(f"💡 回答:\n{response}")
print("-" * 30)
print(f"⚡ 耗时: {end_time - start_time:.2f} 秒")