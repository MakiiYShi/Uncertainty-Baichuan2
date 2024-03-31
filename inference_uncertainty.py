import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载微调后的模型
from peft import AutoPeftModelForCausalLM
FINETUNE_MODEL_PATH = "output-uncertainty"
tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL_PATH, use_fast=False, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(FINETUNE_MODEL_PATH, trust_remote_code=True)

# 读入标注数据
import pandas as pd
data = pd.read_csv("fine-tune/data/raw.csv")
PROMPT_LONG = '''判断以上文本中是否存在以下不确定性：数字（数值表述更多使用约数，如'几百人'）、概率（非必然事件可能发生，如'可能导致'）、范围（事件只占总体的一定比例，如'主要属于'）、条件（事件发生依赖于某些条件，如'一旦发生'）、框架（主观认为、推测想象的内容居多），答案以列表输出'''
PROMPT_SHORT = "判断以上文本中是否存在数字、概率、范围、条件、框架类别的不确定性，答案以列表输出"

# 处理文本截断
MAX_LEN = 512
def trunc(content) -> str: # 节选文本
    if len(content) < MAX_LEN:
        return content
    trunc_len = MAX_LEN - 100
    text = content[:100] + content[-trunc_len:]
    return text.strip()

# 模型预测
from tqdm import tqdm
output = []
for content in tqdm(data["content"]):
    if pd.isna(content) or len(content) <= 1:
        output.append("[]")
        continue
    messages = [{
        "role": "user",
        "content": trunc(content)
    },{
        "role": "user",
        "content": PROMPT_LONG
    }]
    response = model.chat(tokenizer, messages)
    print(response)
    output.append(response)
# 记录结果
data["predict_uncertainty"] = output
data.to_csv("data/predict_uncertainty.csv")