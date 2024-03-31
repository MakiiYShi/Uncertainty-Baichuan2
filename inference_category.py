import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载微调后的模型
from peft import AutoPeftModelForCausalLM
FINETUNE_MODEL_PATH = "./output-class"
tokenizer = AutoTokenizer.from_pretrained(FINETUNE_MODEL_PATH, use_fast=False, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(FINETUNE_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True)

# 读入标注数据
import pandas as pd
raw = pd.read_csv("data/timeline_data.csv")
data = raw.sample(n=50)
CLASSIFICATION = ["不相关", "科学文本", "科普介绍", "个人观点", "技术想象", "虚构写作"]
PROMPT_LONG = f'''根据问答的标题和内容，判断以上文本主要属于以下哪一类（只输出类名）：
“不相关” -> 与技术的科学传播不相关的内容；
”科学文本“ -> 引用科学文献或相关新闻报道，包含较多专业术语；
“科普介绍” -> 以通俗化的语言对相关科学原理、结构、应用等作解释说明；
”个人观点“ -> 对所提出问题的个人看法和观点的陈述，几乎不包含具体科学原理；
“技术想象” -> 对技术的未来应用和风险相关的探讨，基于推测和想象而并非基于事实；
”虚构写作“ -> 虚构的科幻小说或故事。'''
PROMPT_SHORT = f'''根据以上问答，判断文本属于哪一类：不相关、科学文本、科普介绍、个人观点、技术想象、虚构写作'''

# 处理文本截断
MAX_LEN = 510 - len(PROMPT_SHORT)
def trunc(title, content) -> str:
    text = f"《{title}》\n"
    if len(text + content) < MAX_LEN:
        return text + content
    trunc_len = MAX_LEN - len(text) - 100
    text += content[:100] + content[-trunc_len:]
    return text.strip()

# 模型预测
from tqdm import tqdm
output = []
for title, content in tqdm(zip(data["title"], data["content"])):
    if pd.isna(title) or pd.isna(content) or len(title) <= 1 or len(title) <= 1:
        output.append("不相关")
        continue
    messages = [{
        "role": "user",
        "content": trunc(title, content) + "\n\n" + PROMPT_SHORT
    }]
    response = model.chat(tokenizer, messages)
    # print(response)
    output.append(response)
# 记录结果
data["predict_type"] = output
data.to_csv("data/predict_step1.csv")