## 累积解释方差
import matplotlib.pyplot as plt
import argparse
import logging
from typing import Tuple
import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from peft import PeftModel
from accelerate import PartialState, Accelerator
from accelerate.utils import set_seed, gather_object

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from functools import reduce
import datetime


layer_num = 16
base_model_path = "/mnt/luoyingfeng/model_card/Meta-Llama-3.1-8B"
fft_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/fft_1e-5_2epoch/de-en/checkpoint-200"
fft_model = AutoModelForCausalLM.from_pretrained(fft_model_path, trust_remote_code=True, torch_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16)

fft_check_layer = fft_model.model.layers[layer_num]
base_check_layer = base_model.model.layers[layer_num]


print(f"fft weights: {fft_check_layer.self_attn.q_proj.weight.detach().numpy().shape}")
print(f"base weights: {base_check_layer.self_attn.q_proj.weight.detach().numpy().shape}")



# 求出full-sft与base模型之间的参数变化
change_full = fft_check_layer.self_attn.q_proj.weight.detach().numpy() - base_check_layer.self_attn.q_proj.weight.detach().numpy()
print(change_full.shape)


# 1. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(change_full)

# 2. 执行PCA
pca = PCA()
pca.fit(X_scaled)

# 3. 获取每个主成分的方差贡献比例
explained_variance_ratio = pca.explained_variance_ratio_

# 4. 计算累计解释方差
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 输出结果
print("各主成分的解释方差比例:", explained_variance_ratio)
print("累计解释方差:", cumulative_explained_variance)

plt.figure(figsize=(10, 6))
x = range(4096)  # 
y = cumulative_explained_variance  

# 绘制折线图
plt.plot(x, y, color='b')

plt.show()
output_path = 'llm_loss_distribution.png'  # 图像保存路径
plt.savefig(output_path)



