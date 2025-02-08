import argparse
import logging
from typing import Tuple
import tqdm
import time

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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


layer_num = 16
base_model_path = "/mnt/luoyingfeng/model_card/Meta-Llama-3.1-8B"
fft_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/fft_multi_1e-5_2epoch/checkpoint-415"
lora_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/lora_multi_16_5e-5_3epoch/adapter/checkpoint-250"
# base_model_path = "/mnt/luoyingfeng/model_card/Meta-Llama-3.2-1B"
# fft_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.2-1B/fft_1e-5_2epoch/de-en/checkpoint-200"
# lora_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.2-1B/lora_16_5e-5_2epoch/de-en/adapter/checkpoint-168"
fft_model = AutoModelForCausalLM.from_pretrained(fft_model_path, trust_remote_code=True, torch_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16)

fft_check_layer = fft_model.model.layers[layer_num]
base_check_layer = base_model.model.layers[layer_num]


print(f"fft weights: {fft_check_layer.self_attn.q_proj.weight.detach().numpy().shape}")
print(f"base weights: {base_check_layer.self_attn.q_proj.weight.detach().numpy().shape}")

lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
lora_check_layer = base_model.model.layers[layer_num]
print(f"lora A weights: {lora_check_layer.self_attn.q_proj.lora_A.default.weight.detach().numpy().shape}")
print(f"lora B weights: {lora_check_layer.self_attn.q_proj.lora_B.default.weight.detach().numpy().shape}")
change_lora = np.dot(lora_check_layer.self_attn.q_proj.lora_B.default.weight.detach().numpy(),lora_check_layer.self_attn.q_proj.lora_A.default.weight.detach().numpy()) 

## 向量分析
# 分别进行SVD分解，然后观察其奇异值的分布情况（看秩的分布），然后绘制右奇异向量的相关系数图
base_U, base_s, base_VT = np.linalg.svd(base_check_layer.self_attn.q_proj.weight.detach().numpy().astype(np.float64))
fft_U, fft_s, fft_VT = np.linalg.svd(fft_check_layer.self_attn.q_proj.weight.detach().numpy().astype(np.float64))
lora_U, lora_s, lora_VT = np.linalg.svd((base_check_layer.self_attn.q_proj.weight.detach().numpy() + change_lora).astype(np.float64))

#print(base_VT[:16][0].shape)
similarity_matrix = cosine_similarity(base_VT[:4096], fft_VT[:4096])
print(similarity_matrix)
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", fmt='.2f', linewidths=0.5)
output_path = 'base_fft_similarity.png'  # 图像保存路径
plt.savefig(output_path)


similarity_matrix = cosine_similarity(base_VT[:4096], lora_VT[:4096])
print(similarity_matrix)
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", fmt='.2f', linewidths=0.5)
output_path = 'base_lora_similarity.png'  # 图像保存路径
plt.savefig(output_path)




