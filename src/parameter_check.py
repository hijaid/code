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


layer_num = 16
base_model_path = "/mnt/luoyingfeng/model_card/Meta-Llama-3.1-8B"
fft_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/fft_1e-5_2epoch/de-en/checkpoint-200"
lora_model_path = "/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/lora_16_5e-5_2epoch/de-en/adapter/checkpoint-200"
fft_model = AutoModelForCausalLM.from_pretrained(fft_model_path, trust_remote_code=True, torch_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16)

fft_check_layer = fft_model.model.layers[layer_num]
base_check_layer = base_model.model.layers[layer_num]


print(f"fft weights: {fft_check_layer.self_attn.q_proj.weight.detach().numpy().shape}")
print(f"base weights: {base_check_layer.self_attn.q_proj.weight.detach().numpy()[0]}")

lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
lora_check_layer = base_model.model.layers[layer_num]
#for name, param in base_model.named_parameters():
#    print(name)
print(f"lora A weights: {lora_check_layer.self_attn.q_proj.lora_A.default.weight.detach().numpy().shape}")
print(f"lora B weights: {lora_check_layer.self_attn.q_proj.lora_B.default.weight.detach().numpy().shape}")



# 求出full-sft与base模型之间的参数变化
change_full = fft_check_layer.self_attn.q_proj.weight.detach().numpy() - base_check_layer.self_attn.q_proj.weight.detach().numpy()
print(change_full[0])
# 求出lora-sft与base模型之间的参数变化
change_lora = np.dot(lora_check_layer.self_attn.q_proj.lora_B.default.weight.detach().numpy(),lora_check_layer.self_attn.q_proj.lora_A.default.weight.detach().numpy()) 
print(change_lora[0])
# 分别进行SVD分解，然后观察其奇异值的分布情况（看秩的分布），然后绘制右奇异向量的相关系数图
#print(change_full)
fft_change_U, fft_change_s, fft_change_VT = np.linalg.svd(change_full.astype(np.float64))
print("全量微调左奇异向量矩阵 U:\n", fft_change_U.shape)
print("全量微调奇异值数组 s:\n", fft_change_s[:10])
print(np.sum(fft_change_s > fft_change_s[0] * 0.001))
#print(np.linalg.matrix_rank(change_full.astype(np.float64)))
print("全量微调右奇异向量矩阵的转置 VT:\n", fft_change_VT.shape)

lora_U, lora_s, lora_VT = np.linalg.svd(change_lora.astype(np.float64))
print("LoRA微调左奇异向量矩阵 U:\n", lora_U.shape)
#print("LoRA微调奇异值数组 s:\n", lora_s[:20])
print(np.sum(lora_s > 0.00001))
#print(np.linalg.matrix_rank(change_lora.astype(np.float64)))
print("LoRA微调右奇异向量矩阵的转置 VT:\n", lora_VT.shape)

# for name, param in lora_model.named_parameters():
#     if 'lora' in name:  # LoRA的参数通常带有 'lora' 字样
#         print(f"LoRA Parameter: {name}, Shape: {param.shape}")

#for name, param in lora_model.named_parameters():
#    if 'layer' in name:  # 如果参数名称中包含"layer"（可以根据需要调整筛选条件）
#        print(f"{name}: {param.shape}")
#print(lora_model.layers[31].self_attn.v_proj.weight.shape)



## 向量分析
base_U, base_s, base_VT = np.linalg.svd(base_check_layer.self_attn.q_proj.weight.detach().numpy().astype(np.float64))
fft_U, fft_s, fft_VT = np.linalg.svd(fft_check_layer.self_attn.q_proj.weight.detach().numpy().astype(np.float64))
lora_U, lora_s, lora_VT = np.linalg.svd((base_check_layer.self_attn.q_proj.weight.detach().numpy() + change_lora).astype(np.float64))

