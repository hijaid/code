#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
from typing import Tuple
import tqdm
import time
import os
import json
import random

import torch
from accelerate import PartialState, Accelerator
from accelerate.utils import set_seed, gather_object

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    LlamaCrossAttentionEncDec,
    AutoConfig
)
from functools import reduce
from utils import utils
import datetime

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def apply_prompt_txt(args, src_fullname, tgt_fullname, tokenizer):
    model_name_or_path = args.model_name_or_path
    test_dataset = open(args.test_file, encoding='utf8', mode='r').read().strip().split("\n")
    random.shuffle(test_dataset)
    res = []
    for line in test_dataset:
        if "ALMA" in model_name_or_path or "gemma-2" in model_name_or_path:
            prefix = "Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: {src}\n{tgt_fullname}:"
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            res.append(prompt)
        elif "Tower" in model_name_or_path:
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}\n{tgt_fullname}:"
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res.append(text)
        elif "mt5" in model_name_or_path or "stack_L8_D1024_m2m_gfuse_s2" in model_name_or_path or "D1024_mock" in model_name_or_path:
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}"
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            res.append(prompt)
        elif "Meta-Llama-3-8B" in model_name_or_path:
            # Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}\n{tgt_fullname}: "
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            res.append(prompt)
        elif "Meta-Llama-3.1" in model_name_or_path:
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}. Do not provide any explanations or text apart from the translation.\n{src}"
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res.append(text)
        elif "nllb" in model_name_or_path or "wmt23_50M_dict32k_big_40_8" in model_name_or_path:
            res = test_dataset
        elif "aya-23" in model_name_or_path:
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}. Do not provide any explanations or text apart from the translation.\n{src}"
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            res.append(text)
        else:
            print("Not support this model")
            exit()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--lang_pair", type=str, default='de-en')
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--record_path", default=None, type=str)
    args = parser.parse_args()
    set_seed(args.seed)
    
    src_lang, tgt_lang = args.lang_pair.split("-")
    src_fullname = utils.LANG_TABLE[src_lang]
    tgt_fullname = utils.LANG_TABLE[tgt_lang]
    if "ALMA-7B" in args.model_name_or_path or "Tower" in args.model_name_or_path:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = 'auto'
    #torch_dtype = torch.bfloat16
    #torch_dtype = torch.float16
    # for nllb model
    langcodes = {"en": "eng_Latn", "de":"deu_Latn", "cs":"ces_Latn", "ru":"rus_Cyrl", "zh":"zho_Hans"}
    # Initialize the distributed state.        
    if "nllb" in args.model_name_or_path:
        # https://github.com/facebookresearch/flores/tree/main/flores200
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True, src_lang=langcodes[src_lang])
    elif "mt5" in args.model_name_or_path or "wmt23_50M_dict32k_big_40_8" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_special_tokens=False, padding_side="left", trust_remote_code=True)

    if "Llama-2" in args.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if "Llama-3" in args.model_name_or_path:
        tokenizer.pad_token_id = 128002
    if "D1024_mock" in args.model_name_or_path:
        tokenizer.eos_token_id = 128002

    if "nllb" in args.model_name_or_path or "mt5" in args.model_name_or_path or "wmt23_50M_dict32k_big_40_8" in args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map='cuda'
        )
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    elif "stack_L8_D1024_m2m_gfuse_s2" in args.model_name_or_path or "D1024_mock" in args.model_name_or_path:
        model = LlamaCrossAttentionEncDec.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            device_map='cuda'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map='cuda'
        )
        

    if "nllb" in args.model_name_or_path:
        forced_bos_token_id = tokenizer.lang_code_to_id[langcodes[tgt_lang]]
    else:
        forced_bos_token_id = None

    test_dataset = apply_prompt_txt(args, src_fullname, tgt_fullname, tokenizer)

    print("\n\n=====================\n\n".join(random.sample(test_dataset, 10)))
    print(f"predict file {args.test_file}")

    # batch, left pad (for inference), and tokenize
    def make_batch(prompts, batch_size=4):
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
        batches_tok = []
        for prompt_batch in batches:
            input_ids = tokenizer(
                    prompt_batch, 
                    return_tensors="pt", 
                    padding='longest', 
                    truncation=False
                    ).to("cuda") 
            batches_tok.append(input_ids)
                
        return batches_tok

    start = time.time()
    results = []
    num_tokens = 0
   
    prompt_batches = make_batch(test_dataset, batch_size=args.num_batch)
    for prompts_tokenized in tqdm.tqdm(prompt_batches, total=len(prompt_batches)):
        outputs_tokenized = model.generate(
            **prompts_tokenized,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_k = 50,
            top_p = 0.8,
            #num_beams=args.num_beams,
            repetition_penalty=1.0,
            do_sample=True,
            min_new_tokens=150,
            num_return_sequences=1,
            forced_bos_token_id=forced_bos_token_id
        )
        
        # seq2seq model
        if "nllb" in args.model_name_or_path or "mt5" in args.model_name_or_path or "stack_L8_D1024_m2m_gfuse_s2" in args.model_name_or_path or \
            "wmt23_50M_dict32k_big_40_8" in args.model_name_or_path or "D1024_mock" in args.model_name_or_path:
            num_tokens += sum([ len(t) for t in outputs_tokenized ])
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
        # language model need remove prompt from gen. tokens
        else:
            outputs_tokenized = [ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

            # count and decode gen. tokens 
            num_tokens += sum([ len(t) for t in outputs_tokenized ])
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
        
        results += outputs

    timediff = time.time() - start
    print("\n\n".join([f"{x}\n{y}" for x,y in random.sample(list(zip(test_dataset, results)), 10)]))
    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
    with open(args.record_path, "a", encoding="utf-8") as write_f:
        write_f.write(args.model_name_or_path + "\n")
        write_f.write(args.test_file + "\n")
        write_f.write(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}" + "\n\n")

if __name__ == "__main__":
    main()
