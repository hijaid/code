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
   # LlamaCrossAttentionEncDec,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
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

# 
torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))


def apply_prompt_txt(args, src_fullname, tgt_fullname, tokenizer):
    model_name_or_path = args.model_name_or_path
    test_dataset = open(args.test_file, encoding='utf8', mode='r').read().strip().split("\n")
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
        elif "Meta-Llama-3-8B" in model_name_or_path:
            ## sft
            # Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}\n{tgt_fullname}: "
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            res.append(prompt)
        elif "Meta-Llama-3.1" in model_name_or_path or "Meta-Llama-3-8B" in model_name_or_path:
            # instruct
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
        elif "nllb" in model_name_or_path:
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


def apply_prompt_json_direct(args, src_fullname, tgt_fullname, tokenizer):
    model_name_or_path = args.model_name_or_path
    test_dataset = [json.loads(x) for x in open(args.test_file, encoding='utf8', mode='r')]
    res = []
    for line in test_dataset:
        src_lang = line["src_lang"]
        line = line["translation"][src_lang]
        # doc trans or sent trans
        line = line if type(line) is str else " ".join(line)
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
        elif "Meta-Llama-3.1" in model_name_or_path or "Meta-Llama-3-8B" in model_name_or_path:
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
        elif "nllb" in model_name_or_path:
            res.append(line)
        elif "mt5" in model_name_or_path:
            prefix = "Translate the following text from {src_fullname} into {tgt_fullname}.\n{src_fullname}: {src}"
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=line)
            res.append(prompt)
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

# for chat version model
task_prompts = {
    "doc_trans": "Translate the following text from {src_lang} into {tgt_lang}. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}",
    "domain_medical": "Translate the following text from {src_lang} to {tgt_lang}. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}",
    "domain_law": "Translate the following text from {src_lang} to {tgt_lang}. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}",
    "domain_it": "Translate the following text from {src_lang} to {tgt_lang}. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}",
    "domain_literature": "Translate the following text from {src_lang} to {tgt_lang}. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}",
    "domain_colloquial": "Translate the following text from {src_lang} to {tgt_lang}. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}",
    "term_con_trans": "Translate the following text from {src_lang} into {tgt_lang} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated. Do not provide any explanations or text apart from the translation.\nTerminology pairs: {term_text}\n{src_lang}: {src}",
    "ape": "Improve the following machine-generated translation from {src_lang} to {tgt_lang}. Correct errors and generate a more accurate translation. Do not provide any explanations or text apart from the translation.\n{src_lang}: {src}\nMachine translation: {mt_text}",
}


def apply_prompt_json(args, task_type, src_fullname, tgt_fullname, tokenizer):
    model_name_or_path = args.model_name_or_path
    test_dataset = [json.loads(x) for x in open(args.test_file, encoding='utf8', mode='r')]
    res = []
    for line in test_dataset:
        src_lang, tgt_lang = line["src_lang"], line["tgt_lang"]
        # prompt fixed model
        if "ALMA" in model_name_or_path or "gemma-2" in model_name_or_path:
            prefix = "Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: {src}\n{tgt_fullname}:"
            src_text = line["translation"][src_lang]
            prompt = prefix.format(src_fullname=src_fullname, tgt_fullname=tgt_fullname, src=src_text)
            res.append(prompt)
        elif "Tower" in model_name_or_path or "aya-23" in model_name_or_path:
            if task_type == "term_con_trans":
                src_text, hints = line["translation"][src_lang], line["hints"]
                hints = [f"{x[src_lang]} = {x[tgt_lang]}" for x in hints]
                hint_text = " ; ".join(hints)
                prompt = task_prompts[task_type].format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, term_text=hint_text)
            elif task_type == "ape":
                src_text, mt_text = line["translation"][src_lang], line["mt_gen"]
                prompt = task_prompts[task_type].format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, mt_text=mt_text)
            else:
                src_text = line["translation"][src_lang]
                prompt = task_prompts[task_type].format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
          
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res.append(text)
        
        elif "Meta-Llama-3.1-Instruct" in model_name_or_path or "Meta-Llama-3-8B-Instruct" in model_name_or_path:
            if task_type == "term_con_trans":
                src_text, hints = line["translation"][src_lang], line["hints"]
                hints = [f"{x[src_lang]} = {x[tgt_lang]}" for x in hints]
                hint_text = " ; ".join(hints)
                prompt = task_prompts[task_type].format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, term_text=hint_text)
            elif task_type == "ape":
                src_text, mt_text = line["translation"][src_lang], line["mt_gen"]
                prompt = task_prompts[task_type].format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, mt_text=mt_text)
            else:
                src_text = line["translation"][src_lang]
                prompt = task_prompts[task_type].format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
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
        elif "nllb" in model_name_or_path:
            src_text = line["translation"][src_lang]
            res.append(src_text)
        else:
            print("Not support this model")
            exit()
    return res


def clean_pred(pred, remove_special_tokens=[]):
    ## remove special tokens
    for s in remove_special_tokens:
        pred = pred.replace(s, "")
    ## last step: check
    pred = "#" if utils.is_whitespace(pred) else pred
    return pred


def get_special_tokens(tokenizer):
    remove_special_tokens = ["<unk>", "</s>", "<pad>", "\n"]
    if getattr(tokenizer, "pad_token", None):
        remove_special_tokens.append(tokenizer.pad_token)
    if getattr(tokenizer, "eos_token", None):
        remove_special_tokens.append(tokenizer.eos_token)
    if getattr(tokenizer, "bos_token", None):
        remove_special_tokens.append(tokenizer.bos_token)
    if getattr(tokenizer, "unk_token", None):
        remove_special_tokens.append(tokenizer.unk_token)
    return remove_special_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--hypo_file", type=str, default="")
    parser.add_argument("--lang_pair", type=str, default='de-en')
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of 1.0 has no effect, lower tend toward greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2" )
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--num_batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--do_sample", type=bool, default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()

    infos = args.test_file.split(".")
    suffix = infos[-1]
    # test.en-zh.general_trans.wmt22.json
    if suffix in ["json", "jsonl"]:
        lang_pair = infos[-4]
        src_lang, tgt_lang = lang_pair.split("-")
        task_type = infos[-3]
    else:
        src_lang, tgt_lang = args.lang_pair.split("-")
        task_type = "general_trans"
    src_fullname = utils.LANG_TABLE[src_lang]
    tgt_fullname = utils.LANG_TABLE[tgt_lang]
    
    # for nllb model
    langcodes = {
    "af": "afr_Latn", "am": "amh_Ethi", "an": "ast_Latn", "ar": "arb_Arab", "as": "asm_Beng", "av": "bak_Cyrl", "az": "azj_Latn", "be": "bel_Cyrl",
    "bg": "bul_Cyrl", "bn": "ben_Beng", "bo": "bod_Tibt", "br": "ban_Latn", "bs": "bos_Latn", "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn",
    "da": "dan_Latn", "de": "deu_Latn", "dz": "dzo_Tibt", "el": "ell_Grek", "en": "eng_Latn", "eo": "epo_Latn", "es": "spa_Latn", "et": "est_Latn",
    "eu": "eus_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "fy": "fur_Latn", "ga": "gle_Latn", "gd": "gla_Latn", "gl": "glg_Latn",
    "gu": "guj_Gujr", "ha": "hau_Latn", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "ig": "ibo_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr", "kn": "kan_Knda",
    "ko": "kor_Hang", "ku": "kmr_Latn", "ky": "kir_Cyrl", "li": "lim_Latn", "lt": "lit_Latn", "lv": "ltg_Latn", "mg": "plt_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva", "ms": "zsm_Latn", "mt": "mlt_Latn", "my": "mya_Mymr", "nb": "nob_Latn", "ne": "npi_Deva",
    "nl": "nld_Latn", "nn": "nno_Latn", "no": "nno_Latn", "oc": "oci_Latn", "or": "ory_Orya", "pa": "pan_Guru", "pl": "pol_Latn", "ps": "pbt_Arab",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "rw": "kin_Latn", "se": "swe_Latn", "sh": "shn_Mymr", "si": "sin_Sinh", "sk": "slk_Latn",
    "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "ta": "tam_Taml", "te": "tel_Telu", "tg": "tgk_Cyrl", "th": "tha_Thai",
    "tk": "tuk_Latn", "tr": "tur_Latn", "tt": "tat_Cyrl", "ug": "uig_Arab", "uk": "ukr_Cyrl", "ur": "urd_Arab", "uz": "uzn_Latn", "vi": "vie_Latn",
    "wa": "war_Latn", "xh": "xho_Latn", "yi": "ydd_Hebr", "yo": "yor_Latn", "zh": "zho_Hans", "zu": "zul_Latn",
    }

    # Initialize the distributed state.        
    if "nllb" in args.model_name_or_path:
        # https://github.com/facebookresearch/flores/tree/main/flores200
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True, src_lang=langcodes[src_lang])
    elif "mt5" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_special_tokens=False, padding_side="left", trust_remote_code=True)
    remove_special_tokens = get_special_tokens(tokenizer)

    if "Llama-2" in args.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if "Llama-3" in args.model_name_or_path:
        tokenizer.pad_token_id = 128002

    # torch_dtype = 'auto'
    # torch_dtype = torch.bfloat16
    torch_dtype = torch.float16
    if "nllb" in args.model_name_or_path :
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": accelerator.process_index},
        )
        forced_bos_token_id = tokenizer.lang_code_to_id[langcodes[tgt_lang]]
    elif "mt5" in args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": accelerator.process_index},
        )
        forced_bos_token_id = None
    # elif "stack_L8_D1024_m2m_gfuse_s2" in args.model_name_or_path or "D1024_mock" in args.model_name_or_path:
    #     model = LlamaCrossAttentionEncDec.from_pretrained(
    #         args.model_name_or_path, 
    #         torch_dtype=torch_dtype,
    #         device_map='cuda'
    #     )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": accelerator.process_index},
            # attn_implementation="flash_attention_2"
        )
        forced_bos_token_id = None

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    
    if suffix in ["json", "jsonl"]:
        test_dataset = apply_prompt_json(args, task_type, src_fullname, tgt_fullname, tokenizer)
        # test_dataset = apply_prompt_json_direct(args, src_fullname, tgt_fullname, tokenizer)
    # general translation task
    else:
        test_dataset = apply_prompt_txt(args, src_fullname, tgt_fullname, tokenizer)

    if accelerator.is_main_process:
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
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(test_dataset) as prompts:
        results = dict(outputs=[], num_tokens=0)

        # have each GPU do inference in batches
        prompt_batches = make_batch(prompts, batch_size=args.num_batch)
        prompt_batches = tqdm.tqdm(prompt_batches, total=len(prompt_batches), disable=not accelerator.is_local_main_process)
        for prompts_tokenized in prompt_batches:
            outputs_tokenized = model.generate(
                **prompts_tokenized,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_beams=args.num_beams,
                repetition_penalty=args.repetition_penalty,
                #top_p = args.topp,
                #top_k = args.topk,
                do_sample = args.do_sample,
                # do_sample=True,
                num_return_sequences=args.num_return_sequences,
                forced_bos_token_id=forced_bos_token_id
            )

            
            if "nllb" in args.model_name_or_path or "mt5" in args.model_name_or_path or "stack_L8_D1024_m2m_gfuse_s2" in args.model_name_or_path:
                num_tokens = sum([ len(t) for t in outputs_tokenized ])
                outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            # remove prompt from gen. tokens
            else:
                outputs_tokenized = [ tok_out[len(tok_in):] 
                    for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 

                # count and decode gen. tokens 
                num_tokens = sum([ len(t) for t in outputs_tokenized ])
                outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
            
            # print("\n\n".join(outputs))
            # store in results{} to be gathered by accelerate
            outputs = list(map(lambda x: clean_pred(x, remove_special_tokens=remove_special_tokens),  outputs))
            # print(outputs)
            results["outputs"].extend(outputs)
            results["num_tokens"] += num_tokens
    results = [ results ]

    # collect results from all the GPUs
    results_gathered = gather_object(results)
    
    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum([r["num_tokens"] for r in results_gathered ])
        preds = list(reduce(lambda x,y: x+y["outputs"], results_gathered, []))
        print("\n".join(preds))
        print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")
        with open(args.hypo_file, mode='w') as fout:
            fout.write("\n".join(preds) + '\n')
                
if __name__ == "__main__":
    main()
