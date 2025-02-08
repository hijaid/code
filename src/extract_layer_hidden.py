from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BertModel,
    AutoModelForSeq2SeqLM,
    FSMTTokenizer,
    FSMTForConditionalGeneration,
)
import datasets
import transformers
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
import torch
import tqdm
from functools import partial
import logging
import json
import random
import time
from functools import reduce
import pickle

logger = logging.getLogger(__name__)
log_level = "ERROR"
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
from accelerate.utils import set_seed, gather_object
import evaluate
from accelerate import Accelerator


from utils import utils

def extract_pred(pred_text, split_str, remove_special_tokens=[]):
    ## extract pred
    pred = pred_text.split(split_str)[0].strip()
    pred = pred.split("\n")[0].strip()
    ## remove special tokens
    for s in remove_special_tokens:
        pred = pred.replace(s, "")
    ## last step: check
    pred = "#" if utils.is_whitespace(pred) else pred
    return pred


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, )
    parser.add_argument("--test_file", type=str,)
    parser.add_argument("--lang_pair", type=str, default='de-en')
    parser.add_argument("--res_file", type=str, )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    return parser.parse_args()

def extract_hidden():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, add_special_tokens=False, padding_side="left", trust_remote_code=True)
    
    if "Llama-2" in args.model_name_or_path or "Tower" in args.model_name_or_path:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if "Llama-3" in args.model_name_or_path:
        tokenizer.pad_token_id = 128002

    torch_dtype='auto'
    # torch_dtype=torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map={"": accelerator.process_index},
        )
    
    model.eval()

    test_dataset = [json.loads(line) for line in open(args.test_file)][:1000]
    
    src_lang, tgt_lang = args.lang_pair.split("-")
    src_fullname = utils.LANG_TABLE[src_lang]
    tgt_fullname = utils.LANG_TABLE[tgt_lang]

    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n"

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()

    def zero_shot(example):
        src = example["translation"][src_lang]
        prompt = prefix
        prompt += f"{src_fullname}: {src}\n" + f"{tgt_fullname}: "
        example["prompt"] = prompt
        return example

    test_dataset = list(map(zero_shot, test_dataset))
    
    # divide the prompt list onto the available GPUs 
    test_dataset_input = [x["prompt"] for x in test_dataset]
    with accelerator.split_between_processes(test_dataset_input) as prompts:
        results = []

        prompts = tqdm.tqdm(prompts, total=len(prompts), disable=not accelerator.is_local_main_process)
        for prompt in prompts:
            prompts_tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
            outputs = model(**prompts_tokenized, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            hidden_states = list(map(lambda x: x.squeeze().detach().to("cpu"), hidden_states)) # list of n x d
            results.append(hidden_states)


        
    # all_results = [ results ] # transform to list, otherwise gather_object() will not collect correctly

    # collect results from all the GPUs
    results_gathered = gather_object(results)
    if accelerator.is_main_process:
        with open(args.res_file, mode='wb') as fout:
             pickle.dump(results_gathered, fout)
                
if __name__ == "__main__":
    extract_hidden()