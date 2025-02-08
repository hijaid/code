#!/usr/bin/env python
# coding=utf-8

import logging
import copy
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import numpy as np
import datasets
import evaluate
import torch
from datasets import load_dataset, Dataset, DatasetDict
import glob
import regex
import random
import math
random.seed(42)
from functools import reduce
from collections import defaultdict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    LlamaTokenizer,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training , TaskType
from peft import PeftModel, PeftConfig
from collections import defaultdict
from transformers.trainer_callback import TrainerCallback
from datasets import concatenate_datasets
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead
from safetensors import safe_open

from modeling_llama_prefix import LlamaForPrefixLM
from modeling_qwen2_prefix import Qwen2ForPrefixLM

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")

        if os.path.isfile(pytorch_model_path) and torch.distributed.get_rank() == 0:
            os.remove(pytorch_model_path)
            # create an empty toy file to avoid error in deleting old checkpoints
            open(pytorch_model_path, 'w').close()
        return control


LANG_TABLE = {
    "af": "Afrikaans",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dz": "Dzongkha",
    "el": "Modern Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Modern Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kirghiz",
    "li": "Limburgish",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "or": "Oriya",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "se": "Northern Sami",
    "sh": "Serbo-Croatian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovene",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uighur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wa": "Walloon",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}

# task_prompt = {
#     "general_trans":[
#         "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "doc_trans":[
#         "Translate the following long text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "domain_medical":[
#         "Translate the following medical text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "domain_law":[
#         "Translate the following legal text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "domain_literature":[
#         "Translate the following literary text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "domain_colloquial":[
#         "Translate the following colloquial text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "domain_it":[
#         "Translate the following information technology-related text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
#     ],
#     "term_con_trans": [
#         "Translate the following text from {src_lang} into {tgt_lang} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated.\nTerminology pairs: {term_text}\n{src_lang}: {src}"
#     ],
#     "ape": [
#         "Improve the following machine-generated translation from {src_lang} to {tgt_lang}. Correct errors and generate a more accurate translation.\n{src_lang}: {src}\nMachine translation: {mt_text}"
#     ]
# }


task_prompt = {
    "general_trans":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "doc_trans":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_medical":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_law":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_literature":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_colloquial":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "domain_it":[
        "Translate the following text from {src_lang} into {tgt_lang}.\n{src_lang}: {src}"
    ],
    "term_con_trans": [
        "Translate the following text from {src_lang} into {tgt_lang} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated.\nTerminology pairs: {term_text}\n{src_lang}: {src}"
    ],
    "ape": [
        "Improve the following machine-generated translation from {src_lang} to {tgt_lang}. Correct errors and generate a more accurate translation.\n{src_lang}: {src}\n{tgt_lang}: {mt_text}"
    ]
}

## Prefix and suffix for prompt in target language (only from English to target language if the target is non-English)
## Note that prefix and suffix for other languages are only used for zero-shot evaluation of other models.
## ALMA should only use English Prompt.
PREFIX = {
    "de": "Übersetzen Sie dies vom Englischen ins Deutsche:\nEnglisch: ",
    "fr": "Traduisez ceci de l'anglais vers le français :\nAnglais: ",
    "cs": "Přeložte toto z angličtiny do češtiny:\nanglicky: ",
    "is": "Þýddu þetta úr ensku yfir á íslensku:\nEnska: ",
    "zh": "将其从英文翻译成中文：\n英语：",
    "ja": "これを英語から日本語に翻訳してください:\n英語：",
    "ru": "Переведите это с английского на русский:\nАнглийский: ",
    "uk": "Перекладіть це з англійської на українську:\nАнглійська: ",
    "ha": "Fassara wannan daga Turanci zuwa Hausa:\nTuranci: ",
}

SUFFIX = {
    "en": "\nEnglish:",
    "de": "\nDeutsch:",
    "fr": "\nFrançais :",
    "cs": "\nčesky:",
    "is": "\nÍslenska:",
    "zh": "\n中文：",
    "ja": "\n日本語：",
    "ru": "\nРусский:",
    "uk": "\nУкраїнська:",
    "ha": "\nHausa:",
}


def is_whitespace(string):
    # 使用正则表达式匹配空白字符或不可见字符
    pattern = r'^[\s\p{C}[\x00-\xFF]]+$'
    match = regex.match(pattern, string)
    return match is not None

def load_checkpoint(model_path):
    ## load checkpoint
    checkpoint_url = glob.glob(f"{model_path}/*model.bin")
    state = {}
    if len(checkpoint_url) != 0:
        for part in checkpoint_url:
            state.update(torch.load(part))
    else:
        checkpoint_url = glob.glob(f"{model_path}/*safetensors")
        if len(checkpoint_url) == 0:
            print("No checkpoint!")
            exit()
        for part in checkpoint_url:
            with safe_open(part, framework="pt") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
    return state

def make_prefixlm_state_dict(model_path):
    ## get encoder state and lm_head 
    state = load_checkpoint(model_path)
    new_state = {}
    for key, value in state.items():
        if key.startswith("model"):
            key =  "decoder" + key[5:]
        # lm_head
        new_state[key] = value
    new_state["encoder.embed_tokens.weight"] = copy.deepcopy(new_state["decoder.embed_tokens.weight"])
    return new_state

def make_model_state_dict(model_path, config, model_method):   
    
    def concat_attention_layer_wise(k, m, middle=0, decoder_param_method="share"):
        ## init from bottom k layer and top m layers
        decoder_state = {}
        if decoder_param_method in ["share", "init"]:
            if middle == 0:
                sample_middle_layers = []
            else:
                ## unified need keep top and bottom
                if k == 0 and m == 0:
                    m = 1
                    middle -= 1
                    interval = math.ceil((encoder_layer_num - k - m) / middle)
                    if (middle - 1) * interval >= encoder_layer_num:
                        interval -= 1
                else:
                    interval = (encoder_layer_num - k - m) // middle
                sample_middle_layers = [k+i*interval for i in range(middle)]
        elif decoder_param_method == "random":
            sample_middle_layers = random.sample(list(range(k, encoder_layer_num - m)), middle)
            sample_middle_layers.sort()
        else:
            print("Not implement this decoder_param_method yet!")
            exit()

        encoder_layer_index = list(range(k)) + sample_middle_layers + list(range(encoder_layer_num-m, encoder_layer_num))
        decoder_layer_index = list(range(decoder_layer_num))
        assert len(encoder_layer_index) == len(decoder_layer_index)
        config.decoder["share_encoder_layers_id"] = encoder_layer_index

        for i,j in zip(encoder_layer_index, decoder_layer_index):
            encoder_prefix = f"encoder.layers.{i}."
            decoder_prefix = f"decoder.layers.{j}."
            for key,value in new_state.items():
                if key.startswith(encoder_prefix):
                    decoder_key = key.replace(encoder_prefix, decoder_prefix)
                    decoder_state[decoder_key] = copy.deepcopy(value)
                
                ## for cross attention moudle
                if model_method == "CrossAttLW" and key.startswith(encoder_prefix + "self_attn"):
                    decoder_key = key.replace(encoder_prefix + "self_attn", decoder_prefix + "cross_attn")
                    decoder_state[decoder_key] = copy.deepcopy(value)
                if model_method == "CrossAttLW" and key.startswith(encoder_prefix + "input_layernorm"):
                    decoder_key = key.replace(encoder_prefix + "input_layernorm", decoder_prefix + "cross_atten_layernorm")
                    decoder_state[decoder_key] = copy.deepcopy(value)

        ## copy encoder's embed_tokens and norm weight to decoder
        new_state["decoder.embed_tokens.weight"] = copy.deepcopy(new_state["encoder.embed_tokens.weight"])
        new_state["decoder.norm.weight"] = copy.deepcopy(new_state["encoder.norm.weight"])

        new_state.update(decoder_state)
        return new_state

    def init_decoder_from_exist_model(decoder_model_path):
        load_decoder_state = load_checkpoint(decoder_model_path)
        decoder_state = {}
        for key, value in load_decoder_state.items():
            if key.startswith("model"):
                key =  "decoder" + key[5:]
            decoder_state[key] = value
        new_state.update(decoder_state)
        return new_state

    ## get encoder state and lm_head 
    state = load_checkpoint(model_path)
    new_state = {}
    for key, value in state.items():
        if key.startswith("model"):
            key =  "encoder" + key[5:]
        # lm_head
        new_state[key] = value

    ## get decoder other's state according model method
    encoder_layer_num = config.num_hidden_layers
    decoder_layer_num = config.decoder["num_hidden_layers"]
    k = config.decoder.get("share_bottom_k", 0)
    m = config.decoder.get("share_top_m", 0)
    share_middle = config.decoder.get("share_middle", 0)
    decoder_param_method = config.decoder["decoder_param_method"]
    if model_method == "ConAttT2B":
        assert k == 0
        # init from bottom k layer and top m layers
        new_state = concat_attention_layer_wise(k, m)
    elif model_method in ["ConAttLW", "CrossAttLW"]:
        new_state = concat_attention_layer_wise(k, m, share_middle, decoder_param_method) 
    # only init encoder
    elif model_method in ["TinyConAttLW", "TinyCrossAttLW"]:
        decoder_model_name_or_path = config.decoder.get("decoder_model_name_or_path", None)
        # init from existing model
        if decoder_model_name_or_path:
            new_state = init_decoder_from_exist_model(decoder_model_name_or_path)
        else:
            new_state = new_state
    return new_state


def print_dataset(train_raw_data, valid_raw_data, test_raw_data):
    for part, part_data in  {"train":train_raw_data, "validation":valid_raw_data, "test":test_raw_data}.items():
        for lp, datas in part_data.items():
            for task, data in datas.items():
                print(f"{part}, {lp}, {task}, {len(data[part])}") 

def load_mmt_dataset(pairs, trans_task, data_args, model_args, training_args, logger):
    seen_files =set()
    train_raw_data, valid_raw_data, test_raw_data = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for pair in pairs:
        src_lang = pair.split("-")[0]
        tgt_lang = pair.split("-")[1]
        
        first_lang = src_lang if src_lang != "en" else tgt_lang
        second_lang = "en"
        pair_dir = f"{first_lang}-{second_lang}"
            
        for task in trans_task:
            train_file = os.path.join(data_args.mmt_data_path, pair_dir, f"train.{pair_dir}.{task}.json")
            valid_file = os.path.join(data_args.mmt_data_path, pair_dir, f"valid.{pair_dir}.{task}.json")

            # test_file may have multi test dataset
            if task == "general_trans":
                if data_args.test_dataname == "wmt23":
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}.wmt23*json"))
                elif data_args.test_dataname == "wmt22":
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}.wmt22*json"))
                elif data_args.test_dataname == "flores":
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}.FLORES-200*json"))
                else:
                    test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}*json"))
            else:
                test_files = glob.glob(os.path.join(data_args.mmt_data_path, pair_dir, f"test.{pair}.{task}*json"))

            if test_files:
                test_file = test_files[0]
            else:
                # fake file for logger
                test_file = f"test.{pair}.{task}.json"
            
            if not os.path.isfile(train_file):
                logger.info(f"Warning: training file {train_file} does not exist!")
            elif train_file not in seen_files and training_args.do_train:
                logger.info(f"Load training file {train_file}!")
                train_raw_data[f"{first_lang}-{second_lang}"][task] = load_dataset(
                    "json",
                    data_files={"train": train_file},
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    streaming=data_args.streaming,
                    num_proc=training_args.dataloader_num_workers
                    )
            
            if not os.path.isfile(valid_file):
                logger.info(f"Warning: validation file {valid_file} does not exist!")
            elif valid_file not in seen_files and training_args.do_eval:
                logger.info(f"Load valid file {valid_file}!")
                valid_raw_data[f"{first_lang}-{second_lang}"][task] = load_dataset(
                    "json",
                    data_files={"validation": valid_file},
                    cache_dir=model_args.cache_dir,
                    use_auth_token=True if model_args.use_auth_token else None,
                    num_proc=training_args.dataloader_num_workers
                    )
            
            if not os.path.isfile(test_file):
                logger.info(f"Warning: test file {test_file} does not exist!")
            elif test_file not in seen_files and training_args.do_predict:
                logger.info(f"Load test file {test_file}!")
                if data_args.override_test_data_path:
                    test_raw_data[f"{src_lang}-{tgt_lang}"][task] = load_dataset(
                        data_args.override_test_data_path,
                        f"{src_lang}-{tgt_lang}",
                        cache_dir=model_args.cache_dir,
                        use_auth_token=True if model_args.use_auth_token else None,
                    )
                else:
                    test_raw_data[f"{src_lang}-{tgt_lang}"][task] = load_dataset(
                        "json",
                        data_files={"test": test_file},
                        cache_dir=model_args.cache_dir,
                        use_auth_token=True if model_args.use_auth_token else None,
                    )

            seen_files.add(train_file)
            seen_files.add(valid_file)
            seen_files.add(test_file)
    print_dataset(train_raw_data, valid_raw_data, test_raw_data)
    return train_raw_data, valid_raw_data, test_raw_data

def load_a_single_text_file(pairs, data_args, model_args):
    assert len(pairs) == 1, "Specific translation text source file only needs one translation direction!"
    src_lang, tgt_lang = list(pairs)[0].split("-")
    test_raw_data = {}
    pair = f"{src_lang}-{tgt_lang}"
    test_raw_data[pair] = load_dataset(
        'text',
        data_files={"test": data_args.text_test_file},
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
        )
    def format_features(example):
        return {pair: {src_lang: example["text"], tgt_lang: ""}}

    test_raw_data[pair] = test_raw_data[pair].map(format_features, remove_columns=["text"])

    return test_raw_data

def get_first_non_pad_index(input_tensor):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != -100).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

def get_first_special_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_pad_index = (input_tensor == special).nonzero(as_tuple=True)[0]
    if len(first_pad_index) > 0:
        return first_pad_index[0].item()
    else:
        return -1

def get_first_special_index_batch(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 2
    matches = input_tensor.eq(special).long()
    indices = matches.argmax(dim=1)
    indices[matches.sum(dim=1) == 0] = -1
    return indices 

def get_first_non_specical_index(input_tensor, special):
    input_tensor = torch.tensor(input_tensor)
    assert input_tensor.ndim == 1
    first_non_pad_index = (input_tensor != special).nonzero(as_tuple=True)[0][0]
    return first_non_pad_index.item()

# Suffix for splitting and getting the generated sentences
def get_key_suffix(tgt_lang, data_args):
    if data_args.use_target_lang_prompt_eval:
        return SUFFIX[tgt_lang]
    else:
        return f"\n{LANG_TABLE[tgt_lang]}:"

def get_prompt_few_shot(source_lang, target_lang, ex, shots_eval_dict):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    shots = shots_eval_dict[f"{source_lang}-{target_lang}"]
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:"
    shot_prompt = ""
    for shot in shots:
        shot_src = shot['source']
        shot_tgt = shot['target']
        shot_prompt += f"\n{src_fullname}: " + shot_src + f"\n{tgt_fullname}: " + shot_tgt
    suffix = f"\n{tgt_fullname}:"
    prompt = prefix + shot_prompt + f"\n{src_fullname}: " + ex[source_lang] + suffix
    return prompt

def get_prompt_old(source_lang, target_lang, examples, i, shots_eval_dict={}, use_target_lang_prompt_eval=False, architecture="decoder-only"):
    # if len(shots_eval_dict) != 0:
    #     return get_prompt_few_shot(source_lang, target_lang, examples, shots_eval_dict)
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    task_type = examples["task_type"][i]
    if use_target_lang_prompt_eval and target_lang != "en":
        prefix = PREFIX[target_lang]
        suffix = SUFFIX[target_lang]
    else:
        # use the first prompt defaultly
        prefix_temp = task_prompt[task_type][0]
        if task_type == "doc_trans":
            src_text, tgt_txt = examples["translation"][i][source_lang], examples["translation"][i][target_lang] 
            prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
        elif  task_type == "term_con_trans":
            src_text, tgt_txt, hints = examples["translation"][i][source_lang], examples["translation"][i][target_lang], examples["hints"][i]
            hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
            hint_text = " ; ".join(hints)
            prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, term_text=hint_text)
        elif task_type == "ape":
            src_text, tgt_txt, mt_text = examples["translation"][i][source_lang], examples["translation"][i][target_lang], examples["mt_gen"][i]
            prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, mt_text=mt_text)
        # don't need instruction
        elif task_type == "context_learning_trans": 
            meta_task = examples["meta_task"][i]
            shots = examples["shots"][i]
            if meta_task == "term_con_trans":
                context = ""
                for shot in shots:
                    src_text, tgt_txt, hints = shot["translation"][source_lang], shot["translation"][target_lang], shot["hints"]
                    hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
                    hint_text = " ; ".join(hints)
                    context += f"Terminology pairs: {hint_text}\n{src_fullname}: {src_text}\n{tgt_fullname}: {tgt_txt}\n\n"
                src_text, tgt_txt, hints = examples["translation"][i][source_lang], examples["translation"][i][target_lang], examples["hints"][i]
                hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
                hint_text = "; ".join(hints)
                prefix = context +  f"Terminology pairs: {hint_text}\n{src_fullname}: {src_text}"
            elif meta_task == "ape":
                context = ""
                for shot in shots:
                    src_text, tgt_txt, mt_text = shot["translation"][source_lang], shot["translation"][target_lang], shot["mt_gen"]
                    context += f"{src_fullname}: {src_text}\nMachine translation: {mt_text}\nImproved translation: {tgt_txt}\n\n"
                src_text, tgt_txt, mt_text = examples["translation"][i][source_lang], examples["translation"][i][target_lang], examples["mt_gen"][i]
                prefix = context +  f"{src_fullname}: {src_text}\nMachine translation: {mt_text}"
            else:
                context = ""
                for shot in shots:
                    src_text, tgt_txt = shot["translation"][source_lang], shot["translation"][target_lang]
                    context += f"{src_fullname}: {src_text}\n{tgt_fullname}: {tgt_txt}\n\n"
                src_text, tgt_txt = examples["translation"][i][source_lang], examples["translation"][i][target_lang]
                prefix = context + f"{src_fullname}: {src_text}"
        # normal translation task
        else:
            src_text, tgt_txt = examples["translation"][i][source_lang], examples["translation"][i][target_lang]
            prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
        suffix = f"\n{tgt_fullname}: "

    if architecture == "encoder-decoder":
        suffix = ""
    if task_type == "ape":
        suffix = "\nImproved translation: "
    prompt = prefix + suffix
    return prompt, tgt_txt

def get_prompt_direct(source_lang, target_lang, examples, i, shots_eval_dict={}, use_target_lang_prompt_eval=False, architecture="decoder-only"):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    task_type = examples["task_type"][i]
    if use_target_lang_prompt_eval and target_lang != "en":
        prefix = PREFIX[target_lang]
        suffix = SUFFIX[target_lang]
    else:
        # use the first prompt defaultly
        prefix_temp = task_prompt["general_trans"][0]
        if task_type == "doc_trans":
            src_text, tgt_txt = " ".join(examples["translation"][i][source_lang]), " ".join(examples["translation"][i][target_lang]) 
            prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
        else:
            src_text, tgt_txt = examples["translation"][i][source_lang], examples["translation"][i][target_lang]
            prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
        suffix = f"\n{tgt_fullname}: "

    if architecture == "encoder-decoder":
        suffix = ""
    if task_type == "ape":
        suffix = "\nImproved translation: "
    prompt = prefix + suffix
    return prompt, tgt_txt


def get_prompt(source_lang, target_lang, example):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    task_type = example["task_type"]
    
    if task_type != "context_learning_trans":
        prefix_temp = random.choice(task_prompt[task_type])
    
    if task_type == "doc_trans":
        src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
    elif task_type == "term_con_trans":
        src_text, tgt_txt, hints = example["translation"][source_lang], example["translation"][target_lang], example["hints"]
        hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
        hint_text = " ; ".join(hints)
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, term_text=hint_text)
    elif task_type == "ape":
        src_text, tgt_txt, mt_text = example["translation"][source_lang], example["translation"][target_lang], example["mt_gen"]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text, mt_text=mt_text)
    # don't need instruction
    elif task_type == "context_learning_trans": 
        meta_task = example["meta_task"]
        shots = example["shots"]
        if meta_task == "term_con_trans":
            context = f"Translate the following text from {src_fullname} into {tgt_fullname} using the provided terminology pairs, ensuring the specified terms are accurately translated as indicated.\n"
            for shot in shots:
                src_text, tgt_txt, hints = shot["translation"][source_lang], shot["translation"][target_lang], shot["hints"]
                hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
                hint_text = " ; ".join(hints)
                context += f"Terminology pairs: {hint_text}\n{src_fullname}: {src_text}\n{tgt_fullname}: {tgt_txt}\n\n"
            src_text, tgt_txt, hints = example["translation"][source_lang], example["translation"][target_lang], example["hints"]
            hints = [f"{x[source_lang]} = {x[target_lang]}" for x in hints]
            hint_text = "; ".join(hints)
            prefix = context +  f"Terminology pairs: {hint_text}\n{src_fullname}: {src_text}"
        elif meta_task == "ape":
            context = f"Improve the following machine-generated translation from {src_fullname} to {tgt_fullname}. Correct errors and generate a more accurate translation.\n"
            for shot in shots:
                src_text, tgt_txt, mt_text = shot["translation"][source_lang], shot["translation"][target_lang], shot["mt_gen"]
                context += f"{src_fullname}: {src_text}\n{tgt_fullname}: {mt_text}\nImproved translation: {tgt_txt}\n\n"
            src_text, tgt_txt, mt_text = example["translation"][source_lang], example["translation"][target_lang], example["mt_gen"]
            prefix = context +  f"{src_fullname}: {src_text}\n{tgt_fullname}: {mt_text}"
        else:
            context = f"Translate the following text from {src_fullname} into {tgt_fullname}.\n"
            for shot in shots:
                src_text, tgt_txt = shot["translation"][source_lang], shot["translation"][target_lang]
                context += f"{src_fullname}: {src_text}\n{tgt_fullname}: {tgt_txt}\n\n"
            src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
            prefix = context + f"{src_fullname}: {src_text}"
    # normal translation task
    else:
        src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
    
    if task_type == "ape" or (task_type == "context_learning_trans" and meta_task == "ape"):
        suffix = "\nImproved translation: "
    else:
        suffix = f"\n{tgt_fullname}: "
    prompt = prefix + suffix
    return prompt, tgt_txt


def get_prompt_direct_llm(source_lang, target_lang, example):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    task_type = example["task_type"]

    prefix_temp = task_prompt["general_trans"][0]
    if task_type == "doc_trans":
        src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)
    # normal translation task
    else:
        src_text, tgt_txt = example["translation"][source_lang], example["translation"][target_lang]
        prefix = prefix_temp.format(src_lang=src_fullname, tgt_lang=tgt_fullname, src=src_text)

    suffix = f"\n{tgt_fullname}: "
    prompt = prefix + suffix
    return prompt, tgt_txt

def check_add_eos(tokenized_inputs, tokenizer):
    if tokenized_inputs.input_ids[0][-1] != tokenizer.eos_token_id:
        for idx in range(len(tokenized_inputs.input_ids)):
            tokenized_inputs.input_ids[idx].append(tokenizer.eos_token_id)
            tokenized_inputs.attention_mask[idx].append(1)

def check_add_eos_right_pad(tokenized_inputs, tokenizer):
    for idx in range(len(tokenized_inputs.input_ids)):
        first_non_pad_idx = get_first_special_index(tokenized_inputs.input_ids[idx], tokenizer.pad_token_id)
        tokenized_inputs.input_ids[idx][first_non_pad_idx] = tokenizer.eos_token_id
        tokenized_inputs.attention_mask[idx][first_non_pad_idx] = 1

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def clean_outputstring(output, key_word, logger, split_idx):
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logger.info(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return out[1].strip()
        else:
            logger.info(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return out[2].strip()
    except:
        logger.info(f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")
        
    try:
        return output.split(key_word)[2].split("\n")[0].strip()
    except:
        logger.info(f"Can not solve the edge case, recover the translation to empty string! The output is {output}")
        return ""


def set_model_special_tokens(model, model_name_or_path):
    if "Llama-2" in model_name_or_path or "Tower" in model_name_or_path or "TinyLlama" in model_name_or_path or \
        "ALMA" in model_name_or_path:
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
    elif "BigTranslate" in model_name_or_path:
        model.config.pad_token_id = 2
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.generation_config.pad_token_id = 2
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2 
    elif "mpt" in model_name_or_path:
        model.config.pad_token_id = 1
        model.config.bos_token_id = 0
        model.config.eos_token_id = 0
        model.generation_config.pad_token_id = 1
        model.generation_config.bos_token_id = 0
        model.generation_config.eos_token_id = 0
        for name, param in model.named_parameters():
            # To be compatible with AMD cards
            if "norm" in name:
                param.requires_grad = False
    elif "Qwen1.5" in model_name_or_path or "Qwen2" in model_name_or_path:
        model.config.pad_token_id = 151644
        model.config.bos_token_id = 151643
        model.config.eos_token_id = 151643
        model.generation_config.pad_token_id = 151644
        model.generation_config.bos_token_id = 151643
        model.generation_config.eos_token_id = 151643
    elif "Llama-3" in model_name_or_path:
        model.config.pad_token_id = 128002
        model.generation_config.pad_token_id = 128002
    return model

def set_tokenizer_special_tokens(tokenizer, model_name_or_path):
    if "Llama-2" in model_name_or_path or "Tower" in model_name_or_path or "TinyLlama" in model_name_or_path or \
        "ALMA" in model_name_or_path:
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"
        tokenizer.bos_token = "<s>"
    elif "Mistral" in model_name_or_path:
        tokenizer.pad_token_id = 0
    elif "mpt" in model_name_or_path:
        tokenizer.pad_token_id = 1
        tokenizer.bos_token_id = 0
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = "<|padding|>"
    elif "LiteLlama" in model_name_or_path:
        tokenizer.pad_token_id = 29953
        tokenizer.bos_token_id = 50256
        tokenizer.eos_token_id = 50256
        tokenizer.pad_token = "#$"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
    elif "Qwen1.5" in model_name_or_path or "Qwen2" in model_name_or_path:
        tokenizer.pad_token_id = 151644
        tokenizer.bos_token_id = 151643
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token = "<|im_start|>"
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.bos_token = "<|endoftext|>"
    elif "Llama-3" in model_name_or_path:
        tokenizer.pad_token_id = 128002
    return tokenizer


def load_model(data_args, model_args, training_args, tokenizer, logger):
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and (training_args.do_train or training_args.do_predict ) and not training_args.overwrite_output_dir:
        last_checkpoint = training_args.output_dir
        # last_checkpoint = get_last_checkpoint(training_args.output_dir)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "trust_remote_code": True,
        "max_length": data_args.max_source_length + data_args.max_new_tokens,
        # "norm_type": "low_precision_rmsnorm",
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    if "mpt" in model_args.model_name_or_path:
        config.attn_config["prefix_lm"] = data_args.use_prefix_lm

    ## Model Loading
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        
        if model_args.multi_gpu_one_model and not training_args.do_train:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
                device_map="auto",
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            )
        else:
            if data_args.use_prefix_lm:
                if "llama" in model_args.model_name_or_path.lower():
                    model_class = LlamaForPrefixLM  
                    config._attn_implementation = "eager"
                elif "qwen" in model_args.model_name_or_path.lower():
                    model_class =  Qwen2ForPrefixLM
                    config._attn_implementation = "eager"
            else:
                model_class = AutoModelForCausalLM 

            # model =  model_class.from_pretrained(
            #     model_args.model_name_or_path if last_checkpoint is None else last_checkpoint,
            #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
            #     config=config,
            #     cache_dir=model_args.cache_dir,
            #     revision=model_args.model_revision,
            #     use_auth_token=True if model_args.use_auth_token else None,
            #     torch_dtype=torch_dtype,
            #     low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            #     trust_remote_code=True,
            # )
            model =  model_class.from_pretrained(
                model_args.model_name_or_path ,
                config=config,
            )

        model.generation_config.max_length = data_args.max_source_length + data_args.max_new_tokens
        model.generation_config.use_cache = True
        # when do inference only
        if not training_args.do_train and config.torch_dtype is torch.float32:
            model = model.half()
            logger.info("Model dtype is torch.float32, chanege to torch.float16 for inference only")

    ## train from scratch
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.use_peft:
        if model_args.peft_model_id:
            model = PeftModel.from_pretrained(model, model_args.peft_model_id)
            ## If still need to fine-tune
            for name, param in model.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
        else:
            config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_rank * 2,
                target_modules=["down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
        print_trainable_parameters(model)

    model = set_model_special_tokens(model, model_args.model_name_or_path)
    logger.info(model)
    return model


def load_tokenizer(data_args, model_args, training_args, logger, add_eos_token=False):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "padding_side": 'left' if not data_args.right_pad else "right",
        "add_eos_token": add_eos_token,
        "trust_remote_code": True
    }
        
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        if "llama" in model_args.model_name_or_path or "BigTranslate" in model_args.model_name_or_path or "ALMA" in model_args.model_name_or_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_args.model_name_or_path, 
                **tokenizer_kwargs, 
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                **tokenizer_kwargs,
            )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer = set_tokenizer_special_tokens(tokenizer, model_args.model_name_or_path)
    return tokenizer


def train_lora_model(model, model_args, config):
    modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_modules = [f"encoder.{name}" for name, module in model.encoder.named_modules() if len(list(module.children())) == 0 and any([n in name for n in modules])]
    ## train the whole decoder
    modules_to_save = [ f"decoder.{name}" for name, module in model.decoder.named_modules() if len(list(module.children())) == 0]
    ## and the lm_head
    modules_to_save += ["lm_head"]
    # lora_modules = modules
    # modules_to_save = None
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=lora_modules,
        modules_to_save=modules_to_save,
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, config)
    return model

def do_data_reverse(pairs, example):
    direc_tasks = ["ape"]
    direc_data_names = ["wmt19_robustness", "wmt20_robustness"]
    source_lang, target_lang, task_type, data_name = example["src_lang"], example["tgt_lang"], example["task_type"],  example["data_name"]
    flag = True
    if f"{target_lang}-{source_lang}" not in pairs or task_type in direc_tasks:
        flag = False
    # exclude general_trans data
    if task_type == "general_trans" and data_name in direc_data_names:
        flag = False
    # exclude some special fewshot task
    if task_type == "context_learning_trans" and example["meta_task"] in direc_tasks:
        flag = False
    return flag

def do_data_reverse_old(pairs, examples, i):
    direc_tasks = ["ape", "context_aware_trans"]
    direc_data_names = ["wmt19_robustness", "wmt20_robustness", "IdiomsInCtx-MT","PETCI"]
    source_lang, target_lang, task_type, data_name = examples["src_lang"][i], examples["tgt_lang"][i], examples["task_type"][i],  examples["data_name"][i]
    flag = True
    if f"{target_lang}-{source_lang}" not in pairs or task_type in direc_tasks:
        flag = False
    # exclude general_trans data
    if task_type == "general_trans" and data_name in direc_data_names:
        flag = False
    # exclude some special fewshot task
    if task_type == "context_learning_trans" and examples["meta_task"][i] in direc_tasks:
        flag = False
    return flag

def process_mmt_data_for_seq2seq(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer, shots_eval_dict, data_args, training_args, model_args):

    def tokenize_train_eval_for_seq2seq(examples):
        inputs, targets = [], []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:                
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                inputs.append(prompt)
                targets.append(tgt_txt)
            if do_data_reverse(pairs, example):
                prompt, tgt_txt = get_prompt(target_lang, source_lang, example)
                inputs.append(prompt)
                targets.append(tgt_txt)
        # print(("\n\n"+"="*100+"\n\n").join([f"{x}\n{y}" for x,y in zip(inputs, targets)]))
        
        # add_special_tokens is not matter for the source
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=False, truncation=True)
        labels = tokenizer(text_target=targets, max_length=data_args.max_target_length, padding=False, truncation=True)
        check_add_eos(labels, tokenizer)
        model_inputs["labels"] = labels["input_ids"]
    
        return model_inputs

    def tokenize_test_for_seq2seq(examples):
        prompts = []
        targets = []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                # prompt, tgt_txt = get_prompt_direct(source_lang, target_lang, examples, i, shots_eval_dict, data_args.use_target_lang_prompt_eval, architecture=model_args.architecture)
                prompts.append(prompt)
                targets.append(tgt_txt)
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length, padding=False, truncation=True)
    
        return model_inputs

    train_datasets, eval_datasets, test_datasets = None, None, None
    
    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                processed_datasets.append(train_dataset)   
        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_datasets[lg_pair] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test_for_seq2seq,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets[lg_pair][task] = test_dataset
    
    return train_datasets, eval_datasets, test_datasets

def process_mmt_data_for_llm(train_raw_data, valid_raw_data, test_raw_data, pairs, tokenizer,  data_args, training_args):

    def tokenize_train_eval_left_pad(examples):
        prompts, inputs = [], []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                prompts.append(prompt)
                inputs.append(prompt + tgt_txt)
            # exclude some special tasks and dataset
            if do_data_reverse(pairs, example):
                prompt, tgt_txt = get_prompt(target_lang, source_lang, example)
                prompts.append(prompt)
                inputs.append(prompt + tgt_txt)
        # print(("\n\n"+"="*100+"\n\n").join(inputs)) # check data
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length + data_args.max_new_tokens - 1, truncation=True, add_special_tokens=True)
        check_add_eos(model_inputs, tokenizer)
        labels = copy.deepcopy(model_inputs["input_ids"])

        if data_args.ignore_prompt_token_for_loss:
            for idx, prompt in enumerate(prompts):
                prompt = tokenizer(prompt, max_length=data_args.max_source_length, add_special_tokens=False).input_ids
                labels[idx][: len(prompt)] = [-100] * len(prompt) 
        model_inputs["labels"] = labels
        return model_inputs
  
    def tokenize_test(examples):
        prompts, targets = [], []
        examples = [{key: value for key, value in zip(examples.keys(), values)} for values in zip(*examples.values())]   
        for example in examples:
            source_lang, target_lang = example["src_lang"], example["tgt_lang"]
            if f"{source_lang}-{target_lang}" in pairs:
                prompt, tgt_txt = get_prompt(source_lang, target_lang, example)
                # prompt, tgt_txt = get_prompt_direct_llm(source_lang, target_lang, example)
                prompts.append(prompt)
                targets.append(prompt + tgt_txt)
        model_inputs = tokenizer(prompts, max_length=data_args.max_source_length,  truncation=True, add_special_tokens=False)
        
        return model_inputs

    train_datasets, eval_datasets, test_datasets = None, None, None

    if training_args.do_train:
        processed_datasets = []
        for lg_pair, sub_raw_data in train_raw_data.items():
            for task, task_data in sub_raw_data.items():
                train_dataset = task_data["train"]
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))
                with training_args.main_process_first(desc="train dataset map pre-processing"):
                    train_dataset = train_dataset.map(
                        tokenize_train_eval_left_pad,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=train_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on MMT train dataset",
                    )
                    processed_datasets.append(train_dataset)

        train_datasets = concatenate_datasets(processed_datasets)
        train_datasets = train_datasets.shuffle(seed=training_args.seed)
        
    if training_args.do_eval:
        processed_datasets = []
        for lg_pair, sub_raw_data in valid_raw_data.items():
            for task, task_data in sub_raw_data.items():
                eval_dataset = task_data["validation"]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                with training_args.main_process_first(desc="validation dataset map pre-processing"):
                    eval_dataset = eval_dataset.map(
                        tokenize_train_eval_left_pad,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=eval_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer valid dataset",
                    )
                processed_datasets.append(eval_dataset)
        eval_datasets = concatenate_datasets(processed_datasets)
        eval_datasets = eval_datasets.shuffle(seed=training_args.seed)

    if training_args.do_predict:
        test_datasets = {}
        for lg_pair, sub_raw_data in test_raw_data.items():
            test_datasets[lg_pair] = {}
            for task, task_data in sub_raw_data.items():
                test_dataset = task_data["test"]
                if data_args.max_test_samples is not None:
                    max_test_samples = min(len(test_dataset), data_args.max_test_samples)
                    test_dataset = test_dataset.select(range(max_test_samples))
                with training_args.main_process_first(desc="test dataset map pre-processing"):
                    test_dataset = test_dataset.map(
                        tokenize_test,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=test_dataset.column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer test dataset",
                    )
                test_datasets[lg_pair][task] = test_dataset
    
    return train_datasets, eval_datasets, test_datasets

