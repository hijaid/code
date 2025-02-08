# coding=utf8

# pip install accelerate
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, LlamaModel, LlamaForSequenceClassification, AutoConfig,
    # LlamaForSeq2Seq,
    EncoderDecoderModel,
    LlamaConfig,
    EncoderDecoderConfig,
    BertConfig,
    # LlamaConcatAttentionED,
    LlamaModel,
    AutoModelForSeq2SeqLM
)
from tokenizers.processors import TemplateProcessing
import datasets
from datasets import load_dataset
import copy
import os
import re
import pickle

import torch
from safetensors import safe_open
from peft import LoraConfig, get_peft_model, AutoPeftModel

from utils import utils


def test_gemma():
    model_path = "/mnt/luoyingfeng/model_card/TowerBase-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    # model = AutoModelForCausalLM.from_pretrained("/mnt/luoyingfeng/model_card/gemma-2b", torch_dtype=torch.float16).to("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto").to("cuda")
    config = AutoConfig.from_pretrained(model_path)
    decoder_config = copy.deepcopy(config.to_dict())
    decoder_config["num_hidden_layers"] = 2
    config.decoder = decoder_config
    # config.use_cache = False
    # state_dict = make_state_dict("/mnt/luoyingfeng/model_card/LiteLlama-460M-1T")
    # model = LlamaForSeq2Seq.from_pretrained("/mnt/luoyingfeng/model_card/LiteLlama-460M-1T", config=config, state_dict=state_dict)
    # model = LlamaForSeq2Seq.from_pretrained("/mnt/luoyingfeng/model_card/LiteLlama-460M-1T", config=config).to("cuda")
    # model = LlamaForSeq2Seq.from_pretrained("/mnt/luoyingfeng/llm4mt/scripts/save_model")
    # print(model)
    # model.init_decoder_parameters()
    # print(model.model.layers[0])
    # for item in model.model.layers named_parameters():
    #     print(item[0])
    # for p1, p2 in zip(model.model.layers[-4].named_parameters(), model.decoder.model.layers[-4].named_parameters()):
    #     print(p1, p2)
    # exit()
    # model.save_pretrained("./save_model", from_pt=True)
    # exit()
    
    # input_text = "Write me a poem about Machine Learning."
    # input_text = input()
    # text1 = "在 Markdown 中插入代码块可以通过使用三个反引号 ``` 来实现，后跟编程语言的名称（可选）。代码块中的内容将会被渲染成代码格式"
    # text2 = "These integrations mark the first offerings we are launching together as a result of our collaborative partnership with Google. Stay tuned for more!"
    # print(tokenizer.tokenize(text1))
    input_ids = tokenizer(["hello", "who are you?"], return_tensors="pt", padding=True, truncation=True)
    # attention_mask = input_ids["attention_mask"]
    # input_ids["decoder_attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=1)
    # input_ids["decoder_input_ids"] = torch.full((attention_mask.shape[0], 1), tokenizer.bos_token_id)
    input_ids = input_ids.to("cuda")

    # input_ids = tokenizer("hello llama", return_tensors="pt", truncation=True).to("cuda")
    # print(input_ids["input_ids"], len(input_ids["input_ids"][0]))
    print(input_ids)
    
    # out = model(**input_ids)
    # print(out.keys())    
    generated_ids  = model.generate(**input_ids, max_new_tokens=100, num_beams=2, use_cache=False)
    # generated_ids = model.generate(**input_ids, max_new_tokens=80)
    gen_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(gen_text)
    # print(out)
    
    
def print_model_args():
    state = torch.load("/mnt/luoyingfeng/model_card/Qwen1.5-0.5B/model.safetensors")
    print("\n".join(state.keys()))
    # print(state["decoder.block.10.layer.1.layer_norm.weight"].size())


def test_tokenizer():
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/luoyingfeng/model_card/pythia-160m", padding_side='left', add_eos_token=True)
    # tokenizer.pad_token_id = 1
    # tokenizer.eos_token_id = 0
    # url = "/mnt/luoyingfeng/model_card/TinyLlama-1.1B-3T"
    # url = "/mnt/luoyingfeng/model_card/t5-base"
    # url = "/mnt/luoyingfeng/model_card/Qwen1.5-0.5B"
    # url = "/mnt/luoyingfeng/model_card/LiteLlama-460M-1T"
    # url = "/mnt/luoyingfeng/model_card/Llama-2-7b-hf"
    url = "/mnt/luoyingfeng/model_card/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(url)
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = 128002
    # print(tokenizer, len(tokenizer))
    # print(tokenizer.convert_ids_to_tokens([151643, 151644, 151645]))
    print(tokenizer.pad_token_id)
    print(tokenizer.eos_token_id)
    print(tokenizer.bos_token_id)
    # print(tokenizer.unk_token)
    print(tokenizer.convert_ids_to_tokens([0, 1, 2, 259]))

    # tokenizer._tokenizer.post_processor = TemplateProcessing(
    #     single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
    #     special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
    # )
    # print(tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("/mnt/luoyingfeng/model_card/gemma-2b", padding_side='left', add_eos_token=True)
    text1 = "在 Markdown 中插入代码块可以通过使用三个反引号 ``` 来实现，后跟编程语言的名称（可选）。代码块中的内容将会被渲染成代码格式\n"
    # text1 = "All 8 model sizes are trained on the exact same data, in the exact same order.\n"
    text2 = "These integrations mark the first offerings we are launching together as a result of our collaborative partnership with Google. Stay tuned for more!"
    print(tokenizer(text1))
    # # print(tokenizer.encode(text1))
    # print(tokenizer.tokenize([text1, text2]))
    # print(tokenizer.convert_ids_to_tokens(tokenizer(text1)["input_ids"]))
    # print(len(text1), len(tokenizer(text1)["input_ids"]))
    # print(tokenizer(text_target=[text1, text2], return_tensors='pt', truncation=True, padding=True))


def test_dataset():
    def preprocess_function(examples):
        inputs = [ex['en'] for ex in examples["translation"]]
        targets = [ex['ro'] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=1024, padding="max_length", truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=1024, padding="max_length", truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenizer = AutoTokenizer.from_pretrained("/mnt/luoyingfeng/model_card/gemma-2b", padding_side='left', add_eos_token=True)
    data_files = {
        # "train": '/mnt/luoyingfeng/llm4mt/data/ro-en/temp_data/train.jsonl',
        "validation": '/mnt/luoyingfeng/llm4mt/data/ro-en/temp_data/valid.jsonl',
        'test': '/mnt/luoyingfeng/llm4mt/data/ro-en/temp_data/test.jsonl'
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    column_names = raw_datasets["validation"].column_names

    # print(raw_datasets["test"]["translation"])
    processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=16,
            remove_columns=column_names,
        )
    print(processed_datasets)


def test_encoderdecoder():
    # config_encoder = LlamaConfig.from_pretrained("/mnt/luoyingfeng/model_card/TinyLlama-1.1B-3T")
    # config_decoder = copy.deepcopy(config_encoder)
    # config_decoder.num_hidden_layers = 4
    # config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    # model = EncoderDecoderModel(config=config)
    # print(config_encoder)
    # print(config_decoder)
    # print(model)
    config_encoder = BertConfig()
    config_decoder = BertConfig()
    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = EncoderDecoderModel(config=config)
    print(model)


def test_qwen():
    model_path = "/mnt/luoyingfeng/model_card/Qwen1.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16) # device_map="auto"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    # {"translation": {"en": "we go through initiation rites.", "de": "wir durchlaufen initiationsrituale."}}
    # {"translation": {"en": "they speared each other.", "de": "sie durchbohrten sich gegenseitig."}}

    prompt = ["Translate this from English to German:\nEnglish: we go through initiation rites.\nGerman:",
              "Translate this from English to German:\nEnglish: they speared each other.\nGerman:"]
    # input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda() 
    # with torch.no_grad():
    #     generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=False)
    #     outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #     print(outputs)

    input_ids = tokenizer(prompt[0], return_tensors="pt")
    output = model(**input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states
    print(len(hidden_states), hidden_states[0].squeeze().size())


def test_crossattention():
    url = "/mnt/luoyingfeng/model_card/Meta-Llama-3-8B" # bf16
    # url = "/mnt/luoyingfeng/model_card/TinyLlama-1.1B-3T"
    # url = "/mnt/luoyingfeng/model_card/Llama-2-7b-hf"  # fp16
    # url = "/mnt/luoyingfeng/model_card/ALMA-7B-Pretrain"
    src_tokenizer = AutoTokenizer.from_pretrained(url, padding_side="left", add_special_tokens=False)
    tgt_tokenizer = AutoTokenizer.from_pretrained(url, padding_side="right", add_special_tokens=False)
   
    src_tokenizer = utils.set_tokenizer_special_tokens(src_tokenizer, url)
    tgt_tokenizer = utils.set_tokenizer_special_tokens(tgt_tokenizer, url)

    use_cache = True
    torch_dtype = "auto"

    # src = ["hello, word!", "it's nice to meet you."]
    src = ["She loves to read books."] # She loves to read books. The sun is shining. He plays soccer every Saturday.
    tgt = ["你好，世界", "很高兴见到你"]
    src_input_ids = src_tokenizer(src, return_tensors="pt", padding=True).to("cuda")
    tgt_input_ids = tgt_tokenizer(tgt, return_tensors="pt", padding=True).to("cuda")
    # print(src_input_ids)
    # print(tgt_input_ids)
    input_ids = src_input_ids.copy()
    # input_ids.update({
    #     "decoder_input_ids": tgt_input_ids["input_ids"],
    #     "decoder_attention_mask": tgt_input_ids["attention_mask"]
    # })
   
    config = AutoConfig.from_pretrained(url)
    # print(config.rope_scaling)
    # exit()

    config.is_encoder_decoder = True
    config.decoder =  copy.deepcopy(config.to_dict())
    config.decoder_start_token_id = config.bos_token_id
    states = utils.make_model_state_dict(url, config=None, model_type="ConAttLW")
    # model = LlamaConcatAttentionED(config)
    # model.load_state_dict(states)
    model = LlamaConcatAttentionED.from_pretrained(None, config=config, state_dict=states, torch_dtype=torch_dtype).to("cuda")
    model.set_share_paremeters()
    model = utils.set_model_special_tokens(model, url)
    batch_size = src_input_ids["input_ids"].size(0)
    dtype = src_input_ids["input_ids"].dtype
    device = src_input_ids["input_ids"].device
    
    add_forward = {
        "decoder_input_ids": torch.full((batch_size, 1), config.bos_token_id, device=device, dtype=torch.int64),
        "decoder_attention_mask": torch.ones((batch_size, 1), device=device, dtype=torch.int64)
    }
    input_ids.update(add_forward)

    output = model(**input_ids)
    logits = output.logits[0, -1, :]
    print(logits[:100])
    print(input_ids)
    # with torch.no_grad():
    #     model.eval()
    #     generated_ids = model.generate(**input_ids, max_new_tokens=100, num_beams=1, do_sample=False, use_cache=use_cache)
    # # print(generated_ids)
    # gen_text = src_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    # print(gen_text)

    # llama_tokenizer = AutoTokenizer.from_pretrained(url, add_special_tokens=False)
    config = AutoConfig.from_pretrained(url)
    model = AutoModelForCausalLM.from_pretrained(url, config=config, torch_dtype=torch_dtype).to("cuda")
    model = utils.set_model_special_tokens(model, url)
    # model = AutoModel.from_pretrained(url).to("cuda")
    # input_ids = llama_tokenizer([x+y for x,y in zip(src, tgt)], return_tensors="pt", padding=True).to("cuda")


    add_tokens = torch.full((batch_size, 1), config.bos_token_id, device=device, dtype=dtype)

    new_input_ids = {
        "input_ids": torch.cat([src_input_ids["input_ids"], add_tokens], dim=-1),
        "attention_mask": torch.cat([src_input_ids["attention_mask"], torch.ones((batch_size, 1), device=device, dtype=torch.int64)], dim=-1)
    }
    # print(input_ids)
    output = model(**new_input_ids)
    logits = output.logits[0, -1, :]
    print(logits[:100])
    print(new_input_ids)
    # with torch.no_grad():
    #     model.eval()
    #     generated_ids = model.generate(**new_input_ids, max_new_tokens=100, num_beams=1, do_sample=False, use_cache=use_cache)
    # # print(generated_ids)
    # gen_text = src_tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    # print(gen_text)
    # hidden = output.last_hidden_state[:, :src_input_ids["input_ids"].size(1), :]
    # print(hidden.size(), hidden)


def test_parameter_share():
    url = "/mnt/luoyingfeng/model_card/Meta-Llama-3-8B"
    config = AutoConfig.from_pretrained(url)
    # print(config.rope_scaling)
    # exit()

    decoder_config = copy.deepcopy(config.to_dict())
    decoder_config["num_hidden_layers"] = 6
    decoder_config["decoder_param_method"] = "share"
    decoder_config["model_method"] = "ConAttT2B"
    config.decoder =  decoder_config
    ## setting encoder config
    config.use_cache = False
    config.is_encoder_decoder = True
    config.decoder_start_token_id = config.bos_token_id
    ## construct state dict to load model
    state_dict = utils.make_model_state_dict(model_path=url, config=config, model_method="ConAttT2B")
    model = AutoModelForCausalLM.from_pretrained(url)
    print(sum(p.numel() for p in model.parameters()))
    model = LlamaConcatAttentionED.from_pretrained(None, config=config, state_dict=state_dict)
    print(sum(p.numel() for p in model.parameters()))
    model.set_share_paremeters()
    print(sum(p.numel() for p in model.parameters()))


def test_models_function():
    url = "/mnt/luoyingfeng/model_card/TowerBase-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(url)
    tokenizer.pad_token_id = 0
    src = ["hello, word!", "it's nice to meet you."]
    input_ids = tokenizer(src, return_tensors="pt", padding=True)
    print(input_ids)
    model = LlamaModel.from_pretrained(url)
    res = model(**input_ids)


def test_lora():
    url = "/mnt/luoyingfeng/llm4nmt/exps/TinyDecoder/TowerBase-7B-v0.1_TinyConAttLW_d8_dim1024_stage1/checkpoint-41000"
    # config = AutoConfig.from_pretrained(url)
    # config.is_encoder_decoder = True
    # decoder_config =  copy.deepcopy(config.to_dict())
    # config.decoder_start_token_id = config.bos_token_id
    # decoder_config["num_hidden_layers"] = 8
    # decoder_config["num_encoder_layers"] = 32
    # decoder_config["decoder_param_method"] = "freeze"
    # decoder_config["model_method"] = "ConAttLW"
    # decoder_config["hidden_size"] = 1024
    # decoder_config["intermediate_size"] = 2752
    # decoder_config["num_attention_heads"] = 16
    # decoder_config["num_key_value_heads"] = 16
    # config.decoder =  decoder_config
    # states = utils.make_model_state_dict(url, config=None, model_type="ConAttLW")
    # model.set_share_paremeters()
    # lora_modules = ["encoder.*q_proj", "encoder*v_proj", "encoder*k_proj", "encoder*o_proj", "encoder*gate_proj", "encoder*up_proj", "encoder*down_proj",]
    model = LlamaConcatAttentionED.from_pretrained(url)
    print(model)

    modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_modules = [f"encoder.{name}" for name, module in model.encoder.named_modules() if len(list(module.children())) == 0 and any([n in name for n in modules])]
    # lora_modules = modules
    # print(lora_modules)
    # modules_to_save = [ f"decoder.{name}" for name, module in model.decoder.named_modules() if len(list(module.children())) == 0]
    # modules_to_save += ["lm_head"]
    modules_to_save = None
    print(modules_to_save)
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
    print(model)
    model.print_trainable_parameters()
    # encoder_params_lora = [p for n, p in model.named_parameters() if 'lora' in n]
    # print(sum(p.numel() for p in encoder_params_lora))
    # decoder_params = [p for n, p in model.named_parameters() if 'decoder' in n or "lm_head" in n and p.requires_grad]
    # encoder_params = [n for n, p in model.named_parameters() if 'lora' in n]
    # decoder_params = [n for n, p in model.named_parameters() if 'decoder' in n or "lm_head" in n and p.requires_grad]
    # lora_encoder = get_peft_model(model.encoder, config)
    # model.encoder = lora_encoder
    for name, param in model.named_parameters():
        if "decoder" in name or "lm_head" in name:
            param.requires_grad = True
    model.print_trainable_parameters()
    # # print(encoder_params)
    # # print(decoder_params)
    # print(model)
    # print(sum(p.numel() for p in model.decoder.parameters()))
    # model.save_pretrained("./temp1")
    # model.print_trainable_parameters()
   
    # model = AutoPeftModel.from_pretrained("./temp")

    # model_encoder = model.encoder.merge_and_unload()
    # print(model_encoder)
    # print(model)
    model.save_pretrained("./temp1")
    # model = LlamaConcatAttentionED.from_pretrained("./temp1")
    # print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

def lora_model_merge():
    in_url = "/mnt/luoyingfeng/llm4nmt/exps/Lora/TowerBase-7B-v0.1_TinyCrossAttLW_d8_dim1024_lora64/checkpoint-24000"
    out_url = "/mnt/luoyingfeng/llm4nmt/exps/Lora/TowerBase-7B-v0.1_TinyCrossAttLW_d8_dim1024_lora64/checkpoint-24000-merge"
    model = AutoPeftModel.from_pretrained(in_url)
    model = model.merge_and_unload()
    model.save_pretrained(out_url)


def test_qwen2_instuct():
    device = "cuda" # the device to load the model onto

    # print(input_id)

    model_name_or_path = "/mnt/luoyingfeng/model_card/Qwen2-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompt = "帮我写一首五言绝句"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt")

    # model_inputs = tokenizer([text], return_tensors="pt").to(device)
   
    

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)



if __name__ == "__main__":
    # test_gemma()
    # print_model_args()
    # test_tokenizer()
    # test_dataset()
    # test_encoderdecoder()
    # test_qwen()
    # test_crossattention()
    # test_parameter_share()
    # test_models_function()
    # test_lora()
    # lora_model_merge()

    tokenizer = AutoTokenizer.from_pretrained("/mnt/luoyingfeng/model_card/nllb-200-3.3B", use_auth_token=True, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("/mnt/luoyingfeng/model_card/nllb-200-3.3B", use_auth_token=True)

    article = "Police arrest 15 after violent protest outside UK refugee hotel"
    inputs = tokenizer(article, return_tensors="pt", add_special_tokens=False)

    translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30)
    res = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(res)



        