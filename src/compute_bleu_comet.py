# coding=utf8
import os
import pandas as pd
from tqdm import tqdm
import subprocess
import json
import shutil
from collections import defaultdict
import argparse
import datetime
from datetime import datetime
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import  load_from_checkpoint


def bleu_scoring(hypo_file, lp):
    src, tgt = lp.split("2")
    data = [json.loads(x) for x in open(hypo_file) if x.strip()]
    hypos = [x["predict"].replace("\n", " ") for x in data]
    refs = [x["label"].replace("\n", " ") for x in data]
    ref_file = "ref.txt"
    hypo_file = "hypo.txt"
    with open('ref.txt', 'w', encoding="utf-8") as f:
        for line in refs:
            f.write(line+"\n")
    with open('hypo.txt', 'w', encoding="utf-8") as f:
        for line in hypos:
            f.write(line+"\n")
    langpair = f"{src}-{tgt}"
    command = f"sacrebleu -w 2 -b {ref_file} -i {hypo_file} -l {langpair}"
    print(command)
    score = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    os.remove(ref_file)
    os.remove(hypo_file)
    return score.stdout.strip()


def comet22_scoring(test_file, hypo_file, model, lp):
    srcs_data = [json.loads(x) for x in open(test_file , encoding='utf-8') if x.strip()] 
    preds_data = [json.loads(x) for x in open(hypo_file , encoding='utf-8') if x.strip()]
    src_lang,tgt_lang = lp.split("2") 
    srcs = [x["translation"][src_lang] for x in srcs_data]
    refs = [x["label"] for x in preds_data]
    preds = [x["predict"] for x in preds_data]
    assert len(srcs) == len(refs) == len(preds)

    data = [{"src":x, "mt":y, "ref":z} for x,y,z in zip(srcs, preds, refs)]
    print(f"comet22\ntest_file: {test_file}\nhypo_file: {hypo_file}")
    model_output = model.predict(data, batch_size=64, gpus=1)
    score = round(model_output[1]*100, 2)
    return score

def xcomet_scoring(test_file, hypo_file, model, lp):
    srcs_data = [json.loads(x) for x in open(test_file , encoding='utf-8') if x.strip()] 
    preds_data = [json.loads(x) for x in open(hypo_file , encoding='utf-8') if x.strip()]
    src_lang,tgt_lang = lp.split("2")
    
    srcs = [x["translation"][src_lang] for x in srcs_data]
    preds = [x["predict"] for x in preds_data]
    assert len(srcs) == len(preds)

    data = [{"src":x, "mt":y} for x,y in zip(srcs, preds)]
    print(f"xcomet\ntest_file: {test_file}\nhypo_file: {hypo_file}")
    model_output = model.predict(data, batch_size=16, gpus=1)
    score = round(model_output[1]*100, 2)
    return score


def main():
    parser = argparse.ArgumentParser(description="Script with conditional parameters")
    parser.add_argument('--metric', type=str, help='The evaluate metric', default="bleu,comet_22,xcomet_xxl")
    parser.add_argument('--comet_22_path', type=str, default='../model_card/wmt22-comet-da/checkpoints/model.ckpt')
    parser.add_argument('--xcomet_xl_path', type=str, default='/mnt/luoyingfeng/model_card/XCOMET-XL/checkpoints/model.ckpt')
    parser.add_argument('--xcomet_xxl_path', type=str, default='/mnt/luoyingfeng/model_card/XCOMET-XXL/checkpoints/model.ckpt')
    parser.add_argument('--lang_pair', type=str, help='en2de')
    parser.add_argument('--test_file', type=str, help='plain text')
    parser.add_argument('--hypo_file', type=str, help='plain text')
    parser.add_argument('--record_file', type=str)
    parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5", help='plain text')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    metrics = args.metric.split(",")

    if "comet_22" in metrics:
        comet_22_model = load_from_checkpoint(args.comet_22_path)
    if "xcomet_xl" in metrics:
        comet_xl_model = load_from_checkpoint(args.xcomet_xl_path)
    if "xcomet_xxl" in metrics:
        comet_xxl_model = load_from_checkpoint(args.xcomet_xxl_path)
    
    test_file = args.test_file
    hypo_file = args.hypo_file
    lp = args.lang_pair
    
    result = {}
    for metric in metrics:
        if metric == "bleu":
            score = bleu_scoring(hypo_file, lp)
            result[metric] = score
        
        if metric == "comet_22":
            score = comet22_scoring(test_file, hypo_file, comet_22_model, lp)
            result[metric] = score
        
        if metric == "xcomet_xl":
            score = xcomet_scoring(test_file, hypo_file, comet_xl_model)
            result[metric] = score
        
        if metric == "xcomet_xxl":
            score = xcomet_scoring(test_file, hypo_file, comet_xxl_model)
            result[metric] = score
    print(result)
    with open(args.record_file,"a",encoding="utf-8") as f:
        formatted_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write("============="+str(formatted_datetime)+"===================\n")
        f.write(test_file+"\n")
        f.write(hypo_file+"\n")
        f.write("sacre_bleu: "+result["bleu"]+"\n")
        f.write("comet_score: "+str(result["comet_22"])+"\n")


#     =============2025-01-14 15:37:20===================
# /mnt/luoyingfeng/model_card/Meta-Llama-3.1-8B
# /mnt/luoyingfeng/effllm/data/fine-tuning_data/common/is-en/test.en2is.is
# /mnt/luoyingfeng/effllm/exps/en2is/zeroshot/Meta-Llama-3.1-8B.hypo
# sacre_bleu: 6.64
# comet_score: 63.76

if __name__ == '__main__':
    main()

