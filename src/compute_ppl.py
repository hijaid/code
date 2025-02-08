from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss
import argparse
import sys
import tqdm


def compute_ppl(model, tokenizer, all_text, max_length=2048, stride=512):
    encodings = tokenizer(all_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    device = next(model.parameters()).device

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm.tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
    
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def main():

    parser = argparse.ArgumentParser(description="Script with conditional parameters")
    
    parser.add_argument('--model_path', type=str, help='The path to the model', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--file', type=str, help='plain text')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto").to("cuda")
    all_text = open(args.file).read().strip()[:10000]
    ppl = compute_ppl(model, tokenizer, all_text, args.batch_size)
    print(ppl)


if __name__ == "__main__":
    main()