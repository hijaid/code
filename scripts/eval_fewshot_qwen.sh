#! /bin/bash

set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))
export HF_HOME="./cache/"
export HF_DATASETS_CACHE="./cache/huggingface_cache/datasets"
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export NCCL_P2P_LEVEL=NVL
config_file=./configs/accelerate_config_2gpu.yaml


for model_name in Qwen2.5-32B ; do
for l in de cs ru zh he fi kk is; do
# for l in ru; do
for src in en $l; do
	if [ $src = "en" ]; then
		tgt=$l
	else 
		tgt=en
	fi

	eval_mode=fewshot
	shot=3
    lang_pair=${src}-$tgt
	lp=${src}2${tgt}
	test_file=$ROOT_DIR/data/fine-tuning_data/common/${l}-en/test.$lp.json
	few_shot_file=$ROOT_DIR/data/fine-tuning_data/common/${l}-en/valid.json
	src_file=$ROOT_DIR/data/fine-tuning_data/common/${l}-en/test.$lp.$src
	ref_file=$ROOT_DIR/data/fine-tuning_data/common/${l}-en/test.$lp.$tgt
	save_dir=$ROOT_DIR/exps/base_model/common/$lp/3-shot
	hypo_file=$save_dir/$model_name.hypo



	mkdir -p $save_dir
	cp $0 $save_dir
	model_path=$ROOT_DIR/model_card/$model_name
	
	accelerate launch --config_file $config_file --main_process_port 29501 $ROOT_DIR/src/eval_fewshot.py \
		--model_name_or_path $model_path \
		--test_file $test_file \
		--few_shot_file $few_shot_file \
		--res_file $hypo_file \
		--lang_pair $lp \
		--eval_mode $eval_mode \
		--shot $shot \
		--num_batch 16 \
    --max_new_tokens 256

done
done

done
