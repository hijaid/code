#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

# model
for model_name in Qwen2.5-7B Qwen2.5-1.5B Qwen2.5-3B;do
template=default
model_dir=$ROOT_DIR/model_card/$model_name
for dropout in 0.05;do
for rank in 128;do
    for learning_rate in 5e-5; do
    for l in de cs ru zh he fi kk is ; do
	    for src in $l en; do 
        if [ $src = "en" ]; then
			tgt=$l
		else 
			tgt=en
		fi
        task=lora_${rank}_${learning_rate}_${dropout}_2epoch_attention_MLP
        lang_pair=${src}-$tgt
		lp=${src}2${tgt}
        tag=$lang_pair
         # train_stage
         # data
        dataset_dir=$ROOT_DIR/data/fine-tuning_data/multi_llama_factory
        train_data=train-$lang_pair
        eval_dataset=valid-$lang_pair

        config_file=./configs/ds_z2_config.json

        output_dir=$ROOT_DIR/exps/$model_name/$task/$tag/adapter
        mkdir -p $output_dir
        cp $0 $output_dir

        llamafactory-cli train \
            --deepspeed  $config_file \
            --stage sft \
            --finetuning_type lora \
            --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
            --lora_rank $rank \
            --lora_alpha $((rank * 2)) \
            --lora_dropout $dropout \
            --model_name_or_path $model_dir \
            --dataset_dir $dataset_dir \
            --dataset $train_data \
            --eval_dataset $eval_dataset \
            --template $template \
            --cutoff_len  1024 \
            --do_train True \
            --do_eval True \
            --use_fast_tokenizer True \
            --learning_rate $learning_rate \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.01 \
            --num_train_epochs 2 \
            --per_device_train_batch_size 9 \
            --per_device_eval_batch_size 12 \
            --gradient_accumulation_steps 1 \
            --eval_steps 0.1 \
            --save_steps 0.1 \
            --num_beams 5 \
            --max_new_tokens 256 \
            --do_sample False \
            --logging_steps 0.01 \
            --dataloader_num_workers 8 \
            --preprocessing_num_workers 16 \
            --output_dir $output_dir \
            --overwrite_output_dir True \
            --save_only_model True \
            --bf16 True \
            --seed 42  \
            --cache_dir ./cache \
            --evaluation_strategy steps \
            --save_strategy steps \
            --logging_strategy steps \
            --report_to "tensorboard" \
            --ddp_timeout 180000000 \
            | tee $output_dir/train.log


        # merge_stage
        # 指定要查找的目录（例如当前目录）
        # search_dir=$ROOT_DIR/exps/$model_name/$task/$tag/adapter


        # predict_adapter_dir=$(find "$search_dir" -type d -name "checkpoint*" | \
        #  sed -E 's/.*checkpoint-([0-9]+).*/\1 &/' | \
        #  sort -n | \
        #  tail -n 1 | \
        #  awk -F/ '{print $NF}')

        # llamafactory-cli export \
        #     --model_name_or_path $model_dir \
        #     --adapter_name_or_path  $search_dir/$predict_adapter_dir \
        #     --template $template \
        #     --finetuning_type lora \
        #     --export_dir $ROOT_DIR/exps/$model_name/$task/$tag/$predict_adapter_dir \
        #     --export_size 2 \
        #     --export_device cpu \
        #     --export_legacy_format False
 



        # predict_stage
        log_path=$ROOT_DIR/exps/$model_name/$task/$lang_pair/adapter/train.log

        folder=$ROOT_DIR/exps/$model_name/$task/$lang_pair/adapter

        check_point_strs=""

        # 使用 find 命令查找满足条件的文件夹，并将结果存储在数组 folders 中
        folders=($(find "$folder" -type d -name "checkpoint*"))

        # 遍历数组中的每个文件夹
        for ((i = 0; i < ${#folders[@]}; i++)); do
            # 获取当前文件夹的名称（不包含路径）
            folder_name=${folders[$i]}
            # 如果结果字符串不为空，添加逗号分隔符
            if [[ -n "$check_point_strs" ]]; then
                check_point_strs="$check_point_strs,$folder_name"
            else
                check_point_strs="$folder_name"
            fi
        done

        predict_model_id=$(python3 ../src/get_best_checkpoint.py \
                --train_log_path $log_path \
                --check_point_strs $check_point_strs)
        predict_model_dir=$ROOT_DIR/exps/$model_name/$task/$tag/adapter/checkpoint-$predict_model_id
        # predict_model_dir=/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/lora_multi_16_1e-4_3epoch/adapter/checkpoint-1500
        template=$template
        dataset_dir=$ROOT_DIR/data/fine-tuning_data/multi_llama_factory
        if [ $src = "en" ]; then
			test_dataset=test-${src}-${l}
            lp=en2${l}
		else 
		    test_dataset=test-${l}-en
            lp=${l}2en
		fi

        output_dir=$predict_model_dir/decode_result
        mkdir -p $output_dir
        cp $0 $output_dir


        llamafactory-cli train \
            --model_name_or_path $ROOT_DIR/model_card/$model_name \
            --adapter_name_or_path $predict_model_dir \
            --finetuning_type lora \
            --infer_backend huggingface \
            --dataset_dir $dataset_dir \
            --eval_dataset $test_dataset \
            --template $template \
            --cutoff_len  1024 \
            --max_length 1024 \
            --max_new_tokens 256 \
            --do_train False \
            --do_eval False \
            --do_predict \
            --use_fast_tokenizer True \
            --per_device_eval_batch_size 8 \
            --predict_with_generate \
            --logging_steps 0.01 \
            --preprocessing_num_workers 16 \
            --output_dir $output_dir \
            --overwrite_output_dir True \
            --num_beams 5 \
            --do_sample False \
            --bf16 True \
            --seed 42  \
            --cache_dir ./cache \
            --logging_strategy steps \
            --dataloader_num_workers 8 \
            | tee $output_dir/train.log

        # eval_stage
        # folder=/mnt/luoyingfeng/lora4mt/exps/$model_name/$task/$tag/adapter

        # # 查找文件夹，过滤出包含"checkpoint"的文件夹，提取文件夹名中的数字部分，并返回数字最大的文件夹
        # predict_model_dir=$(find "$folder" -type d -name "checkpoint*" | \
        #  sed -E 's/.*checkpoint-([0-9]+).*/\1 &/' | \
        #  sort -n | \
        #  tail -n 1 | \
        #  awk '{print substr($0, index($0,$2))}')
        # predict_model_dir=/mnt/luoyingfeng/lora4mt/exps/Meta-Llama-3.1-8B/lora_multi_16_1e-4_3epoch/adapter/checkpoint-1500
        test_file=$ROOT_DIR/data/fine-tuning_data/common/$l-en/test.$lp.json
        hypo_file=$predict_model_dir/decode_result/generated_predictions.jsonl
        record_file=./lora_eval_result.txt

        python ../src/compute_bleu_comet.py \
            --metric "bleu,comet_22" \
            --lang_pair $lp \
            --test_file $test_file \
            --hypo_file $hypo_file \
            --record_file $record_file
        done
        done
	done
done
done
done