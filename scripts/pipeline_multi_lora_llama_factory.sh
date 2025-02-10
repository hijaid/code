#! /bin/bash
set -eux
ROOT_DIR=$(dirname $(dirname `readlink -f $0`))

# model
for model_name in Qwen2.5-7B Qwen2.5-1.5B Qwen2.5-3B;do
template=default
dropout=0.05
model_dir=$ROOT_DIR/model_card/$model_name
for rank in 128;do
    for learning_rate in 5e-5; do
        task=lora_multi_${rank}_${learning_rate}_${dropout}_2epoch_attention_MLP
        # train_stage
        # data
        dataset_dir=$ROOT_DIR/data/fine-tuning_data/multi_llama_factory
        train_data=train-multi-high
        eval_dataset=valid-multi-high

        config_file=./configs/ds_z2_config.json

        output_dir=$ROOT_DIR/exps/$model_name/$task/adapter
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
            --gradient_accumulation_steps 4 \
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




        log_path=$ROOT_DIR/exps/$model_name/$task/adapter/train.log

        folder=$ROOT_DIR/exps/$model_name/$task/adapter

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
        predict_model_dir=$ROOT_DIR/exps/$model_name/$task/adapter/checkpoint-$predict_model_id


        for lan in de cs ru zh;do
            for src in $lan en ;do
            # predict_stage

            template=$template
            dataset_dir=$ROOT_DIR/data/fine-tuning_data/multi_llama_factory
            if [ $src = "en" ]; then
			    test_dataset=test-${src}-${lan}
                lp=en2${lan}
		    else 
			    test_dataset=test-${lan}-en
                lp=${lan}2en
		    fi

            output_dir=$predict_model_dir/decode_result/$lp
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

            test_file=$ROOT_DIR/data/fine-tuning_data/common/$lan-en/test.$lp.json
            hypo_file=$predict_model_dir/decode_result/$lp/generated_predictions.jsonl
            record_file=./multi_lora_eval_result.txt

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
