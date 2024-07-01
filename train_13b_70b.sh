
source activate /path/to/env 

pwd=/path/to/MuMath-src 

MODEL_PATH=/path/to/Llama-2-70b-hf 
SAVE_PATH=/path/to/saved_model 

DATA_PATH=/path/to/train_data_file 

export WANDB_DISABLED=true 

GPUS_PER_NODE=8 

if [[ $IS_MASTER == 1 ]];then 
  DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
  DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

# 128 batch size, 2 * 2 * 8 * 4 == 128, 4 nodes  

torchrun $DISTRIBUTED_ARGS $pwd/train_llama2_70b.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --bf16 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --deepspeed $pwd/config/zero2_config_13b_70b.json 


python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl 
python eval_math.py --model $SAVE_PATH --data_path ./data/test/MATH_test.jsonl 


