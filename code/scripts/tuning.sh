export GLOO_SOCKET_IFNAME=eth0
export WANDB_MODE=disabled

MAXLEN=8192  # default 4096
EPOCH=3
SAVEINTERVAL=3
LR=$(awk "BEGIN {print 1e-6 * sqrt($NODE_NUM)}")
BETA=0.01
LR_TYPE=linear

ISLORA=$1  # 1 for lora, 0 for full
RLType=$2  # ipo or sigmoid
ROOTPATH=$3
M_TYPE=qwen  # default llama, llama / qwen / baichuan2

if [ ${M_TYPE} == "qwen" ]
then
    pip3 install --upgrade tiktoken transformers_stream_generator==0.0.4
fi

raw_model_path=${ROOTPATH}/model/your_model/
train_data_path=${ROOTPATH}/data/your_train_data.jsonl
eval_data_path=${ROOTPATH}/data/your_eval_data.jsonl
deepspeed_config_path=${ROOTPATH}/code/configs/ds_config.json

case ${raw_model_path} in 
    *"7b"*)
        PER_GPU_BATCH=4
        GRA_ACC=4
        ;;
    *"13b"*)
        PER_GPU_BATCH=1
        GRA_ACC=16
        ;;
    *"72b"*)
        PER_GPU_BATCH=1
        GRA_ACC=16
        ;;
esac

TRAINDATANUM=$(wc -l < "${train_data_path}")
SAVESTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * ${SAVEINTERVAL} * $NODE_NUM)) + 1}")
TOTALSTEP=$(awk "BEGIN {print int(${TRAINDATANUM} * ${EPOCH} / (${PER_GPU_BATCH} * ${GRA_ACC} * $GPU_NUM_PER_NODE * $NODE_NUM)) + 1}")

if [ $ISLORA -ne 0 ]
then
    model_output_path=${ROOTPATH}/output/${RLType}_peft_${M_TYPE}/
    final_model_output_path=${ROOTPATH}/output/${RLType}_${M_TYPE}-lora/
    EVALSTEP=100
else
    model_output_path=${ROOTPATH}/output/${RLType}_${M_TYPE}-full/
    EVALSTEP=100
fi

echo "We use $NODE_NUM nodes to train with ${TRAINDATANUM} samples for ${EPOCH} epochs, resulting in ${TOTALSTEP} running steps, and thus we will save checkpoints every ${SAVESTEP} steps."

# training
torchrun --nnodes=$NODE_NUM \
    --node_rank=$INDEX \
    --nproc_per_node $GPU_NUM_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    ${ROOTPATH}/code/codes/train_dpo.py \
    --model_name_or_path ${raw_model_path} \
    --bf16 True \
    --output_dir ${model_output_path} \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size ${PER_GPU_BATCH} \
    --gradient_accumulation_steps ${GRA_ACC} \
    --save_strategy "steps" \
    --save_steps ${SAVESTEP} \
    --save_total_limit 2 \
    --per_device_eval_batch_size ${PER_GPU_BATCH} \
    --evaluation_strategy "steps" \
    --eval_steps ${EVALSTEP} \
    --learning_rate ${LR} \
    --log_level "info" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type ${LR_TYPE} \
    --deepspeed ${deepspeed_config_path} \
    --tf32 True \
    --model_max_length ${MAXLEN} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --gradient_checkpointing True \
    --report_to "none" \
    --loss_type ${RLType} \
    --if_lora ${ISLORA} \
    --beta ${BETA} \
    --model_type ${M_TYPE}

if  [ $ISLORA -ne 0 ]
then
    # merge lora and base model
    python3 ${ROOTPATH}/code/codes/merge_peft_adapter.py \
        --adapter_model_name ${model_output_path} \
        --base_model_name ${raw_model_path} \
        --output_name ${final_model_output_path}
fi
