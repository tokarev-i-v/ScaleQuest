# Step 1: QFT
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29600 \
qft_train/train.py \
    --model_path deepseek-ai/deepseek-coder-6.7b-instruct \
    --dataset_path /path/to/CodeFeedback-Filtered-Instruction \
    --prompt_type deepseek-code \
    --num_train_epochs 1 \
    --gradient_checkpointing false \
    --max_length 256 \
    --output_dir models/Deepseek-Coder-7B-QFT \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \

Step 2: QPO
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29601 \
train.py \
    --model_path models/Deepseek-Coder-7B-QFT \
    --ref_model models/Deepseek-Coder-7B-QFT \
    --dataset_path /path/to/qpo_data \
    --prompt_type deepseek-code \
    --run_name deepseek-code-qgen-sft-dpo \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --loss_type sigmoid \
    --warmup_steps 20 \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --max_length 1024 \
    --output_dir models/Deepseek-Coder-7B-QGen \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \