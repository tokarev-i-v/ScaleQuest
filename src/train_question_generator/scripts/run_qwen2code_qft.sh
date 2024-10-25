# Step 1: QFT
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file scripts/zero3.yaml \
--main_process_port 29500 \
qft_train/train.py \
    --model_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset_path /path/to/CodeFeedback-Filtered-Instruction \
    --prompt_type qwen2.5-code \
    --num_train_epochs 1 \
    --gradient_checkpointing false \
    --max_length 256 \
    --max_training_samples 20000 \
    --output_dir models/Qwen2.5-Coder-7B-QFT \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \

# Step 2: QPO
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file ./zero3.yaml \
--main_process_port 29051 \
train.py \
    --model_path models/Qwen2.5-Coder-7B-QFT \
    --ref_model models/Qwen2.5-Coder-7B-QFT \
    --dataset_path /path/to/qpo_data \
    --prompt_type qwen2.5-code \
    --run_name qwen2-code-qgen-sft-dpo \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --loss_type sigmoid \
    --warmup_steps 20 \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --max_length 1024 \
    --output_dir models/Qwen2-Coder-7B-QGen \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \