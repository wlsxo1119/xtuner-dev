# Train
NPROC_PER_NODE=1 xtuner train qwen_sft.py --deepspeed deepspeed_zero1

# Convert Deepspeed Checkpoint to Hugging Face Format
xtuner convert pth_to_hf qwen_sft.py work_dirs/qwen_sft/iter_10129.pth ./qwen_sft_hf

# Quantization
python quantization_gptq.py

# Inference
python inference.py