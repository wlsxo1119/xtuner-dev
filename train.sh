# Train
NPROC_PER_NODE=1 xtuner train qwen_sft.py --deepspeed deepspeed_zero1

# Convert Deepspeed Checkpoint to Hugging Face Format
xtuner convert pth_to_hf qwen_sft.py work_dirs/qwen_sft/iter_10129.pth ./qwen_sft_hf

# GPTQ Quantization
python quantization_gptq.py

# Inference
python inference.py

# hf model Convert to Safetensors
python convert_to_safetensor.py qwen-test

# GGUF 변환 && 4bit Q (Llama.cpp 사용)
sudo apt-get install libcurl4-openssl-dev
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .
make
python convert_hf_to_gguf.py [Hugging Face 모델 디렉터리 경로] --outfile [출력 파일 이름].gguf
bin/llama-quantize [GGUF 모델 파일].gguf [출력 파일 이름].gguf [양자화 유형]
q-type
    q4_0: 가장 기본적인 4비트 양자화 방식으로, 정확도는 낮지만 파일 크기가 가장 작습니다. 구형 방식이므로 최신 모델에서는 잘 사용되지 않습니다.
    q4_k_s (Small): q4_0보다 정확도가 개선된 4비트 양자화 방식입니다. 더 작은 모델에 적합합니다.
    q4_k_m (Medium): q4_k_s보다 정확도가 더욱 개선된 4비트 양자화 방식입니다. 현재 가장 널리 사용되는 4비트 양자화 방식 중 하나로, 크기와 정확도 사이에서 균형이 잘 잡혀 있습니다.
