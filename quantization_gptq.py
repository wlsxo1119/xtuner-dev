from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import random
import torch
import os

# --- 1. 모델 및 경로 설정 ---
# 파인튜닝된 원본 모델의 로컬 경로
model_id = "qwen_sft_hf_3epoch"
# 양자화된 모델을 저장할 새로운 경로
quantized_model_dir = "./qwen_sft_hf_3epoch_4bit_gptq"

print(f"원본 모델 경로: {model_id}")
print(f"양자화된 모델 저장 경로: {quantized_model_dir}")

# --- 2. 캘리브레이션 데이터 준비 ---
calib_data_path = 'calib/calib_data.txt'
if not os.path.exists(calib_data_path):
    raise FileNotFoundError(f"캘리브레이션 데이터 파일이 없습니다: {calib_data_path}")

calib_data = []
with open(calib_data_path, 'r', encoding='utf-8') as f:
    # 빈 줄이나 공백만 있는 줄은 제외하고 읽어옵니다.
    calib_data = [line.strip() for line in f if line.strip()]
random.shuffle(calib_data)  # 데이터 섞기
#calib_data = calib_data[:1000]
print(f"캘리브레이션 데이터 로드 완료: {len(calib_data)} 줄")

# --- 3. GPTQ 양자화 설정 ---
# tokenizer를 미리 로드하여 GPTQConfig에 직접 전달하는 것이 더 안정적입니다.
tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    desc_act=False,  # ### --- 수정된 부분 (1): 오타 수정 (desec_act -> desc_act) --- ###
    damp_percent=0.01,
    dataset=calib_data, # 캘리브레이션 데이터는 문자열 리스트로 전달합니다.
    tokenizer=tokenizer, # 미리 로드한 토크나이저 객체를 전달합니다.
    model_seqlen=512,
    max_input_length=512,
)

# --- 4. 양자화 실행 및 모델 로드 ---
# from_pretrained를 호출하는 이 시점에 캘리브레이션과 양자화가 모두 진행됩니다.
# 이 함수가 완료되면 model 변수에는 이미 양자화된 모델이 할당됩니다.
print("양자화를 시작하며 모델을 로드합니다... (시간이 소요될 수 있습니다)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    max_memory={0: "10GiB", "cpu": "10GiB"}
)
print("양자화 및 모델 로드 완료!")

# --- 5. 양자화된 모델 저장 (매우 중요) --- ###
# ### --- 추가된 부분 (3): 양자화된 모델을 파일로 저장하여 나중에 재사용 --- ###
# 양자화는 메모리 상에서만 이루어졌으므로, 파일로 저장해야 결과가 남습니다.
print(f"양자화된 모델을 '{quantized_model_dir}' 경로에 저장합니다...")
model.save_pretrained(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir) # 토크나이저도 함께 저장합니다.
print("모델 저장이 성공적으로 완료되었습니다.")