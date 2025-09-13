import argparse
from transformers import AutoModel, AutoTokenizer
import os

def convert_to_safetensors(model_dir):
    # safetensors로 저장할 새 디렉토리 경로
    save_dir = model_dir + "_safetensors"
    
    print(f"load {model_dir}")
    try:
        # 모델 로드 (모델 종류에 따라 AutoModelForCausalLM 등으로 변경 가능)
        model = AutoModel.from_pretrained(model_dir)

        # 토크나이저 로드 (선택 사항)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except OSError:
            tokenizer = None
        
        print(f"save {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        
        # safetensors 형식으로 저장 (핵심)
        model.save_pretrained(save_dir, safe_serialization=True)
        
        if tokenizer:
            tokenizer.save_pretrained(save_dir)
            
        print("-" * 30)
        print("✅ 변환 완료!")
        print(f"⭐ safetensors 모델이 '{save_dir}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"변환 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face 모델을 safetensors 형식으로 변환합니다.")
    
    parser.add_argument(
        "model_dir", 
        type=str, 
        help="변환할 모델이 있는 디렉토리 경로."
    )
    
    args = parser.parse_args()
    
    convert_to_safetensors(args.model_dir)