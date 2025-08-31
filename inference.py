# 1. 필요한 라이브러리 임포트
from transformers import AutoTokenizer, AutoModelForCausalLM

# 2. 모델 및 토크나이저 이름 지정
#model_name = "qwen_sft_hf_4bit_gptq"  # 예시: "gpt2", "bert-base-uncased", "t5-small" 등
model_name = "qwen_sft_hf_3epoch_4bit_gptq"  # 예시: "gpt2", "bert-base-uncased", "t5-small" 등

# 3. 토크나이저(Tokenizer) 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# 4. 모델(Model) 로드
print('loading model...', model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.to('cuda')

# 5. 추론(Inference) 수행
print('tokensizing...')
# ko-en ko-ja, ko-zh (bidirectional)
fnames = ['en.text', 'ko.text', 'zh.text', 'ja.text']
def to_input(fname, text):
    # if 'en' in fname:
    #     return {'e2k':f'Translate the given English into Korean. [English]: {text} [Korean]:'} 
    # elif 'zh' in fname:
    #     return {'z2k':f'Translate the given Chinese into Korean. [Chinese]: {text} [Korean]:'} 
    # elif 'ja' in fname:
    #     return {'j2k':f'Translate the given Japanese into Korean. [Japanese]: {text} [Korean]:'} 
    # elif 'ko' in fname:
    #     return {'k2e':f'Translate the given Korean into English. [Korean]: {text} [English]:',
    #             'k2z':f'Translate the given Korean into Chinese. [Korean]: {text} [Chinese]:',
    #             'k2j':f'Translate the given Korean into Japanese. [Korean]: {text} [Japanese]:'}
    if 'en' in fname:
        return {'e2k':f'<|im_start|>Translate the given English into 한국어. <|English|>: {text} <|한국어|>:'} 
    elif 'zh' in fname:
        return {'z2k':f'<|im_start|>Translate the given 中文 into 한국어. <|中文|>: {text} <|한국어|>:'} 
    elif 'ja' in fname:
        return {'j2k':f'<|im_start|>Translate the given 日本語 into 한국어. <|日本語|>: {text} <|한국어|>:'} 
    elif 'ko' in fname:
        return {'k2e':f'<|im_start|>Translate the given 한국어 into English. <|한국어|>: {text} <|English|>:',
                'k2z':f'<|im_start|>Translate the given 한국어 into 中文. <|한국어|>: {text} <|中文|>:',
                'k2j':f'<|im_start|>Translate the given 한국어 into 日本語. <|한국어|>: {text} <|日本語|>:'}

test_sets = {}
for fname in fnames:
    with open(f'test_set/{fname}', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            inputs = to_input(fname, line)
            for k, v in inputs.items():
                if k not in test_sets:
                    test_sets[k] = [v]
                else:
                    test_sets[k].append(v)


for k, datas in test_sets.items():
    print(f'---{k}---')
    # 프롬프트를 토크나이저로 인코딩(토큰화)합니다.
    inputs = []
    batch_start, batch_size = 0, 64
    print('split batching...', len(datas))
    while True:
        if len(datas) <= batch_start:
            break
        if len(datas) < batch_start + batch_size:
            batch = datas[batch_start:]
        else:
            batch = datas[batch_start:batch_start+batch_size]
        tokenized_inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs.append((tokenized_inp, batch))
        batch_start += batch_size

    fw = open(f'outputs/{model_name}_{k}.txt', 'w', encoding='utf-8')
    print('generate...', len(inputs))
    # 모델을 사용하여 텍스트 생성
    for idx, (tokenized, raw) in enumerate(inputs):
        #print(idx, tokenized[0].tokens, tokenized[0].ids)
        tokenized.to('cuda')
        outputs = model.generate(**tokenized, max_new_tokens=128, num_return_sequences=1, repetition_penalty=1.1)
        for i, o in zip(raw, outputs):
            generated_text = tokenizer.decode(o, skip_special_tokens=True)
            generated_text = generated_text.split('|>:')[-1].strip().replace('\n',' ')
            # generated_text = generated_text.split(']:')[-1].strip().replace('\n',' ')
            fw.write(generated_text + '\n')
 
    fw.close()
    print(f'outputs/{model_name}_{k}.txt saved.')