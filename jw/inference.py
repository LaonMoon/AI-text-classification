from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from scipy.special import softmax
import pandas as pd
import torch

# 1. 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 체크포인트 경로 지정 (원하는 checkpoint 폴더로)
checkpoint_path = "output/checkpoint-5000"

# 3. 모델 & 토크나이저 불러오기
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(device)
model_name = 'monologg/kobert'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 4. 테스트 데이터 불러오기 및 전처리
test_df = pd.read_csv('./test.csv', encoding='utf-8-sig')
sample_submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')

# 5. Hugging Face Dataset 변환
test_dataset = Dataset.from_pandas(test_df)
test_dataset = test_dataset.remove_columns(['ID', 'title', 'paragraph_index'])

def tokenize_fn(example):
    return tokenizer(example['paragraph_text'], truncation=True, padding='max_length', max_length=512)

tokenized_test = test_dataset.map(tokenize_fn, batched=True)
tokenized_test.set_format(type='torch')

# 6. Trainer 생성 (학습은 안 하고 predict만 함)
training_args = TrainingArguments(
    output_dir='output',
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

# 7. 예측 및 저장
test_preds = trainer.predict(tokenized_test)
logits = test_preds.predictions
probs = softmax(logits, axis=1)
generated_probs = probs[:, 1]  # 1번 클래스 확률

sample_submission['generated'] = generated_probs
sample_submission.to_csv(f'baseline_submission.csv', index=False)
print("✅ baseline_submission.csv 파일이 생성되었습니다.")
