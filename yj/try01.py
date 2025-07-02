!pip install -U transformers

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 코랩에서 처음 unzip 할 때만 사용
# !unzip /content/open.zip -d /content/


##### 데이터 확인 #####
# 데이터 로딩
train = pd.read_csv('./train.csv', encoding='utf-8-sig')
test = pd.read_csv('./test.csv', encoding='utf-8-sig')

# 데이터 크기 확인
print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")

# 평균 문단 길이 확인
train['full_text_len'] = train['full_text'].apply(lambda x: len(str(x)))
print("Train 문단 평균 길이:", train['full_text_len'].mean())
print("Train 문단 최대 길이:", train['full_text_len'].max())

# 문장 수 기준도 보고 싶다면:
train['sentence_count'] = train['full_text'].apply(lambda x: str(x).count('.') + str(x).count('!') + str(x).count('?'))
print("평균 문장 수:", train['sentence_count'].mean())

# 메모리 사용량
print("Memory usage:")
print(train.memory_usage(deep=True))
###########################


train['text'] = train['title'] + ' ' + train['full_text']
test['text'] = test['title'] + ' ' + test['paragraph_text']

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(train['text'], train['generated'], stratify=train['generated'], test_size=0.2, random_state=42)

# 토크나이저 및 모델 로드
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset 정의
class ParagraphDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512, batch_size=10000):
        self.input_ids = []
        self.attention_mask = []
        self.labels = labels
        self.tokenizer = tokenizer

        print("Tokenizing in batches...")

        # 배치 토크나이즈로 메모리 절약
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = list(texts[i:i+batch_size])
            batch_encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors='pt'
            )
            self.input_ids.extend(batch_encodings['input_ids'])
            self.attention_mask.extend(batch_encodings['attention_mask'])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.input_ids)

train_dataset = ParagraphDataset(X_train, y_train, tokenizer=tokenizer)
val_dataset = ParagraphDataset(X_val, y_val, tokenizer=tokenizer)
test_dataset = ParagraphDataset(test['text'], tokenizer=tokenizer)


# 모델 초기화
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    fp16=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=100,
)

# 평가 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {"roc_auc": roc_auc_score(labels, probs)}

# Trainer 설정 및 학습
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

#### 여기 wandb api 입력해야 함 ####
trainer.train()


# 테스트 데이터 예측
preds = trainer.predict(test_dataset)
probs = torch.softmax(torch.tensor(preds.predictions), dim=-1)[:, 1].numpy()


#  제출 파일 저장
submission = pd.read_csv('./sample_submission.csv')
submission['generated'] = probs
submission.to_csv('./submission_kcelectra.csv', index=False)
