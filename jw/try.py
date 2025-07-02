import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from scipy.special import softmax

train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AI글판별/train.csv', encoding='utf-8-sig')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AI글판별/test.csv', encoding='utf-8-sig')
sample_submission = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AI글판별/sample_submission.csv', encoding='utf-8-sig')

def split_paragraphs(full_text: str):
    paras = re.split(r'\n{2,}', str(full_text))
    return [p.strip() for p in paras if p.strip()]

rows = []
for _, row in train_df.iterrows():
    for para in split_paragraphs(row['full_text']):
        rows.append({'paragraph_text': para, 'label': int(row['generated'])})
train_para_df = pd.DataFrame(rows)

train_data, val_data = train_test_split(
    train_para_df, test_size=0.1, stratify=train_para_df['label'], random_state=42
)

datasets = DatasetDict({
    'train': Dataset.from_pandas(train_data.reset_index(drop=True)),
    'validation': Dataset.from_pandas(val_data.reset_index(drop=True)),
    'test': Dataset.from_pandas(test_df.reset_index(drop=True)),
})

model_name = 'monologg/kobert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

def tokenize_function(examples):
    return tokenizer(
        examples['paragraph_text'], truncation=True, padding='max_length', max_length=512
    )

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=['paragraph_text'],
)

if 'ID' in tokenized_datasets['test'].column_names:
    tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['ID', 'title', 'paragraph_index'])

tokenized_datasets['train'].set_format(type='torch')
tokenized_datasets['validation'].set_format(type='torch')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir='output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='output/logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=None,
    compute_metrics=compute_metrics,
)

trainer.train()

test_preds = trainer.predict(tokenized_datasets['test'])
logits = test_preds.predictions
probs = softmax(logits, axis=1)
generated_probs = probs[:, 1]  # 1번 클래스가 'generated'일 경우

sample_submission['generated'] = generated_probs
sample_submission.to_csv('baseline_submission.csv', index=False)
print('baseline_submission.csv 가 생성되었습니다.')
