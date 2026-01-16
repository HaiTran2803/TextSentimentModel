import torch
import numpy as np
import pandas as pd
# from underthesea import word_tokenize
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score,mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
####### get data
dataframe = pd.read_csv("vietnamese_emotion_dataset.csv",nrows=45)
subset = dataframe.iloc[:45]
subset_selected_filtered = subset[["text","labels"]].copy()
# print(subset_selected_filtered)

####### take the unique labels
type = sorted(np.unique(subset_selected_filtered["labels"]))

id2label = {
    1: "anger",
    2: "anxiety",
    3: "fear",
    4: "joy",
    5: "neutral",
    6: "sadness"
}
####### encode data
le = LabelEncoder()
for col in ['labels']:
    subset_selected_filtered[col] = le.fit_transform(subset_selected_filtered[col])
# print(subset_selected_filtered)

###### preprocessing
def preprocessing(text):
    text = text.lower().strip()
    # text = word_tokenize(text, format="text")
    return text

subset_selected_filtered['text'] = subset_selected_filtered['text'].apply(preprocessing)
# print(subset_selected_filtered)

####### model
MODEL_NAME = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(type))

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
dataset = Dataset.from_pandas(subset_selected_filtered)
# dataset = dataset.rename_column("label", "labels")
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch",columns=["input_ids", "attention_mask", "labels"])

####### train model
training_args = TrainingArguments(
    output_dir="./phobert_sentiment", #output in directory
    eval_strategy="epoch", # Đánh giá theo: epoch
    save_strategy="epoch", # khi nào lưu checkpoint: epoch
    learning_rate= 3e-5, # between 1e-5 to 5e-5 -> ~ 2 is good
    per_device_train_batch_size= 4, # 4-16, 8gb VRam -> 4 is good
    per_device_eval_batch_size= 4, # should 4 = train_batch_size
    gradient_accumulation_steps= 4, # giả lập batch lớn hơn mà không tăng Vram
    num_train_epochs= 4, # số lần học toàn bộ dataset
    weight_decay=0.01, # regularization - chống overfitting -> 0.01 is good for transformers
    logging_dir="./logs", # lưu log training with(loss,learning rate, Mertric
    fp16=True, # train with 16 bit -> giảm 50% VRam, tăng tốc (chỉ dùng khi có GPU NVIDIA or support RTX series)
    load_best_model_at_end=True, # sau khi train xong thì tự load checkpoint tốt nhất (evaluation_strategy = save_strategy = "epoch")
    metric_for_best_model="eval_loss", # phần chạy ngầm nếu không chỉnh sửa
    greater_is_better= False, # phần chạy ngầm
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

#### training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

###### Inference
def predict_sentiment(text):
    text = preprocessing(text)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    label = torch.argmax(outputs.logits, dim=1).item()
    return (label,id2label[label])

print(predict_sentiment("Sản phẩm này rất tốt"))
print(type)

