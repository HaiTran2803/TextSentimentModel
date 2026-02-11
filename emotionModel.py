import torch
import numpy as np
import pandas as pd
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

###### preprocessing
def preprocessing(text):
    text = text.lower().strip()
    return text

subset_selected_filtered['text'] = subset_selected_filtered['text'].apply(preprocessing)

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
dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch",columns=["input_ids","attention_mask", "labels"])

####### train model
training_args = TrainingArguments(
    output_dir="./phobert_sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate= 2e-5,
    per_device_train_batch_size= 4,
    per_device_eval_batch_size= 4,
    gradient_accumulation_steps= 4,
    num_train_epochs= 4, 
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better= False,
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

