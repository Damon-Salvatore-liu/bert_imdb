# Author : liupengli
# Contact : a18810865023@163.com
# Time : 2022/05/06 9:42 AM

from datasets import load_dataset
from datasets import load_metric
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer


def main():
    #导入数据,该数据集是一个具有三个键的字典："train","test"和"unsupervised" 。我们使用"train"进行训练，使用 "test"进行验证。
    raw_datasets = load_dataset("imdb")

    # 导入分词器以及模型
    check_point = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(check_point, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(check_point)

    #文本截断：批量处理
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    #计算过程中的指标
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    #获取数据集中的一部分，进行训练（非必须，主要是快，可以用来先进行调试）
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000)) 
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) 
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    #实例化一个 TrainingArguments。这个类包含我们可以为Trainer或标志调整的所有超参数 ，以激活它支持的不同训练选项。
    training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

    #实例化一个Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    #微调
    trainer.train()

    #验证
    trainer.evaluate()



if __name__ == '__main__':
    main()