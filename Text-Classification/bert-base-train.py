from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import optuna
from transformers import get_scheduler
import torch
# from torch.optim import AdamW
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        torch.manual_seed(42)
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Define a function to compute accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

train_dataset = dataset["train"].map(lambda x: tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True), batched=True)
eval_dataset = dataset["validation"].map(lambda x: tokenizer(x["sentence1"], x["sentence2"], padding="max_length", truncation=True), batched=True)
learning_rate = 1.3575952445586316e-05
# Define training arguments
training_args = TrainingArguments(
    output_dir="/scratch/gilbreth/amohanpa/hyperparam/bert-base/mrpc/results/from_finetuned_fixing_weights",
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=21,
    gradient_accumulation_steps = 3,
    logging_dir="/scratch/gilbreth/amohanpa/hyperparam/bert-base/mrpc/logs/from_finetuned_fixing_weights",
    logging_steps=100,
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=1.3575952445586316e-05,
)
model = BertForSequenceClassification.from_pretrained("/scratch/gilbreth/amohanpa/bert-base/mrpc/", from_tf=bool(".ckpt" in "/scratch/gilbreth/amohanpa/bert-base/mrpc/"), num_labels=2)
# model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
# initial_model = torch.load("/home/consus/a/amohanpa/Whisper-tiny-softmax/Hardsigmoid_Conv_Square_Spearmann_WithMSELoss-BertBase.pth",map_location='cpu')#['state_dict']
# # for layer in initial_model.keys():
# #     print(layer)
# for name, param in model.named_parameters():
#     # param.requires_grad = True
#     # print(type(name))
    
#     # print(param.requires_grad)
#     if 'softmax' in name:
#         param.requires_grad = True
#         # if '0.weight' in name:
#         #     print("in 3.weight")
#         #     # param.data = initial_model['conv1.weight']
#         # if '0.bias' in name:
#         #     print("in 3.bias")
#         #     # param.data = initial_model['conv1.bias']
#         # if '2.weight' in name:
#         #     print("in 2.weight")
#         #     # param.data = initial_model['conv2.weight']
#         # if '2.bias' in name:
#         #     print("in 2.bias")
#             # param.data = initial_model['conv2.bias']
#     #     # print(name)
#     #     # print(param.requires_grad)
#     else:
#         # print(name)
#         param.requires_grad = False
# Initialize Trainer
model.apply(init_weights)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_coefficient = 0.0,
    compute_metrics=compute_metrics,
)
# eval_result = trainer.evaluate()
# return eval_result
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Define learning rate scheduler
lr_scheduler = None
if lr_scheduler_type == "linear":
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
elif lr_scheduler_type == "cosine":
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)
elif lr_scheduler_type == "polynomial":
    lr_scheduler = get_scheduler("polynomial",optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)
# Set learning rate scheduler
if lr_scheduler:
    trainer.lr_scheduler = lr_scheduler

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(eval_result["eval_accuracy"])
