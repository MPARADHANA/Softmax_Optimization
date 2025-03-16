#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
import json

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AdamW,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import torch
import torch.nn.init as init

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    layer_num_approx: Optional[int] = field(
        default=None, metadata={"help": "layer where heads are going to be approximated"}
    )
    head_num_approx: Optional[int] = field(
        default=None, metadata={"help": "head which is going to be approximated"}
    )
    layer0_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer1_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer2_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer3_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer4_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer5_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer6_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer7_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer8_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer9_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer10_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer11_list: Optional[str] = field(
        default=None, metadata={"help": "head which is going to be approximated as dict"}
    )
    layer_num_new: Optional[int] = field(
        default=None, metadata={"help": "new layer where heads are going to be approximated"}
    )
    head_num_new: Optional[int] = field(
        default=None, metadata={"help": "new head which is going to be approximated"}
    )
    





parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

if model_args.use_auth_token is not None:
    warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
    if model_args.token is not None:
        raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
    model_args.token = model_args.use_auth_token

# Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
# information sent is the one passed as arguments along with your Python/PyTorch versions.
send_example_telemetry("run_glue", model_args, data_args)
if model_args.layer0_list is not None:
    layer0_list_converted = list()
    a = list(model_args.layer0_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer0_list_converted.append(int(i))

    layer1_list_converted = list()
    a = list(model_args.layer1_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer1_list_converted.append(int(i))

    layer2_list_converted = list()
    a = list(model_args.layer2_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer2_list_converted.append(int(i))

    layer3_list_converted = list()
    a = list(model_args.layer3_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer3_list_converted.append(int(i))

    layer4_list_converted = list()
    a = list(model_args.layer4_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer4_list_converted.append(int(i))

    layer5_list_converted = list()
    a = list(model_args.layer5_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer5_list_converted.append(int(i))

    layer6_list_converted = list()
    a = list(model_args.layer6_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer6_list_converted.append(int(i))

    layer7_list_converted = list()
    a = list(model_args.layer7_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer7_list_converted.append(int(i))

    layer8_list_converted = list()
    a = list(model_args.layer8_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer8_list_converted.append(int(i))

    layer9_list_converted = list()
    a = list(model_args.layer9_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer9_list_converted.append(int(i))

    layer10_list_converted = list()
    a = list(model_args.layer10_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer10_list_converted.append(int(i))

    layer11_list_converted = list()
    a = list(model_args.layer11_list)
    for i in a:
        if (i!='[')and(i!=']')and(i!=',')and(i!=' '):
            layer11_list_converted.append(int(i))

# if len(layer1_list_converted)==0:
#     layer_num_new = 0
#     head_num_new = layer0_list_converted[len(layer0_list_converted)-1]
# if len(layer2_list_converted)==0 and len(layer1_list_converted)!=0:
#     layer_num_new = 1
#     head_num_new = layer1_list_converted[len(layer1_list_converted)-1]
# if len(layer3_list_converted)==0 and len(layer2_list_converted)!=0:
#     layer_num_new = 2
#     head_num_new = layer2_list_converted[len(layer2_list_converted)-1]
# if len(layer4_list_converted)==0 and len(layer3_list_converted)!=0:
#     layer_num_new = 3
#     head_num_new = layer3_list_converted[len(layer3_list_converted)-1]
# if len(layer5_list_converted)==0 and len(layer4_list_converted)!=0:
#     layer_num_new = 4
#     head_num_new = layer4_list_converted[len(layer4_list_converted)-1]
# if len(layer6_list_converted)==0 and len(layer5_list_converted)!=0:
#     layer_num_new = 5
#     head_num_new = layer5_list_converted[len(layer5_list_converted)-1]
# if len(layer7_list_converted)==0 and len(layer6_list_converted)!=0:
#     layer_num_new = 6
#     head_num_new = layer6_list_converted[len(layer6_list_converted)-1]
# if len(layer8_list_converted)==0 and len(layer7_list_converted)!=0:
#     layer_num_new = 7
#     head_num_new = layer7_list_converted[len(layer7_list_converted)-1]
# if len(layer9_list_converted)==0 and len(layer8_list_converted)!=0:
#     layer_num_new = 8
#     head_num_new = layer8_list_converted[len(layer8_list_converted)-1]
# if len(layer10_list_converted)==0 and len(layer9_list_converted)!=0:
#     layer_num_new = 9
#     head_num_new = layer9_list_converted[len(layer9_list_converted)-1]
# if len(layer11_list_converted)==0 and len(layer10_list_converted)!=0:
#     layer_num_new = 10
#     head_num_new = layer10_list_converted[len(layer10_list_converted)-1]
# if len(layer11_list_converted)!=0:
#     layer_num_new = 11
#     head_num_new = layer11_list_converted[len(layer11_list_converted)-1]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Set seed before initializing model.
set_seed(training_args.seed)

if data_args.task_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        "glue",
        data_args.task_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
elif data_args.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
else:
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
# See more about loading any type of standard or custom dataset at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Labels
if data_args.task_name is not None:
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
        print(num_labels)
    else:
        num_labels = 1
else:
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

# Load pretrained model and tokenizer
#
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
)
# print("New addition")
# print(layer_num_new)
# print(head_num_new)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        # torch.manual_seed(42)
        # init.xavier_uniform_(m.weight)
        # if m.bias is not None:
        #     init.constant_(m.bias, 0)
        # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # print("inside apply")
        for name, module in model.named_modules():
            # print(module)
            # print(m)
            if module is m:
                print(name)
                layer_name = name
                break
        if "layer.{}.attention.self.softmax.{}".format(model_args.layer_num_new,model_args.head_num_new) in layer_name:
            print("In")
            torch.manual_seed(42)
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.35.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
# }

# logger = logging.getLogger(__name__)
# torch.autograd.set_detect_anomaly(True)

# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.

#     Using `HfArgumentParser` we can turn this class
#     into argparse arguments to be able to specify them on
#     the command line.
#     """

#     task_name: Optional[str] = field(
#         default=None,
#         metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
#     )
#     dataset_name: Optional[str] = field(
#         default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
#     )
#     dataset_config_name: Optional[str] = field(
#         default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
#     )
#     max_seq_length: int = field(
#         default=128,
#         metadata={
#             "help": (
#                 "The maximum total input sequence length after tokenization. Sequences longer "
#                 "than this will be truncated, sequences shorter will be padded."
#             )
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
#     )
#     pad_to_max_length: bool = field(
#         default=True,
#         metadata={
#             "help": (
#                 "Whether to pad all samples to `max_seq_length`. "
#                 "If False, will pad the samples dynamically when batching to the maximum length in the batch."
#             )
#         },
#     )
#     max_train_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of training examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_eval_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
#                 "value if set."
#             )
#         },
#     )
#     max_predict_samples: Optional[int] = field(
#         default=None,
#         metadata={
#             "help": (
#                 "For debugging purposes or quicker training, truncate the number of prediction examples to this "
#                 "value if set."
#             )
#         },
#     )
#     train_file: Optional[str] = field(
#         default=None, metadata={"help": "A csv or a json file containing the training data."}
#     )
#     validation_file: Optional[str] = field(
#         default=None, metadata={"help": "A csv or a json file containing the validation data."}
#     )
#     test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

#     def __post_init__(self):
#         if self.task_name is not None:
#             self.task_name = self.task_name.lower()
#             if self.task_name not in task_to_keys.keys():
#                 raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
#         elif self.dataset_name is not None:
#             pass
#         elif self.train_file is None or self.validation_file is None:
#             raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
#         else:
#             train_extension = self.train_file.split(".")[-1]
#             assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
#             validation_extension = self.validation_file.split(".")[-1]
#             assert (
#                 validation_extension == train_extension
#             ), "`validation_file` should have the same extension (csv or json) as `train_file`."


# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
#     """

#     model_name_or_path: str = field(
#         metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None,
#         metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
#     )
#     use_fast_tokenizer: bool = field(
#         default=True,
#         metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
#     )
#     model_revision: str = field(
#         default="main",
#         metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
#     )
#     token: str = field(
#         default=None,
#         metadata={
#             "help": (
#                 "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
#                 "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
#             )
#         },
#     )
#     use_auth_token: bool = field(
#         default=None,
#         metadata={
#             "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
#         },
#     )
#     trust_remote_code: bool = field(
#         default=False,
#         metadata={
#             "help": (
#                 "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
#                 "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
#                 "execute code present on the Hub on your local machine."
#             )
#         },
#     )
#     ignore_mismatched_sizes: bool = field(
#         default=False,
#         metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
#     )
#     layer_num_approx: Optional[int] = field(
#         default=None, metadata={"help": "layer where heads are going to be approximated"}
#     )
#     head_num_approx: Optional[int] = field(
#         default=None, metadata={"help": "head which is going to be approximated"}
#     )
#     layer0_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer1_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer2_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer3_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer4_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer5_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer6_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer7_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer8_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer9_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer10_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )
#     layer11_list: Optional[str] = field(
#         default=None, metadata={"help": "head which is going to be approximated as dict"}
#     )












def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # if model_args.use_auth_token is not None:
    #     warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
    #     if model_args.token is not None:
    #         raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
    #     model_args.token = model_args.use_auth_token

    # # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # if training_args.should_log:
    #     # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    #     transformers.utils.logging.set_verbosity_info()

    # log_level = training_args.get_process_log_level()
    # logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.enable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

    # # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # # Set seed before initializing model.
    # set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # # Load pretrained model and tokenizer
    # #
    # # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # # download model & vocab.
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=data_args.task_name,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     token=model_args.token,
    #     trust_remote_code=model_args.trust_remote_code,
    # )
    # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # # tokenizer = AutoTokenizer.from_pretrained(
    # #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    # #     cache_dir=model_args.cache_dir,
    # #     use_fast=model_args.use_fast_tokenizer,
    # #     revision=model_args.model_revision,
    # #     token=model_args.token,
    # #     trust_remote_code=model_args.trust_remote_code,
    # # )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     token=model_args.token,
    #     trust_remote_code=model_args.trust_remote_code,
    #     ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    # )
    model_pseudo = AutoModelForSequenceClassification.from_pretrained(
        "/scratch/gilbreth/amohanpa/bert-base/mrpc/head-by-head-seq_0_5_initfixed/",
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    # print(dir(model))
    for name, param in model.named_parameters():
        for name2, param2 in model_pseudo.named_parameters():
            if (name2 == name) and ('softmax' in name):
                param.data = param2.data
                print('modified')

    # for layer_index in range(12):
    #     model_attention_param = model.encoder.layer[layer_index].attention.state_dict()
    #     model_pseudo_attention_param = model_pseudo.encoder.layer[layer_index].attention.state_dict()
    #     for key1 in model_attention_param.keys():
    #         if ('softmax' in key1) and (key1 in model_pseudo_attention_param):
    #             model_attention_param[key1] = model_pseudo_attention_param[key1] 
        
    #     model.encoder.layer[layer_index].attention.load_state_dict(model_attention_param)
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    # print(data_args.pad_to_max_length)
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
    padding = False
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    # model_fixed = torch.load_state_dict()    
    for name, param in model.named_parameters():
        print(name)
        # param.requires_grad = True
        # print(type(name))
        
        # print(param.requires_grad)
        if 'softmax' in name:
            param.requires_grad = True
            # if '3.weight' in name:
            #     print("in 3.weight")
            #     param.data = initial_model['3.weight']
            # if '0.bias' in name:
            #     print("in 0.bias")
            #     param.data = initial_model['3.bias']
            # if '2.weight' in name:
            #     print("in 2.weight")
            #     param.data = initial_model['conv.weight']
            # if '2.bias' in name:
            #     print("in 2.bias")
            #     param.data = initial_model['conv2.bias']
        else:
            param.requires_grad = False
    
    
    # learning_rate = 0.18436
    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=2785)
    # print("Head list {}".format((model_args.head_list)))
    # a = [*model_args.head_list]
    # for i in range(len(model_args.head_list)-2):
    #     if model_args.head_list[i+1]==":":
    #         a[i]=int(model_args.head_list[i])
    #     else:
    #         a[i]=model_args.head_list[i]
    # print(a)
    # a_dict = dict([model_args.head_list.strip('{}').split(":"),])
    # print(a_dict)
    # print(model_args.layer0_list)
    # print(type(model_args.layer0_list))
    
    
    head_list = None
    if model_args.layer0_list is not None:
        head_list = dict()
        model.apply(init_weights)
        head_list[0]=layer0_list_converted
        head_list[1]=layer1_list_converted
        head_list[2]=layer2_list_converted
        head_list[3]=layer3_list_converted
        head_list[4]=layer4_list_converted
        head_list[5]=layer5_list_converted
        head_list[6]=layer6_list_converted
        head_list[7]=layer7_list_converted
        head_list[8]=layer8_list_converted
        head_list[9]=layer9_list_converted
        head_list[10]=layer10_list_converted
        head_list[11]=layer11_list_converted
    
    head_list = {0: [8, 2, 1, 5], 1: [2, 11, 6], 2: [7, 9, 4, 1], 3: [], 4: [6, 3, 1, 11], 5: [0, 5, 10, 4, 7, 1], 6: [11, 8, 3, 2, 1], 7: [5, 7, 2, 4, 1, 8, 11, 9], 8: [5, 9, 11, 1, 3, 4], 9: [0, 5, 7, 11, 8, 6, 2, 1], 10: [1, 0, 11, 8, 9, 7, 4, 6, 5, 2, 3], 11: [8, 4, 9, 7, 1, 0, 10, 5, 2, 11, 3]}
    # for key in model_args.head_list.keys():
    #     print(key)
    # print(model.args.head_list[0])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_coefficient = 1.38e-08,
        layer_num_approx = model_args.layer_num_approx,
        head_num_approx = model_args.head_num_approx,
        head_list = head_list,
    )
    # trainer.lr_scheduler = lr_scheduler
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # model.apply(init_weights)
        
        # learning_rate = 1.3575952445586316e-05
        # optimizer = AdamW(model.parameters(), lr=learning_rate)
        # num_training_steps = 8343
        # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        # trainer.lr_scheduler = lr_scheduler

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # for name, param in model.named_parameters():
        #     # param.requires_grad = True
        #     # print(type(name))
            
        #     # print(param.requires_grad)
        #     if 'softmax' in name:
        #         param.requires_grad = True
        #         # if '3.weight' in name:
        #         #     print("in 3.weight")
        #         #     param.data = initial_model['3.weight']
        #         # if '0.bias' in name:
        #         #     print("in 0.bias")
        #         #     param.data = initial_model['3.bias']
        #         # if '2.weight' in name:
        #         #     print("in 2.weight")
        #         #     param.data = initial_model['conv.weight']
        #         # if '2.bias' in name:
        #         #     print("in 2.bias")
        #         #     param.data = initial_model['conv2.bias']
        #     #     # print(name)
        #     #     # print(param.requires_grad)
        #     else:
        #         # print(name)
        #         param.requires_grad = False
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        train_loss = metrics["train_loss"]
        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            accuracy_value = metrics["eval_accuracy"]
            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    print(accuracy_value)
    return train_loss

def _mp_fn(index):
    # For xla_spawn (TPUs)
    result = main()
    return result


if __name__ == "__main__":
    result = main()
    print(result)
