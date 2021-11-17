import os
import timeit
from functools import partial
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
import torch.onnx
import torchvision
from transformers.convert_graph_to_onnx import convert
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BartForConditionalGeneration,
    RobertaForSequenceClassification,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2DoubleHeadsModel,
    GPT2Tokenizer,
)
import onnxruntime

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import json


# config_name = "microsoft/DialoGPT-medium"
# cache_dir = "../ai-models/pre-trained/dialogpt/medium"
# save_dir = "../ai-models/pre-trained/dialogpt/medium"
# tokenizer_name = "microsoft/DialoGPT-medium"
# model_name_or_path = "microsoft/DialoGPT-medium"


save_dir = "/home/solidsnake/ai/Golden_Group/ai-models/development/cross-encoders/ms-marco-TinyBERT-L-2-v2/"
model_name_or_path = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


# Natural Language Intent (Zero-Shot Classification)
config = AutoConfig.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    from_tf=False,
    config=config,
)

# print(config)

# # convAI model
# config = GPT2Config.from_pretrained(config_name, cache_dir=cache_dir)
# tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
# model = GPT2DoubleHeadsModel.from_pretrained(
#     model_name_or_path,
#     from_tf=False,
#     config=config,
#     cache_dir=cache_dir,
# )


# config = AutoConfig.from_pretrained(config_name, cache_dir=cache_dir)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     from_tf=False,
#     config=config,
#     cache_dir=cache_dir,
# )

config.save_pretrained(save_dir)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


# load model
# config = AutoConfig.from_pretrained(save_dir)
# tokenizer = AutoTokenizer.from_pretrained(save_dir)
# model = AutoModelForSequenceClassification.from_pretrained(save_dir)

# tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
# model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
