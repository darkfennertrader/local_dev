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


dialoRPT = ["human-vs-machine", "human-vs-rand"]

model = dialoRPT[0]

config_name = f"microsoft/DialogRPT-{model}"
cache_dir = f"../ai-models/pre-trained/dialorpt/{model}"
save_dir = f"../ai-models/pre-trained/dialorpt/{model}"
tokenizer_name = f"microsoft/DialogRPT-{model}"
model_name_or_path = f"microsoft/DialogRPT-{model}"

# given a context and one human response, distinguish it with a random human response
config = AutoModelForSequenceClassification.from_pretrained(
    config_name, cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    from_tf=False,
    config=config,
    cache_dir=cache_dir,
)

config.save_pretrained(save_dir)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# mod_hrand = AutoModelForSequenceClassification.from_pretrained(
#     "microsoft/DialogRPT-human-vs-rand",
#     cache_dir="./models/pre-trained/dialorpt/model-cards/human-vs-rand",
# ).to(device)
# tok_hrand = AutoTokenizer.from_pretrained(
#     "microsoft/DialogRPT-human-vs-rand",
#     cache_dir="./models/pre-trained/dialorpt/tokenizers/human-vs-rand",


# # given a context and one human response, distinguish it with a machine generated response
# mod_hmach = AutoModelForSequenceClassification.from_pretrained(
#     "microsoft/DialogRPT-human-vs-machine",
#     cache_dir="./models/pre-trained/dialorpt/model-cards/human-vs-machine",
# ).to(device)
# tok_hmach = AutoTokenizer.from_pretrained(
#     "microsoft/DialogRPT-human-vs-machine",
#     cache_dir="./models/pre-trained/dialorpt/tokenizers/human-vs-machine",
# )
