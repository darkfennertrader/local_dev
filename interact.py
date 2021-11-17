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
)
import onnxruntime

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import json

model_to_load = "../ai-models/fine-tuned/dialogpt/small"


tokenizer = AutoTokenizer.from_pretrained(model_to_load)
model = AutoModelForCausalLM.from_pretrained(model_to_load)
device = torch.device("cuda")
model = model.to(device)

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        input(">> User:") + tokenizer.eos_token, return_tensors="pt"
    ).to(device)

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1).to(device)
        if step > 0
        else new_user_input_ids
    )

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8,
    )

    # pretty print last ouput tokens from bot
    print(
        "DialoGPT: {}".format(
            tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
        )
    )
