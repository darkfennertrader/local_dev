import os
import timeit
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
)
import onnxruntime

print()
print("-" * 50)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"model loaded to {device}")
print(f"torch version: {torch.__version__}")
print("-" * 50)
print()

model_location = [
    # ("cross-encoder/", "stsb-roberta-base"),
    # ("cross-encoder/", "stsb-roberta-large"),
    ("cross-encoder/", "stsb-distilroberta-base"),
    # ("cross-encoder/", "stsb-TinyBERT-L-4"),
    # ("sentence-transformers/", "paraphrase-distilroberta-base-v1"),
]

dir = "../ai-models/pre-trained/"

# check if model exists otherwise download it
for mod in model_location:
    if not os.listdir(dir + mod[1]):
        model_name = mod[0] + mod[1]
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )
        tokenizer.save_pretrained(dir + mod[1])
        model.save_pretrained(dir + mod[1])
    else:
        print(f"Directory {dir + mod[1]} is not empty")

print()
# model = AutoModelForSequenceClassification.from_pretrained(
#     "cross-encoder/stsb-distilroberta-base"
# )
# print("Model's state dict")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# torch.save(model.state_dict(), "stsb-distilroberta-base.pt")


query = "living in L.A."
# With all sentences in the corpus
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
    "Do you believe in God?",
    "What's your name?",
    "How did you die?",
    "I live in Los Angeles",
]


def smart_batching_collate_text_only(batch):
    texts = [[] for _ in range(len(batch[0]))]

    for example in batch:
        for idx, text in enumerate(example):
            texts[idx].append(text.strip())

    tokenized = tokenizer(
        *texts,
        padding=True,
        truncation="longest_first",
        return_tensors="pt",
        max_length=None,
    )

    for name in tokenized:
        tokenized[name] = tokenized[name].to(device)

    # print(tokenized)

    return tokenized


print()
print("Query: ", query)
print()

# SEMANTIC SEARCH with Sentence-Transformers
print("\nSEMANTIC SEARCH with Sentence-Transformers")
# So we create the respective sentence combinations
sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

for mod in model_location:
    model_name = mod[0] + mod[1]
    model = CrossEncoder(model_name)
    start = timeit.timeit()
    # Compute the similarity scores for these combinations
    similarity_scores = model.predict(sentence_combinations)
    sim_score_argmax = np.argmax(similarity_scores)
    print("Most similar question: ", corpus[sim_score_argmax])
    print("Similarity: ", max(similarity_scores))
    print()
    print(timeit.timeit() - start)

# SEMANTIC SEARCH with Transformers
print("\nSEMANTIC SEARCH with Transformers")
model_name = "../ai-models/pre-trained/" + "stsb-distilroberta-base"
config = AutoConfig.from_pretrained(model_name)
# print(config)
tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)


sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

start = timeit.timeit()
inp_dataloader = DataLoader(
    sentence_combinations,
    batch_size=32,
    collate_fn=smart_batching_collate_text_only,
    num_workers=0,
    shuffle=False,
)

pred_scores = []
model.eval()
model.to(device)
with torch.no_grad():
    for features in inp_dataloader:
        # print("\nfeatures")
        # print(features)
        # print(features["input_ids"].shape)
        # print(features["attention_mask"].shape)
        # print("-" * 50)
        model_predictions = model(**features, return_dict=True)
        # print(model_predictions)
        # print("\nmodel_predictions_logits")
        # print(model_predictions.logits)
        logits = nn.Sigmoid()(model_predictions.logits)
        # print(logits)
        pred_scores.extend(logits)

pred_scores = [score[0] for score in pred_scores]
pred_scores = torch.stack(pred_scores)
print(f"Most similar question: {corpus[torch.argmax(pred_scores)]}")
print(f"Similarity: {torch.max(pred_scores):2.6f}")
print()
print(timeit.timeit() - start)


# SEMANTIC SEARCH with Torchscript
print("\nSEMANTIC SEARCH with TorchScript")
sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]
print(sentence_combinations)
model_name = "../ai-models/fine-tuned/semantic-search/model.pth"
loaded_model = torch.jit.load(model_name, map_location=torch.device("cuda"))
loaded_model.eval()

start = timeit.timeit()
pred_scores = []
with torch.no_grad():
    for features in inp_dataloader:
        # print("\nfeatures")
        # print(features)
        # print(features["input_ids"].shape)
        # print(features["attention_mask"].shape)

        model_predictions = loaded_model(
            features["input_ids"], features["attention_mask"]
        )
        # or in compact form
        # model_predictions = loaded_model(**features)
        logits_tensor, *_ = model_predictions
        logits = nn.Sigmoid()(logits_tensor)
        pred_scores.extend(logits)

pred_scores = [score[0] for score in pred_scores]
pred_scores = torch.stack(pred_scores)
print(f"Most similar question: {corpus[torch.argmax(pred_scores)]}")
print(f"Similarity: {torch.max(pred_scores):2.6f}")
print()
print(timeit.timeit() - start)


# # SEMANTIC SEARCH with pytorch model
# embedder = SentenceTransformer("paraphrase-distilroberta-base-v1")
# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True).to("cuda")
# print(f"corpus embedding shape: {corpus_embeddings.shape}")

# # query sentence
# queries = [query]
# # Find the closest sentence of the corpus for each query sentence based on cosine similarity
# top_k = min(1, len(corpus))
# for query in queries:
#     query_embedding = embedder.encode(query, convert_to_tensor=True).to("cuda")
#     cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
#     top_results = torch.topk(cos_scores, k=top_k)
#     for score, idx in zip(top_results[0], top_results[1]):
#         print(corpus[idx], "(Score: {:.4f})".format(score))


# SEMANTIC SEARCH with onnx model

if __name__ == "__main__":
    pass

    # class MyModule(torch.nn.Module):
    #     def forward(self, x):
    #         return x + 10

    # m = torch.jit.script(MyModule())
    # print(m)
