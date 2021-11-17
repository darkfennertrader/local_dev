from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name_or_dir = "/home/solidsnake/ai/Golden_Group/ai-models/development/cross-encoders/ms-marco-MiniLM-L-12-v2"

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_dir).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)

query = "How many people live in Berlin?"
sentences_list = [
    "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin is a beautiful city",
    "London has a population of around 8 million registered inhabitants",
    "people who live Berlin are very unfriendly",
]

query_list = [query] * len(sentences_list)

features = tokenizer(
    query_list,
    sentences_list,
    padding=True,
    truncation=True,
    return_tensors="pt",
).to("cuda")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)

best_sentence = sentences_list[torch.argmax(scores)]
print(best_sentence)
