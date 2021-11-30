import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

question_generator = "/home/solidsnake/ai/Golden_Group/ai-models/development/question-generation/BeIR-query-gen-msmarco-t5-large-v1"


tokenizer = T5Tokenizer.from_pretrained(question_generator)
model = T5ForConditionalGeneration.from_pretrained(question_generator)
model.eval()

para = "I would be interested in knowing what do you think of donald trump. He is a great guy. Do you still support him after what he have done? "

para = """Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum
and first released in 1991, Python's design philosophy emphasizes code
readability with its notable use of significant whitespace."""

input_ids = tokenizer.encode(para, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=3,
    )

print("Paragraph:")
print(para)

print("\nGenerated Queries:")
for i in range(len(outputs)):
    query = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print(f"{i + 1}: {query}")
