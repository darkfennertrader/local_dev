import os
import subprocess
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    RobertaConfig,
    RobertaForSequenceClassification,
)

# device to trace the model
device = "cuda"  # can be "cuda" or "cpu"

# dir where model is saved
dir = "~/ai/Golden_Group/ai-models/torchscript/"
model_dir = os.path.expanduser(dir)

# model to trace
model_name = "cross-encoder/stsb-distilroberta-base"

# tarball model name
tar_model_name = "traced_stsb_distilroberta_base_" + device + torch.__version__


# instantiate model using torchscript
config = RobertaConfig.from_pretrained(model_name, torchscript=True)
tokenizer = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
model = RobertaForSequenceClassification(config)

model = RobertaForSequenceClassification.from_pretrained(model_name, torchscript=True)

# set model to inference mode
model.eval().to(device)
# print(model)

with torch.no_grad():
    input_ids = torch.randint(50265, (13, 10)).to(device)
    print(input_ids)
    attention_mask = torch.randint(2, (13, 10)).to(device)
    print(attention_mask)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

# print(inputs)

traced_model = torch.jit.trace(model, [input_ids, attention_mask])
torch.jit.save(traced_model, "model.pth")

# tarball the traced model
subprocess.call(["tar", "-czvf", tar_model_name + ".tar.gz", "model.pth"])
# # move tarball to the specified directory
# subprocess.call(["mv", tar_model_name + ".tar.gz", model_dir])

print("\nTraced Graph Model")
print("-" * 50)
print(traced_model.graph)
