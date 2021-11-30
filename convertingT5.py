import os
import time
from statistics import stdev
from pathlib import Path
import torch
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from fastT5 import (
    export_and_get_onnx_model,
    generate_onnx_representation,
    quantize,
    get_onnx_model,
    get_onnx_runtime_sessions,
    OnnxT5,
)

torch.cuda.empty_cache()


#######################  Converting T5 to ONNX format  ########################

def from_pytorch_to_onnx(qg_model):

    # step 1: convert huggingface T5 model to ONNX
    onnx_model_paths = generate_onnx_representation(qg_model)

    # step 2: quantize the ONNX model for fast inference and to reduce model size
    quant_model_paths = quantize(onnx_model_paths)

    tokenizer_onnx = AutoTokenizer.from_pretrained(qg_model)
    config = T5Config.from_pretrained(qg_model)

    tokenizer_onnx.save_pretrained("models/")
    config.save_pretrained("models/")


#######################  ONNX T5 model inference  ########################

def onnx_model_init(qg_model):
    
    # print(Path(qg_model).stem)
    encoder_path = os.path.join(qg_model, f"onnx/{Path(qg_model).stem}-encoder-quantized.onnx")
    decoder_path = os.path.join(qg_model, f"onnx/{Path(qg_model).stem}-decoder-quantized.onnx")
    init_decoder_path = os.path.join(qg_model, f"onnx/{Path(qg_model).stem}-init-decoder-quantized.onnx")
    tokenizer_path = os.path.join(qg_model,"onnx")
    # print(encoder_path)
    
    model_paths = encoder_path, decoder_path, init_decoder_path

    model_sessions = get_onnx_runtime_sessions(model_paths)
    model = OnnxT5(qg_model, model_sessions)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer
    
    


##########################################################################

# print()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"AI model running on: {device}")
# print()


# # qg_model = "BeIR/query-gen-msmarco-t5-large-v1"

# tokenizer = T5Tokenizer.from_pretrained(qg_model)
# model = T5ForConditionalGeneration.from_pretrained(
#     qg_model,
# )
# model.eval().to(device)

# para = "Python is an interpreted, high-level and general-purpose programming language. Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects."

# start = time.time()
# input_ids = tokenizer.encode(para, return_tensors="pt").to(device)
# # print()
# # print(input_ids.shape)
# # print(input_ids)

# average_inf = []
# for _ in range(30):
#     start = time.time()
#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=input_ids,
#             max_length=64,
#             do_sample=True,
#             top_p=0.95,
#             num_return_sequences=3,
#         )
#     end = time.time()
#     average_inf.append(end - start)


# print("Paragraph:")
# print(para)

# print("\nGenerated Queries:")
# for i in range(len(outputs)):
#     query = tokenizer.decode(outputs[i], skip_special_tokens=True)
#     print(f"{i + 1}: {query}")

# print()
# print("Plain GPU Inference:")
# print(f"Mean inference on GPU: {sum(average_inf)/len(average_inf):.3f} sec.")
# print(f"Standard Deviation inference on GPU: {stdev(average_inf):.3f} sec.")


if __name__ == "__main__":
    
    convert_to_onnx = False
    onnx_inference = True
    
    qg_model = "/home/solidsnake/ai/Golden_Group/ai-models/development/question-generation/BeIR-query-gen-msmarco-t5-large-v1/"
    
    if convert_to_onnx:
        from_pytorch_to_onnx(qg_model)
    
    if onnx_inference:
        qg_model, qg_tokenizer = onnx_model_init(qg_model)
        


