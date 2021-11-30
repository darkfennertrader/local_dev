import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

model_name_or_path = "/home/solidsnake/ai/Golden_Group/ai-models/development/question-generation/valhalla-t5-base-e2e-qg"

tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
model.eval()


user_utterance = "Have you seen the TV show Devs? What was it about do you know?"


class E2EQGPipeline:
    def __init__(self, model, tokenizer, use_cuda):

        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        self.model_type = "t5"

        self.default_generate_kwargs = {
            "max_length": 50,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "num_return_sequences": 3,
        }

    def __call__(self, context, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overriding default_generate_kwargs all other arguments need to be passsed find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs

        input_length = inputs["input_ids"].shape[-1]

        outs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            **generate_kwargs,
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        # print(questions)
        questions = [question.strip() for question in questions[:-1]]
        return questions

    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        # source_text = source_text + " </s>"

        inputs = self._tokenize([source_text], padding=False)
        return inputs

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512,
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs


class_output = E2EQGPipeline(model, tokenizer, use_cuda=True)(user_utterance)
print(class_output)
