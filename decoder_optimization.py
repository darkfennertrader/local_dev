import os
import time
import torch
import pickle

from functools import reduce
import numpy as np
import pandas as pd
import GPUtil
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import optuna

from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs

cuda_available = torch.cuda.is_available()

# model_args = ConvAIArgs()
# # Common Options
# # model_args.max_history = 5
# model_args.cache_dir = "./models/pre-trained/gpt2/gpt2/"
# model_args.encoding = "utf-8"
# model_args.output_dir = "./models/fine-tuned/gpt2/gpt2"
# model_args.overwrite_output_dir = True
# model_args.save_model_every_epoch = True


# model = ConvAIModel(
#     model_type="gpt2",
#     model_name="gpt2",
#     use_cuda=True,
#     cache_dir="./models/pre-trained/gpt2/gpt2/",
#     # args=model_args,
# )


#######################   GPU AVAILABILITY   ################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("\nGPU is OK")
else:
    print("\nGPU is KO")
print(GPUtil.showUtilization())

#############################################################################
# loading the benchmark models in the GPU memory
start = time.time()

# updown: given a context and its two human responses, predict which gets more upvotes?
mod_updown = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-updown",
    cache_dir="./models/pre-trained/dialorpt/model-cards/updown",
).to(device)
print(f"\nDialogRPT-updown loaded")
tok_updown = AutoTokenizer.from_pretrained(
    "microsoft/DialogRPT-updown",
    cache_dir="./models/pre-trained/dialorpt/tokenizers/updown",
)
print(GPUtil.showUtilization())

# depth: given a context and its two human responses, predict which gets longer follow-up thread?
mod_depth = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-depth",
    cache_dir="./models/pre-trained/dialorpt/model-cards/depth",
).to(device)
tok_depth = AutoTokenizer.from_pretrained(
    "microsoft/DialogRPT-depth",
    cache_dir="./models/pre-trained/dialorpt/tokenizers/depth",
)
print(f"\nDialogRPT-depth loaded")
print(GPUtil.showUtilization())

# given a context and its two human responses, predict which gets more direct replies?
mod_width = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-width",
    cache_dir="./models/pre-trained/dialorpt/model-cards/width",
).to(device)
tok_width = AutoTokenizer.from_pretrained(
    "microsoft/DialogRPT-width",
    cache_dir="./models/pre-trained/dialorpt/tokenizers/width",
)
print(f"\nDialogRPT-width loaded")
print(GPUtil.showUtilization())

# given a context and one human response, distinguish it with a random human response
mod_hrand = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-human-vs-rand",
    cache_dir="./models/pre-trained/dialorpt/model-cards/human-vs-rand",
).to(device)
tok_hrand = AutoTokenizer.from_pretrained(
    "microsoft/DialogRPT-human-vs-rand",
    cache_dir="./models/pre-trained/dialorpt/tokenizers/human-vs-rand",
)
print(f"\nDialogRPT-human-vs-rand loaded")
print(GPUtil.showUtilization())

# given a context and one human response, distinguish it with a machine generated response
mod_hmach = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-human-vs-machine",
    cache_dir="./models/pre-trained/dialorpt/model-cards/human-vs-machine",
).to(device)
tok_hmach = AutoTokenizer.from_pretrained(
    "microsoft/DialogRPT-human-vs-machine",
    cache_dir="./models/pre-trained/dialorpt/tokenizers/human-vs-machine",
)
print(f"\nDialogRPT-human-vs-machine loaded")
print(GPUtil.showUtilization())

print(f"\nModels loading finished in: {(time.time() - start):.3f} sec.\n")

########################################################################


class Objective(object):
    def __init__(
        self,
        trial_filename,
        questions,
        personality,
        min_history_chat,
        max_history_chat,
        min_max_length,
        max_max_length,
        max_length_step,
        min_temperature,
        max_temperature,
        temperature_step,
        min_top_k,
        max_top_k,
        top_k_step,
        min_top_p,
        max_top_p,
        top_p_step,
        wt_updown,
        wt_width,
        wt_depth,
        wt_hrand,
        wt_hmach,
    ):
        self.trial_filename = trial_filename
        self.questions = questions
        self.personality = personality
        self.min_history_chat = min_history_chat
        self.max_history_chat = max_history_chat
        self.min_max_length = min_max_length
        self.max_max_length = max_max_length
        self.max_length_step = max_length_step
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.temperature_step = temperature_step
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k
        self.top_k_step = top_k_step
        self.min_top_p = min_top_p
        self.max_top_p = max_top_p
        self.top_p_step = top_p_step
        self.wt_updown = wt_updown
        self.wt_width = wt_width
        self.wt_depth = wt_depth
        self.wt_hrand = wt_hrand
        self.wt_hmach = wt_hmach

    def __call__(self, trial):

        # HYPERPARAMETERS SETTING
        history_chat = trial.suggest_int(
            "history_chat", self.min_history_chat, self.max_history_chat
        )
        max_length = int(
            trial.suggest_discrete_uniform(
                "max_length",
                self.min_max_length,
                self.max_max_length,
                self.max_length_step,
            )
        )

        # whether or not to use sampling ; use greedy decoding otherwise
        # picking randomly the next word according to its conditional probability distribution
        do_sample = True

        temperature = trial.suggest_discrete_uniform(
            "temperature",
            self.min_temperature,
            self.max_temperature,
            self.temperature_step,
        )
        top_k = int(
            trial.suggest_discrete_uniform(
                "top_k", self.min_top_k, self.max_top_k, self.top_k_step
            )
        )
        top_p = trial.suggest_discrete_uniform(
            "top_p", self.min_top_p, self.max_top_p, self.top_p_step
        )
        ########################################################################
        # loading chatbot fine-tuned model in the GPU if possibile
        model_args = ConvAIArgs()
        model_args.manual_seed = 42  # set seed for reproducibility
        model_args.max_history = history_chat
        model_args.max_length = max_length
        model_args.do_sample = do_sample
        model_args.temperature = temperature
        model_args.top_k = top_k
        model_args.top_p = top_p

        model = ConvAIModel(
            model_type="gpt",
            model_name="./models/fine-tuned/gpt/",
            use_cuda=False,
            args=model_args,
        )

        # read bechmark questions from txt file
        trials = self.trial_df(self.trial_filename, self.questions)
        lst = list()
        ans = list()
        history = []
        with open(basic_questions, "r") as f:
            for line in f:
                # print()
                line = line.rstrip("\n").replace('"', "")
                # print("USER: ", line)
                bot_answer, history = model.interact_single(
                    line, history, personality=self.personality
                )

                # print(f"BOT: {bot_answer}")
                # print(f"history: {history}")

                # step takes into consideration the history
                history = [] if len(history) > history_chat else history

                ensemble = self.score(line.strip(), bot_answer)
                # append value and answer to list
                lst.append(ensemble)
                ans.append(bot_answer)
        # store column into trial dataframe for later analysis
        trials["trial_" + str(len(study.trials) - 1)] = lst
        trials["trial_" + str(len(study.trials) - 1) + "_ans"] = ans
        # print(trials.head())

        # average of all chatbot answers to the basic questions
        ensemble = reduce(lambda a, b: a + b, lst) / len(lst)
        print("\n", ensemble)
        # storing dataframe in binary format
        trials.to_pickle("./trials/" + self.trial_filename + ".pkl")

        return ensemble

    def score(self, cxt, hyp):
        # rpt-updown
        model_input = tok_updown.encode(
            cxt + "<|endoftext|>" + hyp, return_tensors="pt"
        ).to(device)
        updown = torch.sigmoid(mod_updown(model_input, return_dict=True).logits).item()

        # rpt - depth
        model_input = tok_depth.encode(
            cxt + "<|endoftext|>" + hyp, return_tensors="pt"
        ).to(device)
        depth = torch.sigmoid(mod_depth(model_input, return_dict=True).logits).item()

        # rpt - width
        model_input = tok_width.encode(
            cxt + "<|endoftext|>" + hyp, return_tensors="pt"
        ).to(device)
        width = torch.sigmoid(mod_width(model_input, return_dict=True).logits).item()

        # rpt - hrand
        model_input = tok_hrand.encode(
            cxt + "<|endoftext|>" + hyp, return_tensors="pt"
        ).to(device)

        hrand = torch.sigmoid(mod_hrand(model_input, return_dict=True).logits).item()
        # rpt - hmach
        model_input = tok_hmach.encode(
            cxt + "<|endoftext|>" + hyp, return_tensors="pt"
        ).to(device)
        hmach = torch.sigmoid(mod_hmach(model_input, return_dict=True).logits).item()
        # print(
        #     f"updown:{updown}, depth:{depth}, width:{width}, hrand:{hrand}, hmach:{hmach}"
        # )
        # ensemble evaluation
        ensemble = (
            (self.wt_updown * updown + self.wt_width * width + self.wt_depth * depth)
            / (self.wt_updown + self.wt_width + self.wt_depth)
        ) * ((wt_hrand * hrand + wt_hmach * hmach) / (self.wt_hrand + self.wt_hmach))

        # ensemble = updown

        # ensemble = (
        #     self.wt_updown * updown + self.wt_width * width + self.wt_depth * depth
        # )

        # torch.cuda.empty_cache()
        return ensemble

    def trial_df(self, trial_filename, basic_questions):
        try:
            # print("unpickling dataframe")
            df = pd.read_pickle("./trials/" + trial_filename + ".pkl")
            # print(df.head(10))
        except FileNotFoundError:
            # print("dataframe creation")
            df = pd.read_csv(basic_questions, header=None, sep="\n\n", engine="python")
            df.rename(columns={0: "Questions"}, inplace=True)
            df = df.applymap(lambda row: row.replace('"', ""))
            # print(df.head(10))
        return df


if __name__ == "__main__":

    trial_filename = "experiment"
    basic_questions = "./basic_questions.txt"
    basic_questions = "./basic_questions copy.txt"
    # number of trials for hyperparameter tuning
    n_trials = 1

    # define chatbot personality
    personality = [
        "My name is Steve Jobs",
        "I'm a Buddhist",
        "I met Chrisann in 1972 when we were in high school together",
        "I've taken acid and pot",
        "I grew up in Mountain View and then later Los Altos, in the Bay Area",
        "I've been a vegan for most of my life",
        "The first Apple product I made was the Apple I",
        "I'm interested in the intersection between technology and art",
        "My father's name is Paul and he is from Syria",
        "My mother's name is Clara and she is from Wisconsin",
        "I don't have any brothers",
        "My sisters' names are Patricia and Mona",
        "My biological father's name is Abdulfattah John Jandali",
        "My biological mother's name is Joanna Schieble",
        "I was born on February 24 1955",
        "I died from pancreatic cancer",
        "I love Bob Dylan, the Beatles and the Rolling Stones",
        "I had a relationship with Joan Baez in my late 20s",
        "I am a rich and billionaire",
        "I'm not really interested in sports",
        "My favorite composer is Bach",
        "I spent 7 months in India",
        "I have been married to Laurene Powell since 1991",
        "I was married only once",
        "I am 6 foot 2 inches",
        "I met my biological father a few times at a restaurant where he worked in San Jose",
        "I attended Homestead High in Cupertino",
        "I tried to keep my political views out of the public sphere",
    ]

    # HYPERPARAMETER TUNING
    # max history to be taken into consideration by the chatbot
    min_history_chat = 1
    max_history_chat = 1

    # maximum length of the sequence to be generated (defaults to 20)
    min_max_length = 100
    max_max_length = 400
    max_length_step = 50

    # scaling factor of softmax function
    # low values make softmax more confident (more conservative)
    # high values makes softmax less confident (more diversity)
    # decreasing temperature decrease sensitivity to low probability candidates (make things less random) (defaults to 1)
    # t=1 outputs weird words
    # t=0 outputs repetitive words
    min_temperature = 0.3
    max_temperature = 0.95
    temperature_step = 0.01

    # Top K sampling (defaults to 50)
    min_top_k = 50
    max_top_k = 200
    top_k_step = 25

    # Top p(nucleus) sampling (defaults to 1.0)
    min_top_p = 0.65
    max_top_p = 0.95
    top_p_step = 0.1

    # EVALUATION ensemble model parameters
    wt_updown = 1
    wt_width = -0.5
    wt_depth = 0.48
    wt_hrand = 0.5
    wt_hmach = 0.5

    start = time.time()
    print("\nHyperparameter optimization has started")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        Objective(
            trial_filename=trial_filename,
            questions=basic_questions,
            personality=personality,
            #################################################
            min_history_chat=min_history_chat,
            max_history_chat=max_history_chat,
            #################################################
            min_max_length=min_max_length,
            max_max_length=max_max_length,
            max_length_step=max_length_step,
            #################################################
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            temperature_step=temperature_step,
            #################################################
            min_top_k=min_top_k,
            max_top_k=max_top_k,
            top_k_step=top_k_step,
            #################################################
            min_top_p=min_top_p,
            max_top_p=max_top_p,
            top_p_step=top_p_step,
            #################################################
            wt_updown=wt_updown,
            wt_width=wt_width,
            wt_depth=wt_depth,
            wt_hrand=wt_hrand,
            wt_hmach=wt_hmach,
        ),
        n_trials=n_trials,
        gc_after_trial=True,  # run garbage collector
    )
    pickle.dump(study, open("./trials/" + trial_filename + "_params.pkl", "wb"))
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    print(f"\nEvaluation finished in: {(time.time() - start):.3f} sec.\n")

    print("\nTrials' parameters")
    print(80 * "-")
    study_trials = pickle.load(open("./trials/" + trial_filename + "_params.pkl", "rb"))
    print("Best trial until now:")
    print(f"Trial no.: {study_trials.best_trial.number}")
    print(f"Value: {study_trials.best_trial.value:.3f}")
    print("Params: ")
    for key, value in study_trials.best_trial.params.items():
        print(f"    {key}: {value}")

    print("\nBasic questions' distribution")
    print(80 * "-")
    unpickled_df = pd.read_pickle("./trials/" + trial_filename + ".pkl")
    unpickled_df.to_excel(trial_filename + ".xlsx")

    # docker cp <container_id>:/simple-transformers/<filename>.xlsx .
