import json
import re, unicodedata
import time
import requests
import boto3
import pprint
from requests.exceptions import HTTPError

pp = pprint.PrettyPrinter(indent=4)


class Config(object):
    TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Im84ckE2MlpVcTVpbjV4UDBvaHB0ZSJ9.eyJpc3MiOiJodHRwczovL2Rldi05cnkwNmJsdi51cy5hdXRoMC5jb20vIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMDQwMTI4MjI0NTI2Mjg0ODQ5NDIiLCJhdWQiOiJqczBEMHBEblhNVGl1NkYwSnFseFJtQ3prZkc0NElEYSIsImlhdCI6MTYzNDIwMzU3NiwiZXhwIjoyNDk4MjAzNTc2LCJhdF9oYXNoIjoiUERrcnBJVW1JbzlTN3ZGMHRZSFhCUSIsIm5vbmNlIjoiNkpEQS53eHhONmktVU9wQ3l0cUZ5NUdiVFZyOFIxa0QifQ.o-YfSE157yrMzMZmSpfT2XgauhnVgQixRJYRCZzS8Sw3HkSH1E1yx88Lts_2IpQbBQt1PIBpCdwbRI8AIP3CCYl5yjrK3bBUfBhUlEVmEGrc3_z54mlpqA5kr0N_tZGmye7RDZBLYHEoOclorrUz8jAMvEvH8ITcX3d6JV2GNnwVcVAcMXEgTTsoxMAFLr4s8HaeZ8jjubGX3qDBBK2sT9yjj_ES-xrZy2rbe3Z89PdPyAAX9tvDBbLTev79GuZiOkw6OLamUj0v4cb8BGgEtewj8T35LGJRghjBc3u5OKOa_CTTDUxWBU9Ie8oUsPYY8BtVuBh2hR9sg0bNRQbgRA"

    AWS_API = "https://wc02ao8pne.execute-api.eu-west-1.amazonaws.com/dev/"
    BING_ENDPOINT = "bingsearch"
    GOOGLE_ENDPOINT = "googlesearch"
    BING_URI = AWS_API + BING_ENDPOINT
    GOOGLE_URI = AWS_API + GOOGLE_ENDPOINT
    CROSS_ENCODER_URI = (
        "https://cu3gxquyxb.execute-api.eu-west-1.amazonaws.com/dev/cross-encoder"
    )


def callback(str):
    return str.replace(".", "")


def text_cleaning(text, nr_of_sentences=3, word_per_sentence=3):

    # text_to_search = (
    #     unicodedata.normalize("NFKC", text_to_search).encode("ascii", "ignore").decode()
    # )
    # print()
    # print("before cleaning")
    # print(text)
    # make characters homogenous
    text_cleaned = unicodedata.normalize("NFKC", text)
    # replace non-ascii characters with space
    text_cleaned = re.sub(r"[^\x00-\x7F]", " ", text_cleaned)
    # remove hyperlinks w/wo http(s)
    text_cleaned = re.sub(
        r"((https?://)?(www\.)([a-zA-Z-]+)(\.\w+))|((www\.)([a-zA-Z-]+)(\.\w+))",
        "",
        text_cleaned,
    )
    # remove emails
    text_cleaned = re.sub(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]+", "", text_cleaned
    )
    # remove unnecessary characters
    text_cleaned = re.sub(r"[:;-]+", " ", text_cleaned)
    # remove everything within round brackets
    text_cleaned = re.sub(r"\([^()]*\)", " ", text_cleaned)
    # remove spaces between commas
    text_cleaned = re.sub(r",\s+,", ",", text_cleaned)
    # remove three consecutive dots
    text_cleaned = re.sub(r"\.\.\.", " ", text_cleaned)
    # remove a paraghraph of one to three words at the start of sentence
    text_cleaned = re.sub(r"^(\s*\w+\s*){1,3}\.", " ", text_cleaned)
    # remove a paraghraph of one to three words at the end of sentence
    text_cleaned = re.sub(r"\.(\s+\w+\s*){1,3}\.*$", ".", text_cleaned)
    # remove dots from acronyms
    text_cleaned = re.sub(
        r"(\w{1}\.){2,3}", lambda m: callback(m.group()), text_cleaned
    )
    # remove single/double character(s) followed by a dot and separated by spaces
    # text_cleaned = re.sub(r"(\s+\w{1,2}\.\s+)", " ", text_cleaned)
    text_cleaned = re.sub(r"(\s+[A-Z][a-z]?\.\s+)", " ", text_cleaned)

    # remove unnecessary spaces between words
    text_cleaned = " ".join(text_cleaned.split())
    # print()
    # print("remove unnecessary spaces between words")
    # print(text_cleaned)
    # print()

    # flag last sentence if it doesn't end with a dot
    keep_last_sentence = text_cleaned.endswith(".")
    # print(keep_last_sentence)

    # create list of sentences split by dots
    sentences_list = text_cleaned.split(".")
    # print("create list of sentences split by dots")
    # print(sentences_list)
    # print()

    # remove last sentence if flag is False
    if not keep_last_sentence:
        sentences_list.pop()

    # remove sentences composed of less than three words
    sentences_list = [
        sentences_list[i].strip()
        for i in range(len(sentences_list))
        if len(sentences_list[i].split()) > word_per_sentence
    ]

    # return first N=3 sentences
    return ". ".join(sentences_list[:nr_of_sentences]) + ".", sentences_list


def bing_search(
    query, count=5, length=400, token=Config.TOKEN, bing_uri=Config.BING_URI
):
    # make query
    payload = dict()
    payload["query"] = query
    payload["count"] = str(count)
    payload["length"] = str(length)

    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    try:
        response = requests.post(bing_uri, headers=headers, json=payload, timeout=3)
        print(response.status_code)
        return response.json()["items"], 200

    except Exception as e:
        print(e)
        return ["Sorry, but I don't know how to answer your question"], 500


############   SageMaker Cross-Encoder Model Initialization   #############


def internet_wizard(query, sentences_list):
    # AI Cross-Encoder
    ce_model_URL = Config.CROSS_ENCODER_URI
    # print(ce_model_URL)

    payload = {
        "query": query,
        "sentences_list": sentences_list,
    }

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # ce_model_URL = "https://httpbin.org/post"

    try:
        response = requests.post(ce_model_URL, headers=headers, json=payload, timeout=3)
        # print(dir(response))
        # print()
        # print(response.raise_for_status())
        # print(response.json())
        # print(f"best_sentence: {response.json()['best_sentence']}")
        # print(response.status_code)
        response.raise_for_status()
        return response.json()["best_sentence"], response.status_code

    except HTTPError as http_err:
        # print(dir(http_err.response))

        if http_err.response.status_code == 400:
            error_message = http_err.response.json()["message"]
        if http_err.response.status_code == 404:
            error_message = http_err.response.json()["message"]
        elif http_err.response.status_code == 503:
            error_message = f"{http_err.response.reason} for {http_err.response.url}"
        else:
            error_message = http_err

        return error_message, None

    except Exception as err:
        print(dir(err))
        print(err.response)
        print(f"Other error occurred: {err}")


if __name__ == "__main__":
    query = "Who built Rome?"
    start = time.time()
    raw_searches, status_code = bing_search(query)
    print(f"internet search took {(time.time() - start):3f} sec.")

    if status_code == 200:
        BOT_answer, status_code = internet_wizard(query, raw_searches)
        if status_code == 200:
            print()
            print("raw choice:")
            print(BOT_answer)
            print(f"\nTotal time elapsed: {(time.time() - start):3f} sec.")
            text_cleaned, sentences_list = text_cleaning(BOT_answer)
            print("\n")
            print("choice after noise reduction")
            print(text_cleaned)
            print(f"\nTotal time after cleaning: {(time.time() - start):3f} sec.")
            # paraphrase mining
            for i in range(len(sentences_list) - 1):
                pass

        else:
            print()
            print("ERROR MESSAGE:")
            print(BOT_answer)

    else:
        print()
        print("ERROR MESSAGE:")
        print()


##################### using LOCAL Cross-Encode Model ######################
# import numpy as np
# from sentence_transformers import CrossEncoder

# model = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=1024, device="cuda"
# )
# scores = model.predict(list_to_rank)
# print(scores)

# print(np.argmax(np.asarray(scores)))

# print()
