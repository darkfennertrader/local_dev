import json
import re, unicodedata
import time
import requests
import boto3
import pprint
from requests.exceptions import HTTPError
from keybert import KeyBERT
import numpy as np
from sentence_transformers import CrossEncoder

pp = pprint.PrettyPrinter(indent=4)


# cross-encoder INIT (for keyword extraction)
cross_encoder = "/home/solidsnake/ai/Golden_Group/ai-models/development/cross-encoders/ms-marco-MiniLM-L-12-v2"

model = CrossEncoder(cross_encoder, max_length=1024, device="cuda")
kw_model = KeyBERT(cross_encoder)


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
        # print(response.status_code)
        return response.json()["items"], 200

    except Exception as e:
        print(e)
        return ["Sorry, but I don't know how to answer your question"], 500


############   SageMaker Cross-Encoder Model Initialization   #############


def internet_wizard(query, sentences_list, use_sagemaker=True):

    if use_sagemaker:
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
            response = requests.post(
                ce_model_URL, headers=headers, json=payload, timeout=3
            )
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
                error_message = (
                    f"{http_err.response.reason} for {http_err.response.url}"
                )
            else:
                error_message = http_err

            return error_message, None

        except Exception as err:
            print(dir(err))
            print(err.response)
            print(f"Other error occurred: {err}")

    # use local model
    else:
        sentences_combinations = [[query, sentence] for sentence in sentences_list]
        # print(sentences_combinations)
        scores = model.predict(sentences_combinations)
        # print(scores)
        max_idx = np.argmax(np.asarray(scores))
        # print(np.argmax(np.asarray(scores)))
        return sentences_list[max_idx], 200


def keywords_extraction(query):
    # extracting keywords/keyphrases from user utterance
    keywords = kw_model.extract_keywords(query)
    try:
        k_words = kw_model.extract_keywords(
            query,
            keyphrase_ngram_range=(2, 4),
            stop_words="english",
            use_maxsum=True,
            nr_candidates=3,
            top_n=1,
        )
        print()
        print(k_words)
        return sorted(k_words, key=lambda similarity: similarity[1], reverse=True)[0][0]

    except Exception:
        return "Steve Jobs"


if __name__ == "__main__":
    ###################################################
    use_sagemaker = False
    query = "love you"
    ###################################################

    partial_time_computation = []
    start = time.time()
    query_modified = keywords_extraction(query)
    print(f"Keywords extraction took {(time.time() - start):3f} sec.")
    partial_time_computation.append(time.time() - start)
    # print(query_modified)

    start = time.time()
    raw_searches, status_code = bing_search(query_modified)
    print(f"\nInternet search took {(time.time() - start):3f} sec.")
    partial_time_computation.append(time.time() - start)

    if status_code == 200:
        start = time.time()
        BOT_answer, status_code = internet_wizard(
            query, raw_searches, use_sagemaker=use_sagemaker
        )
        if status_code == 200:
            print()
            print("raw choice:")
            print(BOT_answer)
            print(f"\nCross-Encoder took: {(time.time() - start):3f} sec.")
            partial_time_computation.append(time.time() - start)

            start = time.time()
            text_cleaned, sentences_list = text_cleaning(BOT_answer)
            print("\nBOT answer after cleaning:")
            print(text_cleaned)
            print(f"\nTime after cleaning: {(time.time() - start):3f} sec.")
            partial_time_computation.append(time.time() - start)

            print(f"\nTotal Elapsed Time: {sum(partial_time_computation):3f} sec.")

        else:
            print()
            print("ERROR MESSAGE:")
            print(BOT_answer)

    else:
        print()
        print("ERROR MESSAGE:")
        print()
