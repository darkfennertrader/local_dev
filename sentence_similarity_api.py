import json
import re, unicodedata
import requests
import boto3
from aws_requests_auth.aws_auth import AWSRequestsAuth
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth
import pprint
from requests.auth import HTTPBasicAuth  # authentication with user and password
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


def text_cleaning(text):
    # normalize characters
    text = unicodedata.normalize("NFKC", text)
    # replace non ascii code with space
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    # remove unnecessary characters
    text = re.sub(r"[:;-]+", " ", text)
    # remove everything within round brackets
    text = re.sub(r"\([^()]*\)", " ", text)
    # remove three consecutive dots
    text = re.sub(r"\.\.\.", " ", text)
    # remove emails
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text)
    # T.B.D.
    # remove a paraghraph of one to three words at the start of sentence
    # remove a paraghraph of one to three words at the end of sentence
    # remove all paragraphs with one to three words
    # remove text after three consecutive symbols among (dot, question mark, exclamation mark) (too long for the BOT)

    return " ".join(text.split())


############   SageMaker Cross-Encoder Model Initialization   #############

# AI Cross-Encoder
internet_token = Config.TOKEN
aws_bing_api = Config.BING_URI
ce_model_URL = Config.CROSS_ENCODER_URI
# print(ce_model_URL)


payload = {
    "query": "How many people live in Berlin?",
    "sentences_list": [
        "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        "New York City is famous for the Metropolitan Museum of Art.",
        "Berlin is a beautiful city",
        "London has a population of around 8 million registered inhabitants",
        "people who live Berlin are very unfriendly",
    ],
}

headers = {"Content-Type": "application/json", "Accept": "application/json"}

# ce_model_URL = "https://httpbin.org/post"

try:
    response = requests.post(ce_model_URL, headers=headers, json=payload, timeout=3)
    # print(dir(response))
    print()
    print(response.raise_for_status())
    print()
    print(response.headers)
    print()
    print(response.status_code)
    print(response.json())
    # print(f"best_sentence: {response['best_sentence']}")
    # print(response["statusCode"]

except HTTPError as http_err:
    # print(dir(http_err.response))

    if http_err.response.status_code == 400:
        print(http_err.response.json()["message"])
    elif http_err.response.status_code == 503:
        print(f"{http_err.response.reason} for {http_err.response.url}")
    else:
        print(http_err)

except Exception as err:
    # print(dir(err))
    # print(err.response)
    print(f"Other error occurred: {err}")


################################################################################


# # make query
# question = "Who won the last world cup?"
# payload = dict()
# payload["query"] = question
# payload["count"] = "5"
# payload["length"] = "400"

# headers = {"Accept": "application/json", "Authorization": f"Bearer {Config.TOKEN}"}

# code_response = response.raise_for_status()
# print(response.status_code)

# try:
#     response = requests.post(Config.BING_URI, headers=headers, json=payload, timeout=3)

# except Exception as e:
#     pass


# if not response:
#     print("there was some exception with the API Request")
#     print("Sorry, but I don't know how to answer your question")
# else:
#     print("processing response")
#     print(response.json())

# list_to_rank = []
# for resp in response.json()["items"]:
#     list_to_rank.append((question, resp))

# print(list_to_rank)


# import numpy as np
# from sentence_transformers import CrossEncoder

# model = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=1024, device="cuda"
# )
# scores = model.predict(list_to_rank)
# print(scores)

# print(np.argmax(np.asarray(scores)))

# print()


################################################################################
# aws_host = "https://cu3gxquyxb.execute-api.eu-west-1.amazonaws.com/dev/"
# similarity_api_dev = aws_host + "sentences-similarity"

# # print(aws_host)
# # print(similarity_api_dev)


# query_params = {
#     "user_input": "I love going to the mountains",
#     "true_sentence": "i love the beach",
# }

# try:
#     response = requests.get(similarity_api_dev, params=query_params, timeout=3)
#     # print(dir(response))
#     if response.status_code == 400:
#         print(f"error message: {response.json()['message']}")
#         print(f"status code: {response.status_code}")
#     else:
#         pp.pprint(response.json())
#         print(response.status_code)

# except HTTPError as http_err:
#     # print(dir(http_err.response))

#     if http_err.response.status_code == 400:
#         print(http_err.response.json()["message"])
#     elif http_err.response.status_code == 503:
#         print(f"{http_err.response.reason} for {http_err.response.url}")
#     else:
#         print(f"other errors {http_err}")

# except Exception as err:
#     # print(dir(err))
#     # print(err.response)
#     print(f"Other error occurred: {err}")
