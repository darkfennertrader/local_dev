# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer

# dataset = load_breast_cancer()

# sns.set_style("dark")
# import matplotlib as mpl

# mpl.style.use(
#     [
#         "https://gist.githubusercontent.com/BrendanMartin/01e71bb9550774e2ccff3af7574c0020/raw/6fa9681c7d0232d34c9271de9be150e584e606fe/lds_default.mplstyle"
#     ]
# )
# mpl.rcParams.update({"figure.figsize": (8, 6), "axes.titlepad": 22.0})

# print("Target variables  : ", dataset["target_names"])

# (unique, counts) = np.unique(dataset["target"], return_counts=True)

# print("Unique values of the target variable", unique)
# print("Counts of the target variable :", counts)

# sns.barplot(x=dataset["target_names"], y=counts)
# plt.title("Target variable counts in dataset")
# plt.show()

# from sklearn.preprocessing import StandardScaler

# standardizer = StandardScaler()
# X = standardizer.fit_transform(X)

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=0
# )

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()
# model.fit(X_train, y_train)

# predictions = model.predict(X_test)

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, predictions)

# TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()

# print("True Positive(TP)  = ", TP)
# print("False Positive(FP) = ", FP)
# print("True Negative(TN)  = ", TN)
# print("False Negative(FN) = ", FN)

# accuracy = (TP + TN) / (TP + FP + TN + FN)

# print("Accuracy of the binary classification = {:0.3f}".format(accuracy))

import json
import itertools
import pickle
import numpy as np
import pandas as pd
from pprint import pprint

from sentence_transformers import SentenceTransformer

# import seaborn as sns
import time
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

model_name_or_path = "/home/solidsnake/ai/Golden_Group/ai-models/development/sentence-embeddings/all-mpnet-base-v2/"
se_model = SentenceTransformer(model_name_or_path, device="cuda")


def build_dataframe():
    # extracting generic questions
    df = pd.read_csv(
        "./data/docleaderboard-queries.tsv", sep="\t", usecols=[1], header=None
    )
    df.columns = ["question"]
    df["label"] = "generic"
    # print(df.head())

    print(f"generic dataframe length: {len(df.index)}")

    # extracting Steve Jobs'life questions
    files = [
        "variant_1.json",
        # "variant_2.json",
        # "variant_3.json",
        # "variant_4.json",
        # "variant_5.json",
        # "variant_6.json",
        # "variant_7.json",
        # "variant_8.json",
        "variant_9.json",
    ]

    steve_phrases = []
    for _file in files:
        with open("./training_data/" + _file) as f:
            steve_data = json.load(f)

        # pprint(steve_data["utterances"][:1], indent=2)
        history_list = steve_data["utterances"]

        previuos_len = 0
        for idx, elem in enumerate(history_list):
            if len(elem["history"]) < previuos_len:
                # pprint(elem["history"], indent=2)
                # print(history_list[idx - 1]["history"])
                [
                    steve_phrases.append(phrase)
                    for phrase in history_list[idx - 1]["history"]
                ]

            elif elem == history_list[-1]:
                # pprint(elem["history"], indent=2)
                [steve_phrases.append(phrase) for phrase in elem["history"]]

            previuos_len = len(elem["history"])

        # print(len(steve_phrases))
        # pprint(steve_phrases[:30], indent=2)

    steve_data = pd.DataFrame(steve_phrases, columns=["question"])
    steve_data["label"] = "steve_jobs"
    # print(len(steve_data.index))
    print(f"steve jobs dataframe length: {len(steve_data.index)}")

    df_final = df.append(steve_data).sample(frac=1)
    print()
    print(df_final.head(10))
    print()
    print(f"final dataframe length: {len(df_final.index)}")

    df_final.to_pickle("./data/dataset.pkl")


def train_with_ml():
    # load pickle file
    df = pd.read_pickle("./data/dataset.pkl")

    X = df["question"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print()
    print(f"train samples: {X_train.shape}")
    print(f"test samples: {X_test.shape}")

    # Feature Extraction (sentence-embeddings)
    X_train_embedded = se_model.encode(X_train.to_list())
    X_test_embedded = se_model.encode(X_test.to_list())
    # print(type(X_train_embedded))
    print(X_train_embedded.shape)

    models = {}
    models["Logistic Regression"] = LogisticRegression(n_jobs=1, random_state=42)

    accuracy, precision, recall = {}, {}, {}
    for key in models.keys():
        models[key].fit(X_train_embedded, y_train)
        predictions = models[key].predict(X_test_embedded)
        print()
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))
        # Calculate Accuracy, Precision and Recall Metrics
        # accuracy[key] = accuracy_score(predictions, y_test)
        # precision[key] = precision_score(predictions, y_test)
        # recall[key] = recall_score(predictions, y_test)
        # df_model = pd.DataFrame(
        #     index=models.keys(), columns=["Accuracy", "Precision", "Recall"]
        # )
        # df_model["Accuracy"] = accuracy.values()
        # df_model["Precision"] = precision.values()
        # df_model["Recall"] = recall.values()
        # print(df)

        # pickle model
        filename = "/home/solidsnake/ai/Golden_Group/ai-models/development/classifier/logist-regression-clf.pkl"
        pickle.dump(models[key], open(filename, "wb"))
        filename = "./data/logist-regression-clf.pkl"
        pickle.dump(models[key], open(filename, "wb"))

    return models[key]


# print(df_model)


# accuracy, precision, recall = {}, {}, {}
# for key in models.keys():
#     # Fit the classifier model
#     models[key].fit(X_train, y_train)
#     # Prediction
#     predictions = models[key].predict(X_test)
#     # Calculate Accuracy, Precision and Recall Metrics
#     accuracy[key] = accuracy_score(predictions, y_test)
#     precision[key] = precision_score(predictions, y_test)
#     recall[key] = recall_score(predictions, y_test)


if __name__ == "__main__":

    filename = "./data/logist-regression-clf.pkl"

    # build and pickle dataset if it doesn't exist
    build_dataset = False
    train = True

    if build_dataset:
        build_dataframe()

    if train:
        # train dataset with different ML algorithms
        clf = train_with_ml()

    # predict
    clf = pickle.load(open(filename, "rb"))

    user_utterances = ["did you take acids?"]
    print("\nResults:")
    print()

    for utt in user_utterances:
        embedded_utterance = se_model.encode([utt])
        prediction = clf.predict(embedded_utterance)[0]
        print(
            f"user utterance: {utt}, classification: {prediction} with prob: { clf.predict_proba(embedded_utterance)[0]}"
        )

    # performances = []
    # for _ in itertools.repeat(None, 50):
    #     for utt in user_utterances:
    #         start = time.time()
    #         embedded_utterance = se_model.encode([utt])
    #         prediction = clf.predict(embedded_utterance)[0]
    #         performances.append(time.time() - start)
    #         # print(f"user utterance: {utt}, classification: {prediction}")

    # print(
    #     f"Average time per Prediction: {sum(performances)/len(performances):.3f} sec."
    # )


##########################################################################
# models["Naive Bayes"] = GaussianNB()
# models["Support Vector Machines"] = LinearSVC()
# models["Decision Trees"] = DecisionTreeClassifier()
# models["Random Forest"] = RandomForestClassifier()
# models["K-Nearest Neighbor"] = KNeighborsClassifier()

# accuracy, precision, recall = {}, {}, {}
# for key in models.keys():
#     # Fit the classifier model
#     models[key].fit(X_train, y_train)
#     # Prediction
#     predictions = models[key].predict(X_test)
#     # Calculate Accuracy, Precision and Recall Metrics
#     accuracy[key] = accuracy_score(predictions, y_test)
#     precision[key] = precision_score(predictions, y_test)
#     recall[key] = recall_score(predictions, y_test)

# df_model = pd.DataFrame(
#     index=models.keys(), columns=["Accuracy", "Precision", "Recall"]
# )
# df_model["Accuracy"] = accuracy.values()
# df_model["Precision"] = precision.values()
# df_model["Recall"] = recall.values()

# print(df_model)

# ax = df_model.plot.bar(rot=45)
# ax.legend(
#     ncol=len(models.keys()), bbox_to_anchor=(0, 1), loc="lower left", prop={"size": 14}
# )
# plt.tight_layout()
