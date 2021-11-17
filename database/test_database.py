import json
from pymongo import MongoClient, ASCENDING, DESCENDING
from collections import OrderedDict

MONGO_DATABASE_URI = "mongodb+srv://Darkfenner_1969:pippo45423@goldengroup0.jchhr.mongodb.net/steve_dbprod?retryWrites=true&w=majority"

client = MongoClient(MONGO_DATABASE_URI)
db = client.steve_dbprod
use_coll = "users"
users = db[use_coll]

training_type = "work"
feature = "EQRating"


class User_stats(object):
    def __init__(self, users, training_type, feature):
        self.users = users
        self.training_type = training_type
        self.feature = feature

    def __call__(self):
        # print("Creating instance")
        if self.training_type == "all" and self.feature == "all":
            return self.overall_data()
        else:
            return self.data_for_training_type()

    def data_for_training_type(self):
        list_of_users = []
        for doc in self.users.find(
            {
                "$and": [
                    {"Total_sess": {"$gt": 0}},
                    {"Coaching_sessions.Type": {"$eq": self.training_type}},
                ]
            },
            {
                "_id": 0,
                "Username": 1,
                "Name": 1,
                "Surname": 1,
                "Coaching_sessions.Type": 1,
                "Coaching_sessions.Scores": 1,
                # "Coaching_sessions.End_time": 1,
            },
        ).sort(
            [
                ("Coaching_sessions.Scores." + self.feature, DESCENDING),
                ("Surname", ASCENDING),
            ]
        ):
            user_dict = {}
            user_dict["username"] = doc["Username"]
            user_dict["name"] = doc["Name"]
            user_dict["surname"] = doc["Surname"]
            x = 0
            sessions = []
            reference = 0
            for elem in doc["Coaching_sessions"]:
                # counting non-empty dicts
                if bool(elem) and elem["Type"] == self.training_type:
                    x += 1
                    if elem["Scores"][self.feature] > reference:
                        if sessions:
                            sessions.pop(0)
                        sessions.append(elem)
                        reference = elem["Scores"][self.feature]

            user_dict["coaching_sessions"] = sessions
            user_dict["completed_sessions"] = x
            user_dict["total_sessions"] = len(doc["Coaching_sessions"])
            # print(user_dict)
            list_of_users.append(user_dict)

        # print()
        # print(list_of_users)
        return list_of_users

    def overall_data(self):
        list_of_users = []
        for doc in self.users.find(
            {
                "$and": [
                    {"Total_sess": {"$gt": 0}},
                ]
            },
            {
                "_id": 0,
                "Username": 1,
                "Name": 1,
                "Surname": 1,
                "Coaching_sessions.Scores.EQRating": 1,
                "Coaching_sessions.Type": 1,
            },
        ).sort([("Surname", ASCENDING)]):
            # print(doc)
            user_dict = {}
            user_dict["username"] = doc["Username"]
            user_dict["name"] = doc["Name"]
            user_dict["surname"] = doc["Surname"]
            x = 0
            sessions = []
            for elem in doc["Coaching_sessions"]:
                # counting non-empty dicts
                if bool(elem):
                    x += 1
                    sessions.append(elem)

            user_dict["coaching_sessions"] = sessions
            user_dict["completed_sessions"] = x
            user_dict["total_sessions"] = len(doc["Coaching_sessions"])
            # print(user_dict)
            # print()
            list_of_users.append(user_dict)
            list_of_users_ordered = sorted(
                list_of_users, key=lambda k: k["completed_sessions"], reverse=True
            )
        # print(list_of_users_ordered)
        return list_of_users_ordered


class A(object):

    # It is not called
    def __init__(self, training_type, feature):
        print("Init is called")
        self.training_type = training_type
        self.feature = feature

    def __call__(self):
        print("Creating instance")
        return self.prova()

    def prova(self):
        x = 3
        return x


if __name__ == "__main__":
    pass

    training_type = "work"
    feature = "EQRating"

    # result = A(training_type, feature)()
    # print(result)

    results = User_stats(users, training_type, feature)()
    print(results)
