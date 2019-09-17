import arff
from sklearn import metrics


def helloWorld():
    return "HELLO"

def get_features_from_arff(arff_talk):
    attributes = arff_talk["attributes"]
    features = []
    for attribute in attributes:
        features.append(attribute[0])

    #remove name and emotion from features
    features = features[1:-1]
    return features


