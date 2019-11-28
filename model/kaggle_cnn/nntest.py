import arff


with open("input_arff/male/angry/_JE_a03.arff") as sheiss:
    test = arff.load(sheiss)
sheiss.close()

print(test)
print(test["attributes"])
print(test["data"])


with open("input_arff/male/disgust/_03-01-07-01-01-01-01.arff") as sheiss:
    test["data"] += arff.load(sheiss)["data"]
sheiss.close()

print(test["attributes"])
print(test["data"])