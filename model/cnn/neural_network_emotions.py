import tensorflow as tf
#not standard arff, this is LIAC-arff!!!
import arff
import numpy as np
from sklearn import linear_model, datasets, svm, mixture, preprocessing, metrics
import os, os.path
import random

from feature_extraction import extract_features


#for testing
from time import sleep


NUMBER_ARFFS_IN_BATCH = 20


ARFF_FILES_PATH = "input_arff"

ARFF_FILES_PATH_SORTED_ANGRY = "input_arff/male/angry/"
ARFF_FILES_PATH_SORTED_DISGUST = "input_arff/male/disgust/"
ARFF_FILES_PATH_SORTED_FEAR = "input_arff/male/fear/"
ARFF_FILES_PATH_SORTED_HAPPY = "input_arff/male/happy/"
ARFF_FILES_PATH_SORTED_NEUTRAL = "input_arff/male/neutral/"
ARFF_FILES_PATH_SORTED_SAD = "input_arff/male/sad/"
ARFF_FILES_PATH_SORTED_SURPRISE = "input_arff/male/surprise/"



### Load training and test data ###
#Get number of earnings talks

talks_number_angry = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_ANGRY)])
talks_number_disgust = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_DISGUST)])
talks_number_fear = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_FEAR)])
talks_number_happy = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_HAPPY)])
talks_number_neutral = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_NEUTRAL)])
talks_number_sad = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_SAD)])
talks_number_surprise = len([name for name in os.listdir(ARFF_FILES_PATH_SORTED_SURPRISE)])


#Load indices for all arffs and shuffle them
indices_of_talks_angry = list(range(0, talks_number_angry))
indices_of_talks_disgust = list(range(0, talks_number_disgust))
indices_of_talks_fear = list(range(0, talks_number_fear))
indices_of_talks_happy = list(range(0, talks_number_happy))
indices_of_talks_neutral = list(range(0, talks_number_neutral))
indices_of_talks_sad = list(range(0, talks_number_sad))
indices_of_talks_surprise = list(range(0, talks_number_surprise))

random.shuffle(indices_of_talks_angry)
random.shuffle(indices_of_talks_disgust)
random.shuffle(indices_of_talks_fear)
random.shuffle(indices_of_talks_happy)
random.shuffle(indices_of_talks_neutral)
random.shuffle(indices_of_talks_sad)
random.shuffle(indices_of_talks_surprise)

#assign indices to training and testing data
split_point_angry = round(talks_number_angry*3/4)
split_point_disgust = round(talks_number_disgust*3/4)
split_point_fear = round(talks_number_fear*3/4)
split_point_happy = round(talks_number_happy*3/4)
split_point_neutral = round(talks_number_neutral*3/4)
split_point_sad = round(talks_number_sad*3/4)
split_point_surprise = round(talks_number_surprise*3/4)


train_indices_angry = indices_of_talks_angry[:split_point_angry]
train_indices_disgust = indices_of_talks_disgust[:split_point_disgust]
test_indices_angry = indices_of_talks_angry[split_point_angry:]
test_indices_disgust = indices_of_talks_disgust[split_point_disgust:]
train_indices_fear = indices_of_talks_fear[:split_point_fear]
train_indices_happy = indices_of_talks_happy[:split_point_happy]
test_indices_fear = indices_of_talks_fear[split_point_fear:]
test_indices_happy = indices_of_talks_happy[split_point_happy:]
train_indices_neutral = indices_of_talks_neutral[:split_point_neutral]
train_indices_sad = indices_of_talks_sad[:split_point_sad]
test_indices_neutral = indices_of_talks_neutral[split_point_neutral:]
test_indices_sad = indices_of_talks_sad[split_point_sad:]
train_indices_surprise = indices_of_talks_surprise[:split_point_surprise]
test_indices_surprise = indices_of_talks_surprise[split_point_surprise:]

#load data
def load_arffs_from_disk(number_arffs, index_list, path):
    random.shuffle(index_list)
    train_arffs = []
    counter = 0
    for file in os.listdir(path):
        if counter in index_list[:number_arffs]:
            if train_arffs != []:
                with open(path + file) as fh:
                    loaded = arff.load(fh)
                    train_arffs["data"] += loaded["data"]
                fh.close()
            else:
                #print(path + file)
                with open(path + file) as fh:
                    train_arffs = arff.load(fh)
                fh.close()
        counter += 1

    return train_arffs



features_data = load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_angry, ARFF_FILES_PATH_SORTED_ANGRY)
features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_disgust, ARFF_FILES_PATH_SORTED_DISGUST)["data"]
features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_fear, ARFF_FILES_PATH_SORTED_FEAR)["data"]
features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_happy, ARFF_FILES_PATH_SORTED_HAPPY)["data"]
features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_neutral, ARFF_FILES_PATH_SORTED_NEUTRAL)["data"]
features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_sad, ARFF_FILES_PATH_SORTED_SAD)["data"]
features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_surprise, ARFF_FILES_PATH_SORTED_SURPRISE)["data"]


test_features_data = load_arffs_from_disk(111, test_indices_angry, ARFF_FILES_PATH_SORTED_ANGRY)
test_features_data["data"] += load_arffs_from_disk(111, test_indices_disgust, ARFF_FILES_PATH_SORTED_DISGUST)["data"]
test_features_data["data"] += load_arffs_from_disk(111, test_indices_fear, ARFF_FILES_PATH_SORTED_FEAR)["data"]
test_features_data["data"] += load_arffs_from_disk(111, test_indices_happy, ARFF_FILES_PATH_SORTED_HAPPY)["data"]
test_features_data["data"] += load_arffs_from_disk(111, test_indices_neutral, ARFF_FILES_PATH_SORTED_NEUTRAL)["data"]
test_features_data["data"] += load_arffs_from_disk(111, test_indices_sad, ARFF_FILES_PATH_SORTED_SAD)["data"]
test_features_data["data"] += load_arffs_from_disk(111, test_indices_surprise, ARFF_FILES_PATH_SORTED_SURPRISE)["data"]


#sleep(5)


#features_data = arff.load(open(features_file_path, 'rb'))


# list of features to test model
#features = [
#    'pcm_loudness_sma_upleveltime90', 'logMelFreqBand_sma_de[2]_upleveltime75', 'pcm_fftMag_mfcc_sma[0]_maxPos'
#]

print(extract_features.helloWorld())

features = extract_features.get_features_from_arff(features_data)

print("OKOKOK")
print(features)

new_features = []
for meh in features:
    if "mfcc" in meh:
        new_features.append(meh)

features = new_features



#the classes: lower and higher
emotions = [
    0, 1, 2, 3, 4, 5, 6
]

#map for each emotion: 1 if it's the emotion, 0 else
def emotion_to_class_tensor(emotion):
    return list(map(lambda x: (1 if x == emotion else 0), emotions))

# Get indices for certain features. You can get the list of all features through features_data['attributes']
# The format of that is a list of tuples: (feature_name, feature_type), below we call the tuples "feature"
def get_feature_indices(feature_names):
    # print(feature_names)
    # print(features_data)
    #print(feature_names)
    #print(features_data['attributes'])
    feature_indices = []
    for feature_name in feature_names:
        #print(feature_name)
        feature_indices.append(
            [index for index, feature in enumerate(features_data['attributes']) if feature[0] == feature_name][0]
        )
    return feature_indices

def create_limited_feature_vector(feature_values, feature_indices, label_index):
    # Structure is feature_values[sample][feature]
    x = []
    y = []
    for current_feature_values in feature_values:
        x.append([current_feature_values[i] for i in feature_indices])
        y.append(current_feature_values[label_index])
    return x, y

label_index = get_feature_indices(["class"])[0]
feature_indices = get_feature_indices(features)
print("OKOK")
print(len(feature_indices))

x_train, y_train = create_limited_feature_vector(features_data['data'], feature_indices, label_index)
x_test, y_test = create_limited_feature_vector(test_features_data['data'], feature_indices, label_index)

#scale it between 0 and 1
scaler = preprocessing.MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train_numeric = list(map(lambda x: emotion_to_class_tensor(x), y_train))
y_test_numeric = list(map(lambda x: emotion_to_class_tensor(x), y_test))

x_train_np = np.array(x_train)
y_train_np = np.array(y_train_numeric)


#länge x_train is pro arff so 600. nach jeder batchsize holen wir a komplett neues arff set. also am besten setzen wir batchsize
#später halt auf länge von x_train.
print("BATCHSIZE")
print(len(x_train))

# Parameters
learning_rate = 0.0001
learning_rate = 0.001
#learning_rate = 0.00001
training_epochs = 9000
training_epochs = 9000

#wie oben gesagt...
batch_size = 140

display_step = 1

# Network parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_hidden_3 = 256 # 3rd layer
n_hidden_4 = 16




n_input = len(x_train[0]) # data input
dropout = 0.7

n_classes = len(emotions)

#tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#np.random.seed(10)

def next_batch(batch_size):

    #FUER TESTZWECKE

    features_data = load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_angry, ARFF_FILES_PATH_SORTED_ANGRY)
    features_data["data"] += \
    load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_disgust, ARFF_FILES_PATH_SORTED_DISGUST)["data"]
    features_data["data"] += \
    load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_fear, ARFF_FILES_PATH_SORTED_FEAR)["data"]
    features_data["data"] += \
    load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_happy, ARFF_FILES_PATH_SORTED_HAPPY)["data"]
    features_data["data"] += \
    load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_neutral, ARFF_FILES_PATH_SORTED_NEUTRAL)["data"]
    features_data["data"] += load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_sad, ARFF_FILES_PATH_SORTED_SAD)[
        "data"]
    features_data["data"] += \
    load_arffs_from_disk(NUMBER_ARFFS_IN_BATCH, train_indices_surprise, ARFF_FILES_PATH_SORTED_SURPRISE)["data"]


    x_train, y_train = create_limited_feature_vector(features_data['data'], feature_indices, label_index)

    y_train_numeric = list(map(lambda x: emotion_to_class_tensor(x), y_train))

    x_train_np = np.array(x_train)
    y_train_np = np.array(y_train_numeric)





    shuffle_indices = np.random.permutation(np.arange(len(x_train_np)))
    x_shuffled = x_train_np[shuffle_indices]
    y_shuffled = y_train_np[shuffle_indices]


    print(x_shuffled)
    print("DESWARSX")
    print(y_shuffled)

    return x_shuffled[:batch_size], y_shuffled[:batch_size]

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #layer_1 = tf.nn.dropout(layer_1, dropout)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_2 = tf.nn.dropout(layer_2, dropout)

    # Hidden layer with RELU activation
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.relu(layer_3)

    # Hidden layer with RELI activation
    #layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    #layer_4 = tf.nn.relu(layer_4)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
                      # Uncomment the following to apply l2_loss to layers
                      # + 0.01*tf.nn.l2_loss(weights['h1'])
                      # + 0.01*tf.nn.l2_loss(weights['h2'])
                      # + 0.01*tf.nn.l2_loss(weights['out'])
                      # + 0.01*tf.nn.l2_loss(biases['b1'])
                      # + 0.01*tf.nn.l2_loss(biases['b2'])
                      # + 0.01*tf.nn.l2_loss(biases['out'])
                      )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(x_train) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))



        if epoch % 1 == 0:
            y_p = tf.argmax(pred, 1)
            correct_prediction = tf.equal(y_p, tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Traing Data Metrics:")
            val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_train, y: y_train_numeric})
            print("validation accuracy: ", "{:.9f}".format(val_accuracy))
            y_true = np.argmax(y_train_numeric, 1)

            print("Precision:", metrics.precision_score(y_true, y_pred, average=None))
            print("Recall:", metrics.recall_score(y_true, y_pred, average=None))
            print("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
            print("confusion_matrix")
            print(metrics.confusion_matrix(y_true, y_pred))

            # metrics
            print("Test Data Metrics:")
            val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_test, y: y_test_numeric})
            print("validation accuracy: ", "{:.9f}".format(val_accuracy))
            y_true = np.argmax(y_test_numeric, 1)
            print(metrics.confusion_matrix(y_true, y_pred))



    print("Optimization Finished!")

    # Test model
    y_p = tf.argmax(pred, 1)
    correct_prediction = tf.equal(y_p, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Traing Data Metrics:")
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_train, y: y_train_numeric})
    print("validation accuracy: ", "{:.9f}".format(val_accuracy))
    y_true = np.argmax(y_train_numeric, 1)
    # print ("Y_true: ", y_true)
    # print ("Y_pred: ", y_pred)
    print("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
    print("confusion_matrix")
    print(metrics.confusion_matrix(y_true, y_pred))

    # metrics
    print("Test Data Metrics:")
    val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: x_test, y: y_test_numeric})
    print("validation accuracy: ", "{:.9f}".format(val_accuracy))
    y_true = np.argmax(y_test_numeric, 1)
    # print ("Y_true: ", y_true)
    # print ("Y_pred: ", y_pred)
    print("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print("f1_score:", metrics.f1_score(y_true, y_pred, average=None))
    print("confusion_matrix")
    print(y_true)
    print(y_pred)
    print(metrics.confusion_matrix(y_true, y_pred))