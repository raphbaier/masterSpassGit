import tensorflow as tf
#not standard arff, this is LIAC-arff!!!
import arff
import numpy as np
from sklearn import linear_model, datasets, svm, mixture, preprocessing, metrics

#was wir am arff noch ändern müssen: class zu emotion, numeric zu {lower, higher} ganz hinten dann die class die 0.0 war zu ner emotion, also bspw lower


# Load the feature data file
features_file_path = "arffs/A-2018-Q3.arff"
test_features_file_path = "arffs/A-2018-Q3TEST.arff"
features_file_path = "../../one_arff_with_label/A-2018-Q3.arff"
test_features_file_path = "../../one_arff_with_label/A-2018-Q4.arff"
with open(features_file_path) as fh:
    features_data = arff.load(fh)
fh.close()
with open(test_features_file_path) as fh:
    test_features_data = arff.load(fh)
fh.close()

#features_data = arff.load(open(features_file_path, 'rb'))


# list of features to test model
features = [
    'pcm_loudness_sma_upleveltime90', 'logMelFreqBand_sma_de[2]_upleveltime75', 'pcm_fftMag_mfcc_sma[0]_maxPos'
]

#the classes: lower and higher
emotions = [
    'lower', 'higher'
]

#map for each emotion: 1 if it's the emotion, 0 else
def emotion_to_class_tensor(emotion):
    return list(map(lambda x: (1 if x == emotion else 0), emotions))

# Get indices for certain features. You can get the list of all features through features_data['attributes']
# The format of that is a list of tuples: (feature_name, feature_type), below we call the tuples "feature"
def get_feature_indices(feature_names):
    # print(feature_names)
    # print(features_data)
    print(feature_names)
    feature_indices = []
    for feature_name in feature_names:
        print("Name: " + feature_name)
        print(features_data)
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

label_index = get_feature_indices(["emotion"])[0]
feature_indices = get_feature_indices(features)

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

# Parameters
learning_rate = 0.0001
training_epochs = 9000
training_epochs = 100
batch_size = 1
display_step = 10

# Network parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_hidden_3 = 580 # 3rd layer
n_hidden_4 = 16
n_input = len(x_train[0]) # data input
dropout = 0.7

n_classes = len(emotions)

#tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#np.random.seed(10)

def next_batch(batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(x_train_np)))
    x_shuffled = x_train_np[shuffle_indices]
    y_shuffled = y_train_np[shuffle_indices]
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
    print(metrics.confusion_matrix(y_true, y_pred))