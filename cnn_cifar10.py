from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

from tflearn.datasets import cifar10

# Set logging info level
tf.logging.set_verbosity(tf.logging.INFO)

# CNN model function
def cnn_model_fn(features, labels, mode): # features: first argument from input_fn. labels: second argument from input_fn
    
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3]) # batch_size = -1 means it is dynamiclly computed according to input size
    
    # Convolutional layers and pooling layers
    conv1 = tf.layers.conv2d(input_layer, 32, 3, padding='same', activation=tf.nn.relu) # number of filters: 32, kernel_size: 3. [100,32,32,32]
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2) # [100,16,16,32]
    conv2 = tf.layers.conv2d(pool1, 64, 3, padding='same', activation=tf.nn.relu) # [100,16,16,64]
    conv3 = tf.layers.conv2d(conv2, 64, 3, padding='same', activation=tf.nn.relu) # [100,16,16,64]
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2) # [100,8,8,64]

    # Dense layer
    pool2_flat = tf.reshape(pool3, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
    droupout = tf.layers.dropout(dense, rate=0.4, noise_shape=None, seed=None, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    logits = tf.layers.dense(droupout, units=10) # 10 categories

    # Predictions for PREDICT or EVAL(test) mode
    predictions = {
        "categories": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor") # Apply softmax fn to extract probabilities
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss for TRAIN and EVAL(test) modes
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Optimizer for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics for EVAL(test) mode
    test_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["categories"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=test_metric_ops)

# Format the logging info
def logging_formatter(values_to_log):
    stats = []
    for key in values_to_log:
        stats.append("%s = %s" % (key, values_to_log[key][0]))
    return str(stats[0])


def main(unused_argv):
    # Load data from CIFAR10
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data = train_data.astype(np.float32, copy=False)
    test_data = test_data.astype(np.float32, copy=False)
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    print("Data loading finished")
    print(train_data.shape)

    # # Preprocessing images
    # prep1 = tf.map_fn(lambda imgs: tf.image.random_flip_left_right(imgs), train_data)
    # prep2 = tf.map_fn(lambda imgs: tf.image.random_saturation(imgs, lower=0.4, upper=1.6), prep1)
    # print(prep2.shape)
    print(train_labels.shape)
    # prep3 = tf.map_fn(lambda imgs: tf.contrib.keras.preprocessing.image.random_rotation(imgs, rg=25,row_axis=0,col_axis=1,channel_axis=2), prep2)
    # print("Preprocessing finished")

    # Create the estimator
    cifar10_classifier = tf.estimator.Estimator(cnn_model_fn, model_dir="/tmp/convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100, formatter=logging_formatter)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": train_data}, train_labels, batch_size=100, num_epochs=None, shuffle=True)
    cifar10_classifier.train(train_input_fn,steps=50000, hooks=[logging_hook])
    print("Training finished")

    # Test the model
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_data},y=test_labels,num_epochs=1,shuffle=False)
    test_results = cifar10_classifier.evaluate(input_fn=test_input_fn)
    print(test_results)

if __name__ == "__main__":
    tf.app.run()