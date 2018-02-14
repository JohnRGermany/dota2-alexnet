################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import os
import numpy as np
from PIL import Image
import time
import tensorflow as tf

rows, cols = (160, 160)
train_x = np.zeros((1, rows, cols, 1)).astype(np.float32)
train_y = np.zeros((1, 11))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    conv = convolve(input, kernel)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=name)

x = tf.placeholder(tf.float32, (None,) + xdim)
y = tf.placeholder(tf.float32, (None,ydim))

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = weight_variable((11, 11, 1, 96), 0.01, name='conv1W')
conv1b = bias_variable((96,), name='conv1b')
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)
#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = weight_variable((k_h, k_w, 96, 256), 0.01, name='conv2W')
conv2b = bias_variable((256,), name='conv2b')
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)

#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = weight_variable((k_h, k_w, 256, c_o), 0.01, name='conv3W')
conv3b = bias_variable((c_o,), name='conv3b')
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = weight_variable((k_h, k_w, 384, c_o), 0.01, name='conv4W')
conv4b = bias_variable((c_o,), name='conv4b')
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = weight_variable((k_h, k_w, 384, c_o), 0.01, name='conv5W')
conv5b = bias_variable((c_o,), name='conv5b')
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = weight_variable((256 * 4 * 4, 4096), 0.01, name='fc6W')
fc6b = bias_variable((4096,), name='fc6b')
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = weight_variable((4096, 4096), 0.01, name='fc7W')
fc7b = bias_variable((4096,), name='fc7b')
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(10, relu=False, name='fc8')
fc8W = weight_variable((4096, ydim), 0.01, name='fc8W')
fc8b = bias_variable((ydim,), name='fc8b')
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fc8)
# loss = tf.losses.sigmoid_cross_entropy_with_logits(labels=y, logits=fc8)

lr = .001
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

prob = tf.nn.softmax(fc8)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(fc8,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
################################################################################

#Training

data_folder = 'dota2-dataset'
images_files = ['train-data-'+str(i) for i in range(4)]
labels_files = ['train-labels-'+str(i) for i in range(4)]

MAX_EPOCHS = 10
BATCH_SIZE = 20

def batch_generator(images_path, labels_path, batch_size):
    assert os.path.exists(images_path) and os.path.exists(labels_path)

    # Read images and labels from file
    with open(images_path, 'rb') as image_file:
        images = np.fromfile(image_file, dtype=np.uint8).reshape(-1, rows, cols, 1)
        print(images.shape)



    def one_hot_label(label):
        # in dota2 slot numbers are 0-4 and 128-132
        if label >= 128:
            label -= 123
        one_hot = np.zeros((ydim,))
        one_hot[label] = 1
        return one_hot

    with open(labels_path, 'rb') as labels_file:
        labels = np.fromfile(labels_file, dtype=np.uint8)
        one_hot_labels = map(one_hot_label, labels)
        labels = np.array(list(one_hot_labels))


    assert len(images) == len(labels)

    # Shuffle in unison
    i = np.arange(len(labels))
    np.random.shuffle(i)
    images, labels = images[i], labels[i]

    batch_size = len(labels) if batch_size == 0 else batch_size
    image_batch = lambda i: (images[i:i+batch_size], labels[i:i+batch_size])

    for i in range(0, len(labels), batch_size):
        yield image_batch(i)

# Train with files
for i in range(4):
    images_path = os.path.join(data_folder, images_files[i])
    labels_path = os.path.join(data_folder, labels_files[i])
    for epoch in range(MAX_EPOCHS):
        t = time.time()
        l = 0
        j = 0
        for images, labels in batch_generator(images_path, labels_path, BATCH_SIZE):
            _, _, l_k = sess.run([fc8, train_op, loss], feed_dict={x: images, y: labels})
            j += 1
            l += ((l_k - l)/j)
        print('Finsished epoch {0!s} on file {1!s} in {2!s} with avg loss {3!s}'.format(epoch, i,(t-time.time()), l))

# Test accuracy
images_path = os.path.join(data_folder, 'test-data')
labels_path = os.path.join(data_folder, 'test-labels')
for images, labels in batch_generator(images_path, labels_path, 0):
    a = sess.run(accuracy, feed_dict={x: images, y: labels})
    print(a)

# Save
save_path = saver.save(sess, './models/model.ckpt')
print('Model saved in path {0!s}'.format(save_path))
