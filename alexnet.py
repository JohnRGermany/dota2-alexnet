################################################################################

import os
import numpy as np
import time
import tensorflow as tf
import struct

rows, cols = (160, 160)
train_x = np.zeros((1, rows, cols, 1)).astype(np.float32)
train_y = np.zeros((1, 11))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

# Placeholders
x = tf.placeholder(tf.float32, (None, ) + xdim)
y = tf.placeholder(tf.float32, (None, ydim))
keep_rate = tf.placeholder(tf.float32)

#conv1
#conv(11, 11, 96, 4, 4, padding='SAME', shape=(?, 40, 40, 96))
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1 = tf.layers.conv2d(inputs=x, filters=c_o, kernel_size=[k_h, k_w], strides=[s_h, s_w], padding="SAME", activation=tf.nn.relu)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', shape=(?, 19, 19, 96))
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv2
#conv(5, 5, 256, 1, 1, shape=(?, 19, 19, 256))
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2 = tf.layers.conv2d(inputs=maxpool1, filters=c_o, kernel_size=[k_h, k_w], padding="SAME", activation=tf.nn.relu)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', shape=(?, 9, 9, 256))
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, shape=(?, 9, 9, 384))
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3 = tf.layers.conv2d(inputs=maxpool2, filters=c_o, kernel_size=[k_h, k_w], padding="SAME", activation=tf.nn.relu)

#conv4
#conv(3, 3, 384, 1, 1, shape=(?, 9, 9, 384))
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4 = tf.layers.conv2d(inputs=conv3, filters=c_o, kernel_size=[k_h, k_w], padding="SAME", activation=tf.nn.relu)

#conv5
#conv(3, 3, 256, 1, 1, shape=(?, 9, 9, 256))
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5 = tf.layers.conv2d(inputs=conv4, filters=c_o, kernel_size=[k_h, k_w], padding="SAME", activation=tf.nn.relu)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', shape=(?, 4, 4, 256))
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
reshape = tf.reshape(maxpool5, [-1, 4 * 4 * 256])
fc6 = tf.layers.dense(reshape, 4096, activation=tf.nn.relu)
do6 = tf.layers.dropout(inputs=fc6, rate=keep_rate)

#fc7
#fc(4096, name='fc7')
fc7 = tf.layers.dense(do6, 4096, activation=tf.nn.relu)
do7 = tf.layers.dropout(inputs=fc7, rate=keep_rate)

#fc8
#fc(11, name='fc8')
fc8 = tf.layers.dense(do7, ydim, activation=tf.nn.relu)

tvars = tf.trainable_variables()
l2 = tf.add_n([tf.nn.l2_loss(v) for v in tvars if 'bias' not in v.name ]) * 0.001
loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fc8) + l2

lr = .001
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

correct_prediction = tf.equal(tf.argmax(fc8,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
################################################################################

#Training

data_folder = 'dota2-dataset'
images_files = ['train-data-'+str(i) for i in range(4)]
labels_files = ['train-labels-'+str(i) for i in range(4)]

MAX_EPOCHS = 250
BATCH_SIZE = 64

def batch_generator(images_path, labels_path, batch_size):
    assert os.path.exists(images_path) and os.path.exists(labels_path)

    # Read images and labels from file
    with open(images_path, 'rb') as image_file:
        #magic, num, _, _ = struct.unpack(">iiii", image_file.read(16))
        #assert magic == 1685025889
        images = np.fromfile(image_file, dtype=np.uint8).reshape(-1, rows, cols, 1)

    def one_hot_label(label):
        # in dota2 slot numbers are 0-4 and 128-132
        if label >= 128:
            label -= 123
        one_hot = np.zeros((ydim,))
        one_hot[label] = 1
        return one_hot

    with open(labels_path, 'rb') as labels_file:
        #magic, num = struct.unpack(">ii", labels_file.read(8))
        #assert magic == 1685025890
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
for epoch in range(MAX_EPOCHS):
    for i in range(4):
        images_path = os.path.join(data_folder, images_files[i])
        labels_path = os.path.join(data_folder, labels_files[i])
        t = time.time()
        l = 0
        j = 0
        for images, labels in batch_generator(images_path, labels_path, BATCH_SIZE):
            _, _, l_k = sess.run([fc8, train_op, loss], feed_dict={x: images, y: labels, keep_rate: 0.4})
            j += 1
            l += ((l_k - l)/j)
        print('Finsished epoch {0!s} on file {1!s} in {2!s} with avg loss {3!s}'.format(epoch, i,(time.time()-t), l))

# Test accuracy
correct_predictions = 0
images_path = os.path.join(data_folder, 'test-data')
labels_path = os.path.join(data_folder, 'test-labels')
for images, labels in batch_generator(images_path, labels_path, 1000):
    c = sess.run(correct_prediction, feed_dict={x: images, y: labels, keep_rate: 1.0})
    correct_predictions += np.count_nonzero(c == 1)

print('Accuracy: {0!s}'.format(correct_predictions / 19999))
# Save
save_path = saver.save(sess, './models/model.ckpt')
print('Model saved in path {0!s}'.format(save_path))
