from __future__ import division
from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
from gcn.utils import *
from gcn.models import GCN, MLP
import scipy.sparse as sp
from scipy.io import loadmat
import os
from scipy.io import savemat
import networkx as nx

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 4000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 24, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


dir_path = '/home/fly/github/reid_gcn/gcn/data/data_reid'
a = loadmat(os.path.join(dir_path, 'chen_feature_adj.mat'))
adj = a['adj']
print('adj.shape = ')
print(adj.shape)
# print (adj[0][0:100])

f = loadmat(os.path.join(dir_path, 'chen_label.mat'))
labels = f['label'][0]
print('labels.shape = ')
print(labels.shape)
print('labels = %s' % labels)

m = loadmat('/home/fly/github/reid_gan/pytorch_train_result')
print(m.keys())
gen_names = m['gen_name']
real_gcn_features = m['train_f']
real_num = len(real_gcn_features)

labels_unique = np.unique(labels[:real_num])
print('unique labels = %s' % labels_unique)
print('len(labels_unique) = %d' % len(labels_unique))

labels_count = np.zeros((len(labels_unique), 2), dtype=np.int32)
print('labels_count.shape = ')
print(labels_count.shape)


labels_dict = {}
i = 0
j = 0
while i < real_num:
    # print('i = %d' % i)
    while i < real_num - 1 and labels[i] == labels[i + 1]:
        labels_count[j][0] += 1
        i += 1
    if i < real_num - 1:
        labels_count[j][0] += 1
        labels_dict[labels[i]] = j
        j += 1
        i += 1
    else:
        labels_count[j][0] += 1
        labels_dict[labels[i]] = j
        i += 1

print('labels_dict = %s' % labels_dict)

labels_count[1:, 1] = np.cumsum(labels_count[:, 0])[0:-1]

print('labels_count = %s' % labels_count[:, 0])
print('sum labels_count = %s' % labels_count[:, 1])


m = loadmat(os.path.join(dir_path, 'chen_features.mat'))
print('type(m) = %s' % type(m))
print(m.keys())
features = m['features']
print('len(gcn_features) = %s' % len(features))
print('feature size = %d' % features[0].size)
print('features.shape[0] = %d' % features.shape[0])
print('features.shape[1] = %d' % features.shape[1])

'''the new style of dividing data'''

train_mask = np.zeros((features.shape[0],), dtype=np.bool)
val_mask = np.zeros((features.shape[0],), dtype=np.bool)
test_mask = np.zeros((features.shape[0],), dtype=np.bool)
# print('train_mask = %s' % train_mask)
y_train = np.zeros((features.shape[0], len(labels_unique)))
y_val = np.zeros((features.shape[0], len(labels_unique)))
y_test = np.zeros((features.shape[0], len(labels_unique)))

val_ratio = 0.1
train_ratio = 0.7
test_ratio = 0.2

for i in range(len(labels_unique)):
    for j in range(labels_count[i][0]):
        if j < labels_count[i][0] * val_ratio:
            y_val[labels_count[i][1] + j][labels_dict[labels[labels_count[i][1] + j]]] = 1
            val_mask[labels_count[i][1] + j] = True
        elif j < labels_count[i][0] * (val_ratio + train_ratio):
            y_train[labels_count[i][1] + j][labels_dict[labels[labels_count[i][1] + j]]] = 1
            train_mask[labels_count[i][1] + j] = True
        else:
            y_test[labels_count[i][1] + j][labels_dict[labels[labels_count[i][1] + j]]] = 1
            test_mask[labels_count[i][1] + j] = True


features = sp.csr_matrix(features)
# print('features = %s' % features)


# Some preprocessing
features = preprocess_features(features)

# print('features[2][1] = %s' % features[2][1])
# print('features[2] = %s' % len(features[2]))

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break
    if epoch % 500 == 0:
        # Testing
        test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        a = tf.nn.softmax(model.outputs)
        b = sess.run(a, feed_dict=feed_dict)
        soft_labels_dict = {}
        for i in range(len(gen_names)):
            soft_labels_dict[gen_names[i]] = b[real_num+i]
        savemat(os.path.join(dir_path, 'soft_label_' + str(epoch) + '.mat'), soft_labels_dict)


a = tf.nn.softmax(model.outputs)
b = sess.run(a, feed_dict=feed_dict)

soft_labels_dict = {}
for i in range(len(gen_names)):
    soft_labels_dict[gen_names[i]] = b[real_num+i]
savemat(os.path.join(dir_path, 'soft_label_dict_link24.mat'), soft_labels_dict)
# d = loadmat(os.path.join(dir_path, 'soft_label_dict.mat'))
print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
