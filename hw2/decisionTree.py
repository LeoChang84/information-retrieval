
# coding: utf-8

# In[38]:


import numpy as np
import os

GLOVE_DIR = './data/'

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# In[323]:


def getDicKey(dic):
    for key in dic:
        getKey = key
        return getKey
def getNodeAttribute(idx, nodes):
    for j in range(len(train_content['nodes'])):
        if train_content['nodes'][j][0] == idx:
            return getDicKey(train_content['nodes'][j][1])
def flatten(flat_list):
    flatten = []
    for i in flat_list:
        for j in i:
            flatten.append(j)
    return flatten
def getEmbeddingVector(w1, w2, w3, w4):
    n1 = n2 = n3 = n4 = np.zeros(shape=(1,300))
    if w1 in embeddings_index:
        n1 = embeddings_index[w1]
    if w2 in embeddings_index:
        n2 = embeddings_index[w2]
    if w3 in embeddings_index:
        n3 = embeddings_index[w3]
    if w4 in embeddings_index:
        n4 = embeddings_index[w4]
    n1 = n1.reshape(1, 300).tolist()
    n2 = n2.reshape(1, 300).tolist()
    n3 = n3.reshape(1, 300).tolist()
    n4 = n4.reshape(1, 300).tolist()
#     print(type(n1), len(n1), type(n2), len(n2), type(n3), len(n3), type(n4), len(n4))
    n = n1 + n2 + n3 + n4
    return flatten(n)
def setValList(val_list, key):
    if key == 'fact': # fact
        val_list.append(0)
    elif key == 'analogy': # analogy 
        val_list.append(1)
    else:
        val_list.append(2) # equivalence
    return val_list


# In[326]:


def load_data(DATA_DIR):
    import json
    count = 0
    data_list = []
    val_list = []
    with open(DATA_DIR) as f:
        for line in f:
            train_content = json.loads(line)
            for i in range(len(train_content['edges'])):
                idx_1 = train_content['edges'][i][0]
                idx_2 = train_content['edges'][i][1]
                key = getDicKey(train_content['edges'][i][2])
                val_list = setValList(val_list, key)
                idx_1_token = train_content['tokens'][idx_1[0]].lower()
                idx_2_token = train_content['tokens'][idx_2[0]].lower()
                idx_1_attri = getNodeAttribute(idx_1, train_content['nodes'])
                idx_2_attri = getNodeAttribute(idx_2, train_content['nodes'])
    #             print(idx_1_token, idx_2_token, idx_1_attri, idx_2_attri)
                tmp = getEmbeddingVector(idx_1_token, idx_1_attri, idx_2_token, idx_2_attri)
                data_list.append(tmp)
#     print(np.vstack(data_list).size, np.hstack(val_list).size)
    return np.vstack(data_list), np.hstack(val_list)


# In[328]:


from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics

TRAIN_DIR = './data/train.json'
TEST_DIR = './data/test.json'

# load data
train_X, train_Y = load_data(TRAIN_DIR)
test_X, test_Y = load_data(TEST_DIR)
# print(edges_X)
# print('-----')
# print(edges_Y)
#  = load_data(TEST_DIR)

# split training set and validation set
# train_X, test_X, train_y, test_y = train_test_split(edges_X, edges_Y, test_size = 0.3)

# build classifier
clf = tree.DecisionTreeClassifier()
edges_clf = clf.fit(train_X, train_Y)
# predict
test_y_predicted = edges_clf.predict(test_X)
# print(test_y_predicted)
# ans
# print(test_Y)

accuracy = metrics.accuracy_score(test_Y, test_y_predicted)
print(accuracy)

