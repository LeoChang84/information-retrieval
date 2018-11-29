# -*- coding: utf-8 -*-
# @Author  : leo.chang
# @File    : train.py
# @Environment : Python 3.6+
# @Reference1 : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM

# import packages
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K
from util import make_w2v_embeddings, split_and_zero_padding, ManDist

# -----------------pre loading------------------ #

task_type = input('type a, b or c for choosing task:')
if task_type == 'a':
    TRAIN_CSV = './data/QC/train-dev.tsv'
elif task_type == 'b':
    TRAIN_CSV = './data/QQ/train-dev.tsv'
elif task_type == 'c':
    TRAIN_CSV = './data/QRC/train-dev.tsv'

embedding_path = 'GoogleNews-vectors-negative300.bin'
embedding_dim = 300
max_seq_length = 10
savepath = './data/en_SiameseLSTM_' + task_type.upper() + '.h5'
flag = 'en'
# loading embedding
print('Loading word2vec model(it may takes 2-3 mins) ...')
embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

# load pretrained model
train_df = pd.read_csv(TRAIN_CSV, sep = "\t")
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]
#  set word to vector
train_df, embeddings = make_w2v_embeddings(flag, embedding_dict, train_df, embedding_dim=embedding_dim)

# spilt training set
X = train_df[['question1_n', 'question2_n']]
Y = train_df['label']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)
X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# transform label to vector
Y_train = Y_train.values
Y_validation = Y_validation.values

# assert preprocess done
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# -----------------basic function------------------ #

def shared_model(_input):
    # word embedding
    embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
                         trainable=False)(_input)

    # multi-layers Bi-LSTM
    activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(embedded)
    activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(activations)

    # dropout
    # activations = Dropout(0.5)(activations)

    # Attention
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_hidden * 2)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([activations, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    # dropout
    # sent_representation = Dropout(0.1)(sent_representation)

    return sent_representation


def shared_model_cnn(_input):
    # 词向量化
    embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
                         trainable=False)(_input)

    # CNN
    activations = Conv1D(250, kernel_size=5, activation='relu')(embedded)
    activations = GlobalMaxPool1D()(activations)
    activations = Dense(250, activation='relu')(activations)
    activations = Dropout(0.3)(activations)
    activations = Dense(1, activation='sigmoid')(activations)

    return activations


# -----------------main----------------- #

if __name__ == '__main__':

    # parameter
    batch_size = 1024
    n_epoch = 9
    n_hidden = 50

    left_input = Input(shape=(max_seq_length,), dtype='float32')
    right_input = Input(shape=(max_seq_length,), dtype='float32')
    left_sen_representation = shared_model(left_input)
    right_sen_representation = shared_model(right_input)

    # 引入曼哈顿距离，把得到的变换concat上原始的向量再通过一个多层的DNN做了下非线性变换、sigmoid得相似度
    # 没有使用https://zhuanlan.zhihu.com/p/31638132中提到的马氏距离，尝试了曼哈顿距离、点乘和cos，效果曼哈顿最好
    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))
    print(malstm_trained)
    # Plot accuracy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('./data/history-graph.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          '(max: ' + str(max(malstm_trained.history['val_acc']))[:6] + ')')
    print('Done.')
