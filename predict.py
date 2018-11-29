# -*- coding: utf-8 -*-
# @Author  : leo.chang
# @File    : predict.py
# @Environment : Python 3.6+
# @Reference : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM

# import packages
import sys
import pandas as pd
import keras
from gensim.models import KeyedVectors
from util import make_w2v_embeddings, split_and_zero_padding, ManDist
from operator import itemgetter

task_type = sys.argv[1]
if task_type == 'a':
    TEST_CSV = './data/QC/test.tsv'
elif task_type == 'b':
    TEST_CSV = './data/QQ/test.tsv'
elif task_type == 'c':
    TEST_CSV = './data/QRC/test.tsv'
else:
    print('error task type')

embedding_path = 'GoogleNews-vectors-negative300.bin'
embedding_dim = 300
max_seq_length = 10
savepath = './data/en_SiameseLSTM_' + task_type.upper() + '.h5'
flag = 'en'

# initilize with embedding
print("Loading word2vec model(it may takse 2-3 mins)...")
embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
# read testing set
test_df = pd.read_csv(TEST_CSV, sep ='\t')
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

test_question1_id_list = test_df['question1_id'].values.tolist()
test_question2_id_list = test_df['question2_id'].values.tolist()
# make testing set to word2vec
test_df, embeddings = make_w2v_embeddings(flag, embedding_dict, test_df, embedding_dim=embedding_dim)

# preprocess
X_test = split_and_zero_padding(test_df, max_seq_length)
Y_test = test_df['label'].values
# assert all preprocess done
assert X_test['left'].shape == X_test['right'].shape
assert len(X_test['left']) == len(Y_test)

# loading pretrain nodel
model = keras.models.load_model(savepath, custom_objects={"ManDist": ManDist})
model.summary()


# -----------------main----------------- #

if __name__ == '__main__':

    # predict accurarcy
    prediction = model.predict([X_test['left'], X_test['right']])
    # print(prediction)
    prediction_list = prediction.tolist()
    accuracy = 0
    for i in range(len(prediction_list)):
        if i % 100== 0:
            top100_list = []
        if prediction_list[i][0] < 0.5:
            predict_pro = 0
        else:
            predict_pro = 1
        if predict_pro == Y_test[i]:
            accuracy += 1
        tmp = [test_question1_id_list[i], test_question2_id_list[i], prediction_list[i][0], "false"]
        top100_list.append(tmp)
        if i % 100 == 99:
            # top10_list.sort(key=lambda x: x[2], reverse=True)
            rank = 1
            for top in top100_list:
                print(str(top[0]) + "\t" + str(top[1]) + "\t" + str(rank) + "\t" + str(top[2]) + "\t" + 'false' )
                rank += 1
    # print(accuracy / len(Y_test))
