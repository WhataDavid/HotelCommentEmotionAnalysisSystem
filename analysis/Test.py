import numpy as np
import re
import jieba
from gensim.models import KeyedVectors
import warnings
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GRU,LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
warnings.filterwarnings("ignore")
cn_model = KeyedVectors.load_word2vec_format('./sgns.zhihu.bigram', binary=False)
embedding_dim = 300
resultString = ""
test_list = [
    '客服小姐姐很漂亮，也很有礼貌'
]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
embedding_matrix = np.zeros((50000, embedding_dim))
model = Sequential()
model.add(Embedding(50000,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=236,
                    trainable=False))
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
path_checkpoint = './LSTMweigths.h5'
model.load_weights(path_checkpoint)
def predict_sentiment(text):
    print(text)
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    tokens_pad = pad_sequences([cut_list], maxlen=236, padding='pre', truncating='pre')
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('Positive', 'rate=%.2f' % coef)
        resultString = text + "是一条积极评论"
    else:
        print('Negative', 'rate=%.2f' % coef)
        resultString = text + "是一条消极评论"
    print(resultString)

for text in test_list:
    predict_sentiment(text)
