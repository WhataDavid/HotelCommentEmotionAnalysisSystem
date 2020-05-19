import json
from os import path

import keras
from django.shortcuts import render
import numpy as np
import re
import jieba
from gensim.models import KeyedVectors
import warnings
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import os
import tensorflow as tf

warnings.filterwarnings("ignore")
global cn_model
global graph
global resultString
global model
global positiveTag
global severalResult
severalResult = []
model = None
cn_model = None
graph = None
embedding_dim = 300
print("正在加载")
resultString = "123"
global test_list

cn_model = KeyedVectors.load('/Users/david/PycharmProjects/Tensor/django/mysite/analysis/model.bin', mmap='r')
cn_model.syn0norm = cn_model.syn0  # prevent recalc of normed vectors
print("加载完成")
test_list = ['客服小姐姐很漂亮，也很有礼貌啊啊啊']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
embedding_matrix = np.zeros((100000, embedding_dim))
path_checkpoint = '/Users/david/PycharmProjects/Tensor/100000weigths.h5'
# model = Sequential()
global commentScore
global positiveNum
global negativeNum
global sumNum
commentScore = 0
positiveNum = 0
negativeNum = 0
sumNum = 0


def predict_sentiment(text):
    global resultString
    global positiveTag
    global model
    global positiveNum
    global negativeNum
    global commentScore
    global sumNum
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    print("text")
    print(text)

    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            if cn_model.vocab[word].index < 100000:
                cut_list[i] = cn_model.vocab[word].index
            else:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0
    tokens_pad = pad_sequences([cut_list], maxlen=236, padding='pre', truncating='pre')
    try:
        result = model.predict_classes(x=tokens_pad)
    except Exception:
        # keras.backend.clear_session()
        model = Sequential()
        model.add(Embedding(100000,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=236,
                            trainable=False
                            ))
        model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
        model.add(LSTM(units=16, return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.load_weights(path_checkpoint)
        result = model.predict_classes(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        resultString = " ' " + text + " ' " + "是一条积极评论"
        print('积极', 'rate=%.2f' % coef)
        positiveTag = True
        positiveNum += 1
        commentScore += coef
    else:
        resultString = " ' " + text + " ' " + "是一条消极评论"
        print('消极', 'rate=%.2f' % coef)
        positiveTag = False
        negativeNum += 1
        commentScore += coef
    print(resultString)
    severalResult.append(resultString)


def home(request):
    global resultString
    global test_list
    global cn_model
    global model
    global path_checkpoint
    # keras.backend.clear_session()
    # model = Sequential()
    # model.add(Embedding(50000,
    #                     embedding_dim,
    #                     weights=[embedding_matrix],
    #                     input_length=236,
    #                     trainable=False
    #                     ))
    # model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    # model.add(LSTM(units=16, return_sequences=False))
    # model.add(Dense(1, activation='sigmoid'))
    # model.load_weights(path_checkpoint)

    for text in test_list:
        predict_sentiment(text)
    return render(request, 'home.html', {})


def user(request):
    if request.method == 'POST':
        user = request.POST['user']
        import requests
        import json
        user_request = requests.get("https://api.github.com/users/" + user)
        username = json.loads(user_request.content)
        return render(request, 'user.html', {'user': user, 'username': username})
    else:
        notfound = "请在搜索框输入查询的用户"
        return render(request, 'user.html', {'notfound': notfound})


def analysis(request):
    global resultString
    global test_list
    global cn_model
    global model
    global path_checkpoint
    global inputString
    global positiveTag
    # inputString=''
    try:
        if request.method == 'POST':
            inputString = request.POST['inputString']
        if request.method == 'GET':
            inputString = request.GET['inputString']
    except:
        print("出现故障")
    # keras.backend.clear_session()
    # model = Sequential()
    # model.add(Embedding(50000,
    #                     embedding_dim,
    #                     weights=[embedding_matrix],
    #                     input_length=236,
    #                     trainable=False))
    # model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    # model.add(LSTM(units=16, return_sequences=False))
    # model.add(Dense(1, activation='sigmoid'))
    # model.load_weights(path_checkpoint)
    if inputString:
        predict_sentiment(inputString)
    else:
        resultString = '输入为空'
    return render(request, 'analysis.html',
                  {'inputString': inputString, 'resultString': resultString, 'positiveTag': positiveTag})


def file(request):
    global test_list
    selected = False
    selectedCorrect = False
    global severalResult
    global positiveNum
    global negativeNum
    global commentScore
    global sumNum
    commentScore = 0
    positiveNum = 0
    negativeNum = 0
    sumNum = 0
    if request.method == 'POST':
        userfile = request.FILES['userfile']
        destination = open(os.path.join("/Users/david/PycharmProjects/Tensor/django/mysite/analysis/", userfile.name),
                           'wb+')
        for chunk in userfile.chunks():
            destination.write(chunk)
        destination.close()
        if not userfile:
            message = "未选择文件"
            selected = False
            return render(request, 'file.html',
                          {'message': message, 'selected': selected, 'selectedCorrect': selectedCorrect})
        else:
            suffix = userfile.name.split(".")[1]
            print(suffix)
            message = suffix
            selected = True
            selectedCorrect = True
            severalResult = []
            if suffix in ['csv', 'txt']:
                f = open("/Users/david/PycharmProjects/Tensor/django/mysite/analysis/" + userfile.name, "r",
                         encoding='utf-8')
                with f:
                    data = f.read()
                    test_list = data.split('\n')
                    f.close()
                positiveNum = 0
                negativeNum = 0
                commentScore = 0
                sumNum = 0
                for text in test_list:
                    predict_sentiment(text)
                sumNum = positiveNum + negativeNum
                commentScore = commentScore / sumNum * 100
                commentScore = round(commentScore, 1)
                return render(request, 'file.html',
                              {'message': message, 'selected': selected, 'selectedCorrect': selectedCorrect,
                               'severalResult': severalResult, 'positiveNum': positiveNum, 'negativeNum': negativeNum,
                               'sumNum': sumNum, 'commentScore': commentScore})
            elif suffix in ['json']:
                f = open("/Users/david/PycharmProjects/Tensor/django/mysite/analysis/" + userfile.name, "r",
                         encoding='utf-8')
                with f:
                    data = json.loads(f.read())
                    text_lists = np.array(data['comment'])
                    f.close()
                for text in test_list:
                    predict_sentiment(text)
                return render(request, 'file.html',
                              {'message': message, 'selected': selected, 'selectedCorrect': selectedCorrect,
                               'severalResult': severalResult, 'positiveNum': positiveNum, 'negativeNum': negativeNum,
                               'sumNum': sumNum, 'commentScore': commentScore})
            else:
                selected = True
                selectedCorrect = False
                message = "格式不符合"
            return render(request, 'file.html',
                          {'message': message, 'selected': selected, 'selectedCorrect': selectedCorrect})

    return render(request, 'file.html', {})


def crawl(request):
    if request.method == 'GET':
        city = request.GET['city']
        hotelId = int(request.GET['hotelId'])
        pageNum = int(request.GET['pageNum'])

        from urllib import request as req

        global severalResult
        global positiveNum
        global negativeNum
        global commentScore
        global sumNum
        global test_list
        test_list = []
        content = []

        def getResponse(url,hotelId,Pagenum):
            data = {"hotelId": hotelId, "pageIndex": 1, "tagId": 0, "pageSize": Pagenum*10, "groupTypeBitMap": 2,
                    "needStatisticInfo": 0, "order": 0, "basicRoomName": "", "travelType": -1,
                    "head": {"cid": "09031144211504567945", "ctok": "", "cver": "1.0", "lang": "01", "sid": "8888",
                             "syscode": "09", "auth": "", "extension": []}}
            data = json.dumps(data).encode(encoding='utf-8')

            header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
                           "Content-Type": "application/json"}

            url_request = req.Request(url=url, data=data, headers=header_dict)
            print("这个对象的方法是：", url_request.get_method())

            url_response = req.urlopen(url_request)
            return url_response

        http_response = getResponse(
            "http://m.ctrip.com/restapi/soa2/16765/gethotelcomment?_fxpcqlniredt=09031144211504567945", hotelId,pageNum)

        data = http_response.read().decode('utf-8')
        data = json.loads(data)
        # print(data)
        othersCommentList = data['othersCommentList']
        tdk = data['tdk']
        hotelName = tdk['title']
        hotelName = hotelName.split('点')[0]
        commentScore = 0
        positiveNum = 0
        negativeNum = 0
        sumNum = 0
        test_list = []
        severalResult=[]

        for each in othersCommentList:
            contentCurrent = each['content']
            content.append(contentCurrent)
        test_list=content
        print(test_list)
        for text in test_list:
            predict_sentiment(text)
        sumNum = positiveNum + negativeNum
        commentScore = commentScore / sumNum * 100
        commentScore = round(commentScore, 1)

        return render(request, 'crawl.html',
                      {'city': city, 'hotelId': hotelId, 'pageNum': pageNum, 'hotelName': hotelName, 'sumNum': sumNum,
                       'positiveNum': positiveNum, 'negativeNum': negativeNum, 'severalResult': severalResult,'commentScore': commentScore})
