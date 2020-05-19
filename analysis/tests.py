# from django.test import TestCase
#
# Create your tests here.

from gensim.models import KeyedVectors
from threading import Semaphore
model = KeyedVectors.load_word2vec_format('/Users/david/PycharmProjects/Tensor/sgns.zhihu.bigram', binary=False)
model.init_sims(replace=True)
model.save('model.bin')
print("success")