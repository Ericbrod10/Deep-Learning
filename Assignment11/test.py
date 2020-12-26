import sys
from nltk import RegexpTokenizer
from gensim.models import Word2Vec

data = sys.argv[1]
model = sys.argv[2]

QueryFile = open(sys.argv[3], 'r')
RealModel1 = Word2Vec.load(model + "1_real")
RealModel2 = Word2Vec.load(model + "2_real")
RealModel3 = Word2Vec.load(model + "3_real")
RealModel4 = Word2Vec.load(model + "4_real")

FakeModel1 = Word2Vec.load(model + "1_fake")
FakeModel2 = Word2Vec.load(model + "2_fake")
FakeModel3 = Word2Vec.load(model + "3_fake")
FakeModel4 = Word2Vec.load(model + "4_fake")

for line in QueryFile.readlines():
    word = line.split("\n")[0].lower()
    print('-------------------------------%s-------------------------------' % word)
    print('--------------%s--------------' % 'REAL')
    if word in list(RealModel1.wv.vocab):
        print('Node Size: 300 & Window Size 2: Real News Similar words to %s: %s \n' % (word, RealModel1.wv.most_similar(word, topn=5)))

    if word in list(RealModel2.wv.vocab):
        print('Node Size: 300 & Window Size 5: Real News Similar words to %s: %s \n' % (word, RealModel2.wv.most_similar(word, topn=5)))

    if word in list(RealModel3.wv.vocab):
        print('Node Size: 1000 & Window Size 2: Real News Similar words to %s: %s \n' % (word, RealModel3.wv.most_similar(word, topn=5)))

    if word in list(RealModel4.wv.vocab):
        print('Node Size: 1000 & Window Size 5: Real News Similar words to %s: %s \n' % (word, RealModel4.wv.most_similar(word, topn=5)))
    
    print('--------------%s--------------' % 'FAKE')

    if word in list(FakeModel1.wv.vocab):
        print('Node Size: 300 & Window Size 2: Fake News Similar words to %s: %s \n' % (word, FakeModel1.wv.most_similar(word, topn=5)))

    if word in list(FakeModel2.wv.vocab):
        print('Node Size: 300 & Window Size 5: Fake News Similar words to %s: %s \n' % (word, FakeModel2.wv.most_similar(word, topn=5)))

    if word in list(FakeModel3.wv.vocab):
        print('Node Size: 1000 & Window Size 2: Fake News Similar words to %s: %s \n' % (word, FakeModel3.wv.most_similar(word, topn=5)))

    if word in list(FakeModel4.wv.vocab):
        print('Node Size: 1000 & Window Size 5: Fake News Similar words to %s: %s \n' % (word, FakeModel4.wv.most_similar(word, topn=5)))
        print('\n')
