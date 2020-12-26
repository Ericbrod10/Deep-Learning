import sys
import re
from pandas import read_csv
from nltk.corpus import stopwords
import nltk
from gensim.parsing.preprocessing import remove_stopwords
from nltk import RegexpTokenizer
from gensim.models import Word2Vec

NewsFile = read_csv(sys.argv[1])
#NewsFile = read_csv('fake_or_real_news.csv')
model = sys.argv[2]
word_re = r"\w+"


#NewsFile['title'] = NewsFile['title']
#NewsFile['title'].apply(lambda x: [item for item in str(x).split(' ') if item not in stop])



#making real news model
label_news = NewsFile.loc[NewsFile['label'] == 'REAL'].reset_index()
label_news = label_news.drop(['label', 'Unnamed: 0'], axis=1)
label_news_text = label_news['title'].map(str) + "\n " + label_news['text'].map(str)
label_news_text = label_news_text.str.lower()

#for i in range(len(label_news_text)):
#    label_news_text[i] = remove_stopwords(label_news_text[i].strip())

#label_news_text = label_news_text.apply(lambda x: [item for item in str(x).split(' ') if item not in stop])
label_news_text = label_news_text.apply(RegexpTokenizer(word_re).tokenize)

#myfile = open('RealNews5', 'a',encoding="utf-8")
#print(label_news_text ,file=myfile)
realModel1 = Word2Vec(label_news_text, size = 300, window = 2, min_count = 1, workers = 4)
realModel2 = Word2Vec(label_news_text, size = 300, window = 5, min_count = 1, workers = 4)
realModel3 = Word2Vec(label_news_text, size = 1000, window = 2, min_count = 1, workers = 4)
realModel4 = Word2Vec(label_news_text, size = 1000, window = 5, min_count = 1, workers = 4)


#making fake model
label_news = NewsFile.loc[NewsFile['label'] == 'FAKE'].reset_index()
label_news = label_news.drop(['label', 'Unnamed: 0'], axis=1)
#label_news['title'] = label_news['title'].apply(lambda x: [item for item in str(x).lower().split(' ') if item not in stop])
#label_news['text'] = label_news['text'].apply(lambda x: [item for item in str(x).lower().split(' ') if item not in stop])
label_news_text = label_news['title'].map(str) + "\n" + label_news['text'].map(str)
label_news_text = label_news_text.str.lower()

#for i in range(len(label_news_text)):
#    label_news_text[i] = remove_stopwords(label_news_text[i].strip())

#label_news_text = label_news_text.apply(lambda x: [item for item in str(x).lower().split(' ') if item not in stop])
#label_news_text = remove(label_news_text)
label_news_text= label_news_text.apply(RegexpTokenizer(word_re).tokenize)

#myfile = open('fakenews2', 'a',encoding="utf-8")
#print(label_news_text.all() ,file=myfile)
fakeModel1 = Word2Vec(label_news_text, size = 300, window = 2, min_count = 1, workers = 4)
fakeModel2 = Word2Vec(label_news_text, size = 300, window = 5, min_count = 1, workers = 4)
fakeModel3 = Word2Vec(label_news_text, size = 1000, window = 2, min_count = 1, workers = 4)
fakeModel4 = Word2Vec(label_news_text, size = 1000, window = 5, min_count = 1, workers = 4)


realModel1.save(model+"1_real")
realModel2.save(model+"2_real")
realModel3.save(model+"3_real")
realModel4.save(model+"4_real")

fakeModel1.save(model +"1_fake")
fakeModel2.save(model +"2_fake")
fakeModel3.save(model +"3_fake")
fakeModel4.save(model +"4_fake")