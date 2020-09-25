import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Practises\\sent")

message = pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t')

message.isnull().sum()

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

lem = WordNetLemmatizer()
Corpus = []

for i in range(len(message['Review'])):
    review = re.sub('[^a-zA-Z]',' ', message['Review'][i])
    review = review.lower()
    review = review.split()
    review = [lem.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    comment = ' '.join(review)
    Corpus.append(comment)
    
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


TF = TfidfVectorizer(max_features = 1500)
X = TF.fit_transform(Corpus).toarray()
Y = message['Liked']

X = pd.DataFrame(X)

FullRaw = pd.concat([X,Y],axis =1)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['Liked'],axis =1)
Train_Y = Train['Liked']
Test_X = Test.drop(['Liked'],axis =1)
Test_Y = Test['Liked']

M1 = MultinomialNB().fit(Train_X,Train_Y)

Test_Pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix

Con_Mat = confusion_matrix(Test_Pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

import pickle

pickle.dump(M1,open('model.pkl','wb'))
pickle.dump(TF,open('tf.pkl','wb'))

