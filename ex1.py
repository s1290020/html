import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression
import nltk

import os
for dirname, _, filenames in os.walk('/Users/y.taisei'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data=pd.read_csv("/Users/y.taisei/Desktop/pythonanalysis/blogtext.csv")
data.head(10)
print()

data.isna().any()
print()

data.shape
print()

data=data.head(10000)
data.info()
print()

data.drop(['id','date'], axis=1, inplace=True)
data.head()
print()

data['age']=data['age'].astype('object')
data.info()
print()

#Data Wrangling for data['text'] column to remove all unwanted text from the column
data['clean_data']=data['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ',x))
data['clean_data']=data['clean_data'].apply(lambda x: x.lower())
data['clean_data']=data['clean_data'].apply(lambda x: x.strip())
print("Actual data=======> {}".format(data['text'][1]))
print()

print("Cleaned data=======> {}".format(data['clean_data'][1]))
print()

#I had to do import nltk to use nltk

#Remove all stop words
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))

data['clean_data']=data['clean_data'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords]))

data['clean_data'][6]
print()

#Merging all the other columns into labels columns
data['labels']=data.apply(lambda col: [col['gender'],str(col['age']),col['topic'],col['sign']], axis=1)
data.head()
print()

data=data[['clean_data','labels']]
data.head()
print()

#Splitting the data into X and Y
X=data['clean_data']
Y=data['labels']

#Lets perform count vectorizer with bi-grams and tri-grams to get the count vectors of the X data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(binary=True, ngram_range=(1,2))
X=vectorizer.fit_transform(X)
X[1]
print()

#Let us see some feature namesLet us see some feature names
vectorizer.get_feature_names()[:5]
print()

label_counts=dict()

for labels in data.labels.values:
    for label in labels:
        if label in label_counts:
            label_counts[label]+=1
        else:
            label_counts[label]=1

label_counts
print()

#Pre-processing the labels
from sklearn.preprocessing import MultiLabelBinarizer
binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
Y=binarizer.fit_transform(data.labels)

#Splitting the data into 80% Train set :20% Test set
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='lbfgs')
model=OneVsRestClassifier(model)
model.fit(Xtrain,Ytrain)
print()

Ypred=model.predict(Xtest)
Ypred_inversed = binarizer.inverse_transform(Ypred)
y_test_inversed = binarizer.inverse_transform(Ytest)
for i in range(5):
    print('Text:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        Xtest[i],
        ','.join(y_test_inversed[i]),
        ','.join(Ypred_inversed[i])
    ))
    
print()

#Pre-processing the labels
from sklearn.preprocessing import MultiLabelBinarizer
binarizer=MultiLabelBinarizer(classes=sorted(label_counts.keys()))
Y=binarizer.fit_transform(data.labels)

#Splitting the data into 80% Train set :20% Test set
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='lbfgs')
model=OneVsRestClassifier(model)
model.fit(Xtrain,Ytrain)
print()

Ypred=model.predict(Xtest)
Ypred_inversed = binarizer.inverse_transform(Ypred)
y_test_inversed = binarizer.inverse_transform(Ytest)
for i in range(5):
    print('Text:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        Xtest[i],
        ','.join(y_test_inversed[i]),
        ','.join(Ypred_inversed[i])
    ))

print()

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(Ytest, Ypred):
    print('Accuracy score: ', accuracy_score(Ytest, Ypred))
    print('F1 score: ', f1_score(Ytest, Ypred, average='micro'))
    print('Average precision score: ', average_precision_score(Ytest, Ypred, average='micro'))
    print('Average recall score: ', recall_score(Ytest, Ypred, average='micro'))

print_evaluation_scores(Ytest, Ypred)
print()









