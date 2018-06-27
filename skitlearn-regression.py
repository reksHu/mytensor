from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import nltk
def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

path =r"E:\python\tools\data\temp_spam_data.csv"

df = pd.read_csv(path,delimiter=',',header=None)
print(type(df[1]))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0],train_size=0.8,test_size=0.2)
vectorizer=TfidfVectorizer(tokenizer=tokenizer,stop_words='english', max_features=1000)

print(type(X_test_raw))
spam_email = ["Yeah do! Dont stand to close tho- youll catch something!"]

X_train=vectorizer.fit_transform(X_train_raw)
X_test=vectorizer.transform(spam_email)

classifier=LogisticRegression()
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)
for i ,prediction in enumerate(predictions):
    print('预测类型：%s.信息：%s' %(prediction,X_train_raw.iloc[i]))