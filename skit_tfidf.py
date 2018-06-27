from sklearn.feature_extraction.text import TfidfVectorizer
import  string
import nltk
import  numpy as np
from sklearn.linear_model import LogisticRegression

spam =['I am waiting machan. Call me once you free.'
    ,'I like you peoples very much:) but am very shy pa'
    ,'Your free ringtone is waiting to be collected. Simply text the password "MIX" to 85069 to verify. Get Usher and Britney. FML, PO Box 5249, MK17 92H. 450Ppw 16']

lable= [1,1,0]

texts = [x.lower() for x in spam]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
sparse_tfidf_texts = tfidf.fit_transform(texts)
wight = sparse_tfidf_texts.toarray()
print(wight)
log_reg= LogisticRegression(class_weight="balanced")
log_reg.fit(sparse_tfidf_texts,np.asarray(lable))

check_test =['the password "XYZ" to 85069 to verification']
check = [x.lower() for x in check_test]
check = [''.join(c for c in x if c not in string.punctuation) for x in check]
check = [''.join(c for c in x if c not in '0123456789') for x in check]
check = [' '.join(x.split()) for x in check]
result = log_reg.predict_proba([check,1])
print(result)
# data = TfidfVectorizer.fit_transform()


