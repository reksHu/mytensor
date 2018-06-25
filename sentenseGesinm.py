from scipy import spatial
import numpy as np
from gensim import corpora,similarities,models
from gensim import utils
def get_content():
    file_name =r"E:\python\tools\text8"
    sentence, rest = [], b''
    max_sentence_length = 10000
    with utils.smart_open(file_name) as fin:
        while True:
            text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
            if text == rest:  # EOF
                words = utils.to_unicode(text).split()
                sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                if sentence:
                    yield sentence
                break
            last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration
            words, rest = (utils.to_unicode(text[:last_token]).split(),
                           text[last_token:].strip()) if last_token >= 0 else ([], text)
            sentence.extend(words)
            while len(sentence) >= max_sentence_length:
                yield sentence[:max_sentence_length]
                sentence = sentence[max_sentence_length:]

sentences = get_content()
print(sentences[0:50])
dictionary = corpora.Dictionary(sentences)
dictionary.save(r'E:\python\tools\mydict.dic')


def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
