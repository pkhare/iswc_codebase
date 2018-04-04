#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:57:34 2017

@author: pk4634
"""


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import csv
import json
import codecs
import numpy as np
from nltk.tokenize import regexp_tokenize
import string

warnings.filterwarnings("ignore", category=DeprecationWarning)

dictn = dict()
dictn2 = dict()

stemmer = PorterStemmer()

stringFirst = ''
stringSecond = ''


def tokenize_and_stem(text):
    # tokens = word_tokenize(text)
    # tokens = regexp_tokenize(text, pattern=r"\s|[\.,:;'()?!]", gaps=True)
    tokens = regexp_tokenize(text, pattern=r"\s|[\.,:;'()?!]", gaps=True)
    # strip out punctuation and make lowercase
    tokens = [token.lower().strip(string.punctuation)
              for token in tokens if token.isalnum()]

    # now stem the tokens
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

# sem_negative_savar_tweets
# with open('/Users/pk4634/Documents/new data/recurssive_exp/savar_test/sem_positive_train_tweets.txt') as fline, \
# open('/Users/pk4634/Documents/new data/recurssive_exp/savar_test/sem_negative_savar_tweets.txt') as fline2:


with open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/language_iswc/balanced/stat_n_hypernym_dbpedia_en/all_en_train_balanced_random_stat_hypernym_dbpedia_en_semantics.csv') as fline, \
        open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/language_iswc/balanced/stat_n_hypernym_dbpedia_en/all_es_test_balanced_random_stat_hypernym_dbpedia_en_semantics.csv') as fline2:

    #stop = set(stopwords.words('english'))

    data_json = json.load(codecs.open(
        '/Users/pk4634/Documents/phase2/extensionphase1/selected_lang_stopwords.json', 'r', 'utf-8'))

    s = set(data_json['en'])
    s.update(data_json['es'])
    s.update(data_json['it'])
    s.update(data_json['fr'])
    s.update(data_json['pt'])
    s.update(data_json['hi'])
    s.update(data_json['nl'])
    s.update(data_json['ru'])
    s.update(data_json['de'])

    stop = frozenset(s)

    for line in csv.reader(fline, delimiter='\t', skipinitialspace='True', quotechar=None):

        # if line[4] == '1' or line[4] == '0':
        if line[8] == '1' or line[8] == '0':
            sentence = line[1]

        # for sentence in fline:

            sentence = sentence.decode('utf-8')
            sentence = sentence.lower()

            # sentence = " ".join(filter(lambda x:x[0]!='#', sentence.split()))
            sentence = " ".join(filter(lambda x: x[0] != '@', sentence.split()))
            sentence = sentence.replace('rt', '')
            # sentence = sentence.replace('#','')

            sentence = sentence.replace('!', '')
            sentence = sentence.replace('*', '')

            lst = [i for i in sentence.lower().split() if i not in stop]

            for x in lst:
                stringFirst += x
                # if dictn.has_key(stemmer.stem(x.strip())):
                if dictn.has_key(x.strip()):
                    #dictn[stemmer.stem(x.strip())] += 1
                    dictn[x.strip()] += 1

                else:
                    #dictn[stemmer.stem(x.strip())] = 1
                    dictn[x.strip()] = 1

    for line2 in csv.reader(fline2, delimiter='\t', skipinitialspace='True', quotechar=None):
        # for sentence2 in fline2:
        # if line2[4] == '1' or line2[4] == '0':
        if line2[8] == '1' or line2[8] == '0':
            sentence2 = line2[1]

            sentence2 = sentence2.decode('utf-8')
            sentence2 = sentence2.lower()

            # sentence2 = " ".join(filter(lambda x:x[0]!='#', sentence2.split()))
            sentence2 = " ".join(filter(lambda x: x[0] != '@', sentence2.split()))

            sentence2 = sentence2.replace('rt', '')
            # sentence2 = sentence2.replace('#','')
            sentence2 = sentence2.replace('!', '')
            sentence2 = sentence2.replace('*', '')

            lst2 = [i for i in sentence2.lower().split() if i not in stop]

            for x2 in lst2:
                stringSecond += x2
                # if dictn2.has_key(stemmer.stem(x2.strip())):

                if dictn2.has_key(x2.strip()):
                    #dictn2[stemmer.stem(x2.strip())] += 1
                    dictn2[x2.strip()] += 1

                else:
                    #dictn2[stemmer.stem(x2.strip())] = 1
                    dictn2[x2.strip()] = 1


first_dict_size = len(dictn)
second_dict_size = len(dictn2)
sum_first = sum(dictn.values())
sum_second = sum(dictn2.values())
# print sum_first, ' ', sum_second

dictn = {k: (float)(v) / sum_first for k, v in dictn.items()}
dictn2 = {k: (float)(v) / sum_second for k, v in dictn2.items()}

tempdict = dictn2
tempdict.update(dictn)
list_keys = tempdict.keys()

list_vector_first = []
list_vector_second = []

for x in list_keys:
    # for first vector
    if dictn.has_key(x):

        list_vector_first.append(dictn[x])

    else:
        list_vector_first.append(0)

    # for second vector
    if dictn2.has_key(x):

        list_vector_second.append(dictn2[x])

    else:
        list_vector_second.append(0)

#print (len(list_vector_first))
# print('\n')
# ßprint (len(list_vector_second))
#print (list_vector_second)


def cosine_similarity2(list_vector_first, list_vector_second):

    first_vector_magnitude = 0
    second_vector_magnitude = 0
    product_vectors = 0

    for l in range(len(list_vector_first)):

        x = list_vector_first[l]
        y = list_vector_second[l]

        first_vector_magnitude += x * x
        second_vector_magnitude += y * y
        product_vectors += x * y

    return product_vectors / math.sqrt(first_vector_magnitude * second_vector_magnitude)


print 'Cosine Sim(main)- ', cosine_similarity2(list_vector_first, list_vector_second)

print 'Cosine Sim(inbuilt)- ', cosine_similarity(np.asarray([list_vector_first]), np.asarray([list_vector_second]))

documents = (stringSecond, stringFirst)
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words=stop, tokenizer=tokenize_and_stem,
                                   ngram_range=(1, 1), lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

count_vectorizer = CountVectorizer(analyzer='word', stop_words=stop, tokenizer=tokenize_and_stem,
                                   ngram_range=(1, 1), lowercase=True)
count_matrix = count_vectorizer.fit_transform(documents)

print 'Matrix Shape: ', tfidf_matrix.shape
print 'Cosine Sim TfIdf- ', cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

print 'Count Matrix Shape: ', count_matrix.shape
print 'Cosine Sim Count- ', cosine_similarity(count_matrix[0:1], count_matrix)

#-------------------------
print('------------------------')
