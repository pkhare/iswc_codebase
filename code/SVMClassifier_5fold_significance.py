import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from nltk.tokenize import regexp_tokenize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as prfs_score
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_scores
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import pickle
import json
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

import codecs
import csv
from scipy import stats
from sklearn.model_selection import KFold  # import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
reload(sys)
sys.setdefaultencoding('utf8')

stemmer = PorterStemmer()

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

fs = frozenset(s)


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


data_csv = []
data = ''


data_csv_balanced_full = []
data_balanced_full = ''
lang = dict()
correct_classify = dict()
incorrect_classify = dict()

# data_test

# with open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/stat_n_hypernym_dbpedia_en/balanced_random_stat_hypernym_dbpedia_en_semantics.csv') as fline,\
with open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/language_iswc/balanced/statistical_features/all_3_train_balanced_stat_random.csv') as fline,\
        open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/statistical_features/balanced_stat_random.csv') as flineX:
    # with open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/stat_n_dbpedia_en/filtered_dbpedia_semantics/balanced_random_stat_dbpedia_en_semantics.csv') as fline,\

    # with open('/Users/pk4634/Documents/phase2/extensionphase1/new_annotation_en_it_es_pt_fr_ru_hi/stat_n_hypernym_dbpedia_en/balanced_random_stat_hypernym_dbpedia_en_semantics.csv') as fline:

    data_csv_balanced_full = list(csv.reader(flineX, delimiter='\t', quoting=csv.QUOTE_NONE))
    data_balanced_full = np.array(data_csv_balanced_full[0:])

    for rec in data_balanced_full:
        lang[rec[0]] = rec[5]

    data_csv = list(csv.reader(fline, delimiter='\t', quoting=csv.QUOTE_NONE))

    data = np.array(data_csv[0:])

# For statistical_features file arrangement X = data[:, [1, 7, 8, 9, 10, 11, 12]]
X = data[:, [1, 7, 8, 9, 10, 11, 12]]
Y = data[:, 4].astype(np.float32)
# X = data[:, 1]  # just for the tweets based features
# Y = data[:, 4].astype(np.float32)
# Y = data[:, 2].astype(np.float32)  # jusr for the dbpedia balanced tweets

# X = data[:, 1:8]
# Y = data[:, 8].astype(np.float32)

t = ['Document', 'NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns',
     'TweetLength', 'NumberOfWords', 'NumberOfHashTag']

# t = ['Document']

# str(nouns), str(verbs), str(pronouns), str(tweet_length), str(token_count), str(numHashTag)

frm = pd.DataFrame(X, columns=t)
print type(frm)
# tokenizer=tokenize_and_stem,
vectorizer = CountVectorizer(analyzer='word',
                             stop_words=fs, lowercase=True, ngram_range=(1, 1), max_features=40000)
doc_vectorize = vectorizer.fit_transform(frm.Document)
tf_transform = TfidfTransformer()
tf_vectorize = tf_transform.fit_transform(doc_vectorize)

X_data = sp.sparse.hstack((tf_vectorize, frm[['NumberOfNouns', 'NumberOfVerbs', 'NumberOfPronouns', 'TweetLength',
                                              'NumberOfWords', 'NumberOfHashTag']].values.astype(np.float32)), format='csr')

# X_data = tf_vectorize
print type(X_data)
print type(Y)
print X_data.shape
# X_data.toarray().shape

# create training and testing vars
# X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.3)

#############################
print 'done1'
unique, counts = np.unique(Y, return_counts=True)
print dict(zip(unique, counts))

# svc = SVC(kernel='linear', degree=3, gamma='auto', tol=0.001)
svc_L = SVC(kernel='linear', degree=3, gamma='auto', tol=0.001)
svc_R = SVC(kernel='rbf', degree=3, gamma='auto', tol=0.001)
svc_P = SVC(kernel='poly', degree=3, gamma='auto', tol=0.001)
lr_cls = LogisticRegression(solver='sag')
# print X_train.shape
# print y_train.shape
# svc.fit(X_train, y_train)
print 'done2'
# scores = cross_validation.cross_val_score(svc, X_data, Y, cv=5)
# print scores
# print len(scores)

n_folds = 5
n_repeats = 10
# n_repeats = 1  # for evaluating language based analysis
my_rand_state = 0
skfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats,
                                 random_state=my_rand_state)


'''scores = cross_validation.cross_val_score(svc, X_data, Y, cv=skfold)
print scores
print len(scores)'''

last_score_L = []
pr_last_score_L = []
rc_last_score_L = []

last_score_R = []
last_score_P = []
last_score_lrcls = []
'''last_score_R = []
pr_last_score_R = []
rc_last_score_R = []

last_score_P = []
pr_last_score_P = []
rc_last_score_P = []

last_score_lrcls = []
pr_last_score_lrcls = []
rc_last_score_lrcls = []'''

c = 0
for train_index, test_index in skfold.split(X_data, Y):
    # print 'train index--', train_index
    # print 'test index--', test_index
    X_train, X_test = X_data[train_index], X_data[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    # print X_train.shape
    # print Y_train.shape
    print c
    list_test_id_indx = []
    for indx in test_index:
        list_test_id_indx.append(data[indx][0])
        # print data[indx][0]

    svc_L.fit(X_train, Y_train)
    Y_test_predict_L = svc_L.predict(X_test)

# to find the language prediction
    '''
    for l in range(0, len(Y_test)):
        if Y_test[l].astype(np.float32) == Y_test_predict_L[l].astype(np.float32):

            if lang[list_test_id_indx[l]] in correct_classify:
                correct_classify[lang[list_test_id_indx[l]]
                                 ] = correct_classify[lang[list_test_id_indx[l]]] + 1
            else:
                correct_classify[lang[list_test_id_indx[l]]] = 1

        else:
            # print 'n'
            if lang[list_test_id_indx[l]] in incorrect_classify:
                incorrect_classify[lang[list_test_id_indx[l]]
                                   ] = incorrect_classify[lang[list_test_id_indx[l]]] + 1
            else:
                incorrect_classify[lang[list_test_id_indx[l]]] = 1
    '''
    f1_val_L = f1_score(Y_test, Y_test_predict_L.astype(np.float32), average='macro')
    p_val_L = precision_score(Y_test, Y_test_predict_L.astype(np.float32), average='macro')
    r_val_L = recall_score(Y_test, Y_test_predict_L.astype(np.float32), average='macro')

    last_score_L = np.append(last_score_L, f1_val_L)
    pr_last_score_L = np.append(pr_last_score_L, p_val_L)
    rc_last_score_L = np.append(rc_last_score_L, r_val_L)

    # could be commented out for later

    svc_R.fit(X_train, Y_train)
    Y_test_predict_R = svc_R.predict(X_test)
    f1_val_R = f1_score(Y_test, Y_test_predict_R.astype(np.float32), average='macro')
    last_score_R = np.append(last_score_R, f1_val_R)

    print 'fake test'
    svc_P.fit(X_train, Y_train)
    Y_test_predict_P = svc_P.predict(X_test)
    f1_val_P = f1_score(Y_test, Y_test_predict_P.astype(np.float32), average='macro')
    last_score_P = np.append(last_score_P, f1_val_P)

    lr_cls.fit(X_train, Y_train)
    Y_test_predict_lrcls = lr_cls.predict(X_test)
    f1_val_lrcls = f1_score(Y_test, Y_test_predict_lrcls.astype(np.float32), average='macro')
    last_score_lrcls = np.append(last_score_lrcls, f1_val_lrcls)

    c = c + 1

mean_L = np.mean(last_score_L)
mean_pr_L = np.mean(pr_last_score_L)
mean_rc_L = np.mean(rc_last_score_L)

# to be commented out later
mean_R = np.mean(last_score_R)
mean_P = np.mean(last_score_P)
mean_lrcls = np.mean(last_score_lrcls)

std_L = np.std(last_score_L)
std_pr_L = np.std(pr_last_score_L)
std_rc_L = np.std(rc_last_score_L)

# to be commented out later
std_R = np.std(last_score_R)
std_P = np.std(last_score_P)
std_lrcls = np.std(last_score_lrcls)

'''t_LR = (mean_L - mean_R) / np.sqrt(((std_L * std_L) / len(last_score_L)) +
                                   ((std_R * std_R) / len(last_score_R)))

t_LP = (mean_L - mean_P) / np.sqrt(((std_L * std_L) / len(last_score_L)) +
                                   ((std_P * std_P) / len(last_score_P)))

t_Llrcls = (mean_L - mean_R) / np.sqrt(((std_L * std_L) / len(last_score_lrcls)) +
                                       ((std_lrcls * std_lrcls) / len(last_score_lrcls)))

df_LR = 2 * len(last_score_L) - 2

p_LR = 1 - stats.t.cdf(t_LR, df=df_LR)
p_LP = 1 - stats.t.cdf(t_LP, df=df_LR)
p_Llrcls = 1 - stats.t.cdf(t_Llrcls, df=df_LR)

print("t_LR = " + str(t_LR))
print("p_LR = " + str(2 * p_LR))

print("t_LP = " + str(t_LP))
print("p_LP = " + str(2 * p_LP))

print("t_Llrcls = " + str(t_Llrcls))
print("p_Llrcls = " + str(2 * p_Llrcls))'''


print c
print len(last_score_L)
print("mean fvalue L = ") + str(mean_L)
print("std fvalue L = ") + str(std_L)
print("mean pr L = ") + str(mean_pr_L)
print("std pr L = ") + str(std_pr_L)
print("mean rc L = ") + str(mean_rc_L)
print("std rc L = ") + str(std_rc_L)
print '################'
# print last_score_L

'''print '################'
print last_score_R
print '################'
print last_score_P
print '################'
print last_score_lrcls'''

# df = len(last_score_L) - 1
# t_score = mean_L

print 'correct_classify:\n'
# print correct_classify
print 'incorrect_classify:\n'
# print incorrect_classify
print("mean fvalue R = ") + str(mean_R)
print("std fvalue R = ") + str(std_R)

print("mean fvalue P = ") + str(mean_P)
print("std fvalue P = ") + str(std_P)

print("mean fvalue lrcls = ") + str(mean_lrcls)
print("std fvalue lrcls = ") + str(std_lrcls)
