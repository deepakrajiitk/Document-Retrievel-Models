#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[68]:


import re
import math
import pickle
import csv
import sys
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ### Importing relevant data from pickle file

# In[69]:


dbfile = open(sys.path[0]+'/../Question_1/pickle_data', 'rb')     
database = pickle.load(dbfile)


# In[70]:


main_dict = database['main_dict']
norm_list = database['norm_list']
docs_nwords_list = database['docs_nwords_list']
n_docs = database['n_docs']
file_index = database['file_index']


# In[71]:


len(main_dict)


# In[72]:


stemmer = PorterStemmer()


# In[73]:


def cleanQuery(query):
    # applying same operations on query as applied on each document 
    query = re.sub('[^a-zA-Z0-9]',' ',query)
    query = query.lower()
    query = query.split()
    query = np.array([stemmer.stem(word) for word in query if not word in set(stopwords.words('english'))])
    return query


# In[74]:


# this function return top ten relevant documents and their relevance status
def find_top_relevant(score, n):
#     getting top 10 relevant documents id
    score_index = np.argsort(score)[::-1][:n]
#     getting
    relevance = [0 if score[i]==0 else 1 for i in score_index]
    relevant_files = list(map(file_index.__getitem__, score_index))
    return relevant_files, relevance


# In[75]:


# this function calculates cosine similarities between document vectors and query vector
def find_cosine_sim(n_docs, query_vector, doc_vectors):
    cosine_sims = []
    for i in range(n_docs):
        dot = np.dot(query_vector, doc_vectors[i])
        query_norm = np.linalg.norm(query_vector)
        doc_norm = norm_list[i]
        cosine_sims.append(dot/(query_norm*doc_norm))
    return cosine_sims


# ### Boolean Retreival Model

# In[83]:


# cleaned query must be sent to this model
def brModel(query, main_dict, n_docs):
    boolean_dict = {} # dictionary storing boolean vector of each word
    # format => {word : list}
    for word in np.unique(query):
        bit_vector = [0 for i in range(n_docs)]
        if word in main_dict.keys():
            word_dict = main_dict[word]
            for key in word_dict.keys():
                bit_vector[key] = 1
        boolean_dict[word] = bit_vector

    # applying '&' operations on each boolean vector
    ans = [1 for i in range(n_docs)]
    for value in boolean_dict.values():
        for i in range(n_docs):
            ans[i] = ans[i] & value[i]
    return find_top_relevant(ans,10)


# ### Implementing TFIDF family system

# Formulae used:
# \begin{align*}
# & TF = \text{no of time a word occurs in a document} \\
# & idf = \ln\frac{N+1}{df_t+1} \\
# & tfidf = tf*idf \\
# where, \\
# & tf = \text{term frequency} \\
# & idf = \text{ document frequency} \\
# & N = \text{total no of documents} \\
# & df_t = \text{no of document in which a particular term is presnt}
# \end{align*}

# In[84]:


# cleaned query must be sent to this model
def tfidf(n_docs, docs_nwords_list, main_dict, query):
    # finding document vectors
    tfidf_dict = {} # dictionary which stores tfidf of each document
    doc_vectors = [] 
    for i in range(n_docs):
        n_words = docs_nwords_list[i]
        vector = []
        for word in np.unique(query):
            if word in main_dict.keys() and i in main_dict[word].keys():
#                 tf = main_dict[word][i]/n_words
                tf = main_dict[word][i]
                idf = math.log((n_docs+1)/(len(main_dict[word].keys())+1))
                tf_idf = tf*idf
                vector.append(tf_idf)
            else:
                vector.append(0)
        doc_vectors.append(vector)
    word_freq_query_dict = Counter(query)
    # finding query vector
    query_vector = []
    # remember np.unique return result in alphabetical order
    for word in np.unique(query):
        if word in main_dict.keys():
#             tf = word_freq_query_dict[word]/len(query)
            tf = word_freq_query_dict[word]
            idf = math.log((n_docs+1)/(len(main_dict[word].keys())+1))
            tf_idf = tf*idf
            query_vector.append(tf_idf)
        else:
            query_vector.append(0)
    scores = find_cosine_sim(n_docs, query_vector, doc_vectors)
    return find_top_relevant(scores, 10)


# ### Implementing BM25

# Formulae used 
# $$ \sum_{\forall t \in q} (1 + \ln (\frac{n - df_t + 0.5}{df_t + 0.5})) . \frac{(k1+1).(tf_d)}{tf_d + k1 * (1-b+b*\frac{L_d}{L_{avg}})}$$
# where,
# $$ n - \text{total documents} $$
# $$ df_t - \text{total no of ducuments in which term is present} $$
# $$ tf_d - \text{frequency of term in a document d} $$
# $$ k1, b - \text{tuning parameters} $$

# In[85]:


# clean query must be sent to this model
def BM25(b, k1, main_dict, docs_nwords_list, n_docs, query):
    score = []
    L_avg = sum(docs_nwords_list)/len(docs_nwords_list) #avg length of document
    for i in range(n_docs):
        L_d = docs_nwords_list[i] # length of document
        ans = 0
        for word in np.unique(query):
            if word in main_dict.keys() and i in main_dict[word]:
                df_t = len(main_dict[word].keys())+1
                tf_d = main_dict[word][i]
                idf = math.log(1 + (n_docs - df_t + 0.5 )/(df_t+0.5))
                first_term = (k1+1)*tf_d
                second_term = tf_d + k1*(1-b+b*(L_d/L_avg))
                temp = idf * (first_term/second_term)
                ans += temp
        score.append(ans)
        top_relevant = find_top_relevant(score, 10)
    return top_relevant


# ### Reading query file

# In[86]:


queries_fname = sys.argv[1] # reading file name from command line


# In[107]:


import pandas as pd
try:
    df = pd.read_csv(sys.path[0]+'/../'+queries_fname, sep="\t")
    queryIds = df['QueryId'].values
    queries = df['Query'].values
except:
    df = pd.read_csv(sys.path[0]+'/../'+queries_fname, sep="\t",header=None)
    queryIds = df. iloc[:, 0].values
    queries = df.iloc[:, 0].values


# ### Running set of queries on different algorithms 

# In[93]:


BRM_result = []
TFIDF_result = []
BM25_result = []

print("Getting result from all the models for given queries")
for i in range(len(queries)):
#     cleaning each query
    query = cleanQuery(queries[i])
#     getting result from BM25 algorithm
    result1 = brModel(query, main_dict, n_docs)
    result2 = tfidf(n_docs, docs_nwords_list, main_dict, query)
    result3 = BM25(0.75, 1.2, main_dict, docs_nwords_list, n_docs, query)

    for j in range(10):
        row1 = [queryIds[i], 1, result1[0][j], result1[1][j]]
        BRM_result.append(row1)
        row2 = [queryIds[i], 1, result2[0][j], result2[1][j]]
        TFIDF_result.append(row2)
        row3 = [queryIds[i], 1, result3[0][j], result3[1][j]]
        BM25_result.append(row3)
        


# ### Saving result of Boolean retreival model

# In[ ]:


f = open(sys.path[0]+'/../Question_4/boolean.txt', 'w')
writer = csv.writer(f)
writer.writerow(['QueryId', 'Iteration', 'DocId', 'Relevance'])
writer.writerows(BRM_result)
f.close()


# ### Saving result of TFIDF model

# In[52]:


f = open(sys.path[0]+'/../Question_4/tfidf.txt', 'w')
writer = csv.writer(f)
writer.writerow(['QueryId', 'Iteration', 'DocId', 'Relevance'])
writer.writerows(TFIDF_result)
f.close()


# ### Saving result of BM25 model

# In[ ]:


f = open(sys.path[0]+'/../Question_4/bm25.txt', 'w')
writer = csv.writer(f)
writer.writerow(['QueryId', 'Iteration', 'DocId', 'Relevance'])
writer.writerows(BM25_result)
f.close()


# In[ ]:


print("Files have been successfully created for all the models")


# In[58]:


# f = open(sys.path[0]+'/../Question_3/Ground_Truth.txt', 'w')
# writer = csv.writer(f)
# writer.writerow(['QueryId', 'Iteration', 'DocId', 'Relevance'])
# writer.writerows(BM25_result)
# f.close()


# In[8]:


# import pandas as pd
# df = pd.read_csv("QRelevance.txt")


# In[9]:


# df

