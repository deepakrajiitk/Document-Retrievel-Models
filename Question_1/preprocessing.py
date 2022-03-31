#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[2]:


import glob
import os
import re
from collections import Counter
import numpy as np
import nltk
import sys
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ### Importing files

# In[3]:


path = sys.path[0] + '/../english-corpora'
files = glob.glob(path + '/**/*.txt', 
                   recursive = True)
n_docs = len(files)


# In[51]:


files = sorted(files)


# In[52]:


# mapping of int to file name for better access of files
file_index={}
i=0
for file in files:
    file_index[i] = file[18:]
    i+=1


# In[53]:


# getting all documents in a list
docs = []
for i in range(n_docs):
    docs.append(open(files[i], encoding="utf8").read())


# In[46]:


stemmer = PorterStemmer()


# ### cleaning documents

# - Steps used to clean documents:
#     - removing all characters except a-z, A-Z and 0-9 using ‘re’ library
#     - changing all cases letters to lower case
#     - tokenization using ‘split’
#     - removing all stopwords
#     - lemmatization using ntlk library

# In[54]:


# list which stores all cleaned documents
clean_docs = []
# list which stores no of words in each document 
docs_nwords_list = []
all_words = np.array([])
for doc in docs:
#     removing all characters expect a-z, A-Z and 0-9 using regular expression
    doc = re.sub('[^a-zA-Z0-9]',' ',doc)
#     changing all cases to lower case
    doc = doc.lower()
#     this step will give all the words as tokens
    doc = doc.split()
#     1. removes all stop words
#     2. performs lemmatization using standard library
    doc = np.array([stemmer.stem(word) for word in doc if not word in set(stopwords.words('english'))])
#     docs_nwords_list.append(doc.size)
    clean_docs.append(doc)


# ### Creating required lists or dictionaries

# In[55]:


main_dict = {} # main dictionary whose keys are all the unique words and the value is a dictionary containing file id as key and frequency of that word as value
# format => {word : {fid : frequency}}
norm_list = [] # list containing norm of each document 
i=0
for doc in clean_docs:
#     counter library is used to find the frequency of each word in a particular document
    words_freq_dict = Counter(doc) 
#     finding norm of each document
    values = np.array(list(words_freq_dict.values()))
    norm = np.linalg.norm(values)
    norm_list.append(norm)
#     creating main dictionary
    for key in words_freq_dict.keys():
        if key in main_dict.keys():
            main_dict[key].update({i:words_freq_dict[key]})
        else:
            main_dict[key] = {i: words_freq_dict[key]}
    i+=1


# In[58]:


# main_dict


# In[60]:


# norm_list


# ### Creating pickle file

# In[65]:


import pickle
database = {}
database['main_dict'] = main_dict
database['norm_list'] = norm_list
database['docs_nwords_list'] = docs_nwords_list
database['n_docs'] = n_docs
database['file_index'] = file_index
dbfile = open('pickle_data', 'ab')
pickle.dump(database, dbfile)
dbfile.close()

