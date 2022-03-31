# Document-Retrievel-Models
## About the Project
This project involves implementation of various Information Retrieval systems like Boolean retrieval model, BM25 family system and TF-IDF family system. All of these systems uses processed document. Processing of document involves tokenization, stemming, lemmatization, removing stopwords, etc. TF-IDF family system is implemented with appropriate forms of the functions and tuned parameters and query is matched using cosine similarity. Similarly BM25 is also implemented with appropriate forms of the functions and tuned parameters. Since all of these are IR systems, we have ensured that the implementations for all of these are efficient.

## Prequisites
- Follow this URL(https://www.cse.iitk.ac.in/users/arnabb/ir/english/) to download english-corpora.zip 
  - Please download and extract above zip file and put “english-corpora” folder in parent directory 'Assignment1'.
- Insert the queries text file in parent directory 'Assignment1'
- Pickle file which contain preprocessed data (we get this data from question 1)
  - Pickle file already provided in folder Question_1

## Libraries Used
ntlk
```sh
npm install nltk
```
pickle
```sh
npm install pickle
```
glob
```sh
npm install glob
```
re
```sh
npm install re
```
counter
```sh
npm install counter
```

## File Description
| Plugin | README |
| ------ | ------ |
|Question_1.py|lets you do the cleaning, processing of data and also helps to arrange data in a structured way using nltk and re python libraries.|
|Question_2|consists of 3 python files in which all 3 models have been implemented. It takes “queries.txt” from command line as an argument and returns “*.txt” corresponding to each model with comma separated format. |
|Question_3|contains 2 manually written files with .txt format.|
|Question_4|contains 3 model generated files with .txt format named as boolean.txt, tfidf.txt, and bm25.txt with top 5 document search. These files are the result of each query running in each model.|
|Question_5|contains makefile which runs all 3 models in one click.|


## Commands to run each python file
**Question 1**
```sh
python3 preprocessing.py or python preprocessing.py
```
**Question 2**
```sh
python3 models.py queries_file_name.txt
```
or
```sh
python models.py queries_file_name.txt
```
**Make File**
```sh
make run ARGS=”query_file_name.txt”
```
NOTE: If you want to do preprocessing part, use make file in Question_1 folder only
