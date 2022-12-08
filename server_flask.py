

from collections import OrderedDict

import inspect as inspector

import flask
import json
from flask import Flask, render_template
from flask import request
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util

import pickle
import tensorflow as tf
from flask_cors import CORS, cross_origin

# if you are using python 3, you should 
import urllib.request

app = Flask(__name__)
CORS(app,resources={r"/getOutput": {"origins": "*"}})


senTransformer = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1')
model = pickle.load(open('mnb_model_v1.pkl','rb'))
count_vect = pickle.load(open('countvect_v1.pkl','rb'))
url = f'http://34.85.227.59:8983/solr'

BATCH_SIZE=50
MODEL_NAME1="msmarco-distilbert-base-dot-prod-v3"
model1 = SentenceTransformer(MODEL_NAME1)
# if torch.cuda.is_available():
#     model1 = model1.to(torch.device("cuda"))
queries = [{"topic":'education','values':[]},{"topic":'healthcare','values':[]},
        {"topic":'politics','values':[]},{"topic":'technology','values':[]},{"topic":'nature','values':[]},
        {"topic":'chitchat','values':[]}]

def processSentence(sen):
    embeddings = model1.encode(sen)
    return str(list(embeddings))

def sendRequestToChitchat(inputQuery):
    reqUrl = url+'/chitchat_v2/query?q=question:'+inputQuery+'&fl=*,score'
    reqUrl = reqUrl.replace(" ", "%20")
    content = urllib.request.urlopen(reqUrl).read()
    #content = urllib.request.urlopen( url+'/chitchat_v2/query?q=question:'+inputQuery+'&fl=*,score' ).read()
    response = json.loads(content)
    docs = response['response']['docs']
    answers = []
    queries[0]['values'].append(0)
    queries[1]['values'].append(0)
    queries[2]['values'].append(0)
    queries[3]['values'].append(0)
    queries[4]['values'].append(0)
    queries[5]['values'].append(1)
    if len(docs)==0:
        return "Didn't get you."
    for doc in docs:
        answers.append(doc['answer']) 
    query_emb = senTransformer.encode(inputQuery)
    doc_emb = senTransformer.encode(answers)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = list(zip(answers, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    result = []
    for doc, score in doc_score_pairs:
        temp = {}
        temp['score']=score
        temp['doc']=doc
        result.append(temp)
    return result[0]['doc']
        


def sendRequestToReddit(inputQuery,topic):
    queryVector = processSentence(inputQuery)
    if len(topic) != 0:
        reqUrl = url+'/IR_reseach/query?fq=topic:'+topic[0]+'&indent=true&q.op=AND&q={!knn f=question_vector}'+queryVector
    else:
        reqUrl = url+'/IR_reseach/query?indent=true&q={!knn f=question_vector}'+queryVector
    reqUrl = reqUrl.replace(" ", "%20")
    content = urllib.request.urlopen(reqUrl).read()
    response = json.loads(content)
    docs = response['response']['docs']
    answers = []
    if len(docs)==0:
        return "Didn't get you."
    edu,nature,tech,health,politics = 0,0,0,0,0
    for doc in docs:
        if doc['topic'] == 'nature':
            nature+=1
        elif doc['topic'] == 'education':
            edu+=1
        elif doc['topic'] == 'technology':
            tech+=1
        elif doc['topic'] == 'healthcare':
            health+=1
        elif doc['topic'] == 'politics':
            politics+=1
        answers.append(doc['answer']) 
    edu/=len(docs)
    nature/=len(docs)
    tech/=len(docs)
    health/=len(docs)
    politics/=len(docs)
    queries[0]['values'].append(edu)
    queries[1]['values'].append(health)
    queries[2]['values'].append(politics)
    queries[3]['values'].append(tech)
    queries[4]['values'].append(nature)
    queries[5]['values'].append(0)
    query_emb = senTransformer.encode(inputQuery)
    doc_emb = senTransformer.encode(answers)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = list(zip(answers, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    result = []
    for doc, score in doc_score_pairs:
        temp = {}
        temp['score']=score
        temp['doc']=doc
        result.append(temp)
    return result[0]['doc']

@app.route("/getOutput", methods=['POST'])
@cross_origin(origin='*')
def execute_query():
    data = request.get_json()
    topic = data['topics']
    if len(topic)==0:
        te = count_vect.transform([data['input']])
    else:
        te = count_vect.transform([topic[0]+':'+data['input']])
    inputQuery = data['input']
    op = model.predict(te)
    print(op)
    if op == 'chitchat':
        answer = sendRequestToChitchat(inputQuery)
    else:
        answer = sendRequestToReddit(inputQuery,topic)
   
    response = {
        "response":answer
    }
    return response

@app.route("/getStatistics", methods=['GET'])
@cross_origin(origin='*')
def execute_statistics():
    return queries


@app.route("/app")
@cross_origin(origin='*')
def execute_html():
    return render_template('index.html')

@app.route("/statistics")
@cross_origin(origin='*')
def execute_statistics_page():
    return render_template('statistics.html')

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5050)
