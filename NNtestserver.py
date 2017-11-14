from newspaper import Article
from rake_nltk import Rake
import json
import pickle
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np

from flask import Flask, request
from flask_cors import CORS
import requests
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    conn = pymongo.MongoClient()["test"]["Article"]
    record = conn.find({"validated":{"$exists" : False}})

    with open('trainingdata.pkl', 'r') as f:
        all_documents, y, i = pickle.load(f)

    tokenize = lambda doc: doc.lower().split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)        
    sklearn_tfidf.fit_transform(all_documents)    
   
    print('The number of valid training articles in the data set is {i:d}'.format(i=i))

    for article in record:
        try:
            art = Article(article['_id'], language='en')  # English
            art.download()
            art.parse()    
            art_content =  art.text
            art_tr = sklearn_tfidf.transform([art_content])  # obtain its transformation
            
            sess = tf.Session()
            saver = tf.train.import_meta_graph('mynet.meta')
            saver.restore(sess.tf.train.latest_checkpoint('./'))
            
            xtest = art_tr.toarray()
            xtest = xtest.astype(np.float32)
            out_1 = tf.nn.sigmoid(tf.matmul(xtest,W_1) + b_1)
            drop_out_1 = tf.nn.dropout(out_1 , keep_prob = 0.5) 
            
            #layer 2
            #out_2 = tf.nn.relu(tf.matmul(drop_out_1,W_2) + b_2)
            out_2 = tf.nn.sigmoid(tf.matmul(drop_out_1,W_2) + b_2)
            drop_out_2 = tf.nn.dropout(out_2 , keep_prob = 0.5) 
            
            #layer 3
            out_3 = tf.nn.relu(tf.matmul(drop_out_2,W_3) + b_3)
            drop_out_3 = tf.nn.dropout(out_3 , keep_prob = 0.5) 
            
            #layer 4
            #out_4 = tf.nn.relu(tf.matmul(drop_out_3,W_4) + b_4)
            out_4 = tf.nn.sigmoid(tf.matmul(drop_out_3,W_4) + b_4)
            drop_out_4 = tf.nn.dropout(out_4 , keep_prob = 0.5) 
            
            #Output layer
            out_fin = tf.nn.relu(tf.matmul(drop_out_4,W_fin) + b_fin)
            
            # convert output final (out_fin) to probabilities
            out_probs = tf.nn.softmax(out_fin)
            
            if out_probs[0] > out_probs[1]: #'article matches users interest' otherwise 'article probably doesnt match our users interest'
                article['validated'] = 1
            else:
                article['validated'] = 0
            conn.save(article)
            sess.close()    
            print({"status":"OK"})
        except:
            article['validated'] = 0
            conn.save(article)
            print({"status":"error with file id " + article["_id"]})
            #obtain the tensorflow graph from saved session
    return 'OK'

app.run(host='0.0.0.0', threaded=True, port=5005)
