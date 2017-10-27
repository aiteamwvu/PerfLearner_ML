from newspaper import Article
from rake_nltk import Rake
import json
import pickle
#import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np



with open('trainingdata.pickle', 'rb') as f:
    all_documents, y, i = pickle.load(f)

tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)        
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)    
   
print('The number of valid training articles in the data set is {i:d}'.format(i=i))

# given any url as below, parse the article and obtain its text data
url = 'https://www.digitaltrends.com/cool-tech/'
art = Article(url, language='en')  # English
try:
    art.download()
    art.parse()    
    art_content =  art.text
    art_tr = sklearn_tfidf.transform([art_content])  # obtain its transformation
except:
    print('bad article, probably a video or the url is broken')


#obtain the tensorflow graph from saved session
sess = tf.Session()
saver = tf.train.import_meta_graph('mynet.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

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


#you should obtain the following results
#array([[ 0.76648879,  0.23351125]], dtype=float32)
