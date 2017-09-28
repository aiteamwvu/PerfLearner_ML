# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:32:27 2017

@author: Samson

I created a bunch of texts from websites using the following searches on google.com and clicked on the first
link:
    techcrunch.com gadgets
    wired.com electronics
    theverge.com latest phones
    gizmodo accessories and gadgets
    random news on bulgaria
    phone reviews
    wifi devices in home
    cooking italian sausages
    asus phone
    lg laptop review
    
    
    the following websites were obtained:
        https://techcrunch.com/video/gadgets/
        https://www.wired.com/2017/04/soon-print-simple-electronics-deskjet/
        https://www.theverge.com/phone-review
        https://gizmodo.com/tag/gadgets
        https://www.boredpanda.com/googly-eyebombing-street-art-bulgaria/
        http://www.techradar.com/news/phone-and-communications/mobile-phones/20-best-mobile-phones-in-the-world-today-1092343
        https://www.tomsguide.com/us/best-smart-home-gadgets,review-2008.html
        http://www.foodnetwork.com/recipes/alton-brown/italian-sausage-recipe-2131680
        https://www.androidcentral.com/these-are-top-asus-phones-you-need-know
        https://www.pcworld.com/article/3050738/hardware/lg-gram-15-review-this-shockingly-lightweight-laptop-is-beyond-ultraportable.html
    
    I copied the text from these webpages into csv files and labelled them
        doc1
        doc2
        doc3
        doc4
        doc5
        doc6 
        doc7
        doc8
        doc9
        doc10
        
    Now for testing we would like to see how the NN in tensorflow rates the following websites:
        https://www.pcworld.idg.com.au/review/lg/g6-phone/616890/
        http://fortune.com/2017/02/17/smart-home-tech-internet-of-things-connected-home/
        
        obtained from:
        lg phone reviews    
        home internet of things devices
        
    I copied the text from these webpages into csv files and labelled them 
        doc11
        doc12
"""

import csv
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np

print(__doc__)
# Define the tokenizer
tokenize = lambda doc: doc.lower().split(" ")

# create a default dictionary which would contain text from documents as lists
combined  = defaultdict(list)

for i in range(1,13):
    docname = 'doc' + str(i) + '.csv'
    with open(docname) as file_to_read:
        reader = csv.reader(file_to_read)
        
        for row in reader:
            combined[i].append(row)
        combined[i] = ''.join([item for inner in combined[i] for item in inner])

#Combine all documents into a single all_document        
all_documents = [content for _, content in combined.items()]

# Get the sklearn tfidf and transform the all_document based on the tfidf
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)

# define a generic function to obtain variables and bias for the Neural net
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

sz = sklearn_representation.toarray().shape
x = tf.placeholder(tf.float32, [None,  sz[1]])
y_ = tf.placeholder(tf.float32, [None,  2])

# training data
xtrain = sklearn_representation.toarray()[:10,:]
ytrain = [[1,0],
      [1,0],
      [1,0],
      [1,0],
      [0,1], # not like -  bulgria news
      [1,0],
      [1,0],
      [0,1], # not like - italian pasta
      [1,0],
      [1,0]]

# Test data
xtest = sklearn_representation.toarray()[10:,:]

# parameters of the Neural net
nn1 = 10 #number of neurons in layer_1
nn2 = 10 #number of neurons in layer_2

# Create a tensoflow computational graph for the NN with dropout to prevent overfitting
# layer 1
W_1 = weight_variable([sz[1],nn1])
b_1 = bias_variable([nn1])
out_1 = tf.nn.relu(tf.matmul(x,W_1) + b_1)
drop_out_1 = tf.nn.dropout(out_1 , keep_prob = 0.5) 

#layer 2
W_2 = weight_variable([nn1,nn2])
b_2 = bias_variable([nn2])
out_2 = tf.nn.relu(tf.matmul(drop_out_1,W_2) + b_2)
drop_out_2 = tf.nn.dropout(out_2 , keep_prob = 0.5) 

#Output layer
W_fin = weight_variable([nn2,2])
b_fin = bias_variable([2])
out_fin = tf.nn.relu(tf.matmul(drop_out_2,W_fin) + b_fin)

# convert output final (out_fin) to probabilities
out_probs = out_fin/tf.transpose([tf.reduce_sum(out_fin , 1)])

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out_fin))

#Training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# predictions and accuracy
correct_prediction = tf.equal(tf.argmax(out_fin,1), tf.argmax(y_,1))
accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#run session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    sess.run(train_step, feed_dict = {x:xtrain, y_: ytrain})

print('The accuracy is %s %%'% (sess.run(accuracy, feed_dict = {x:xtrain, y_: ytrain})))

# predictions on test data
print(sess.run(out_probs, feed_dict = {x: xtest}))

