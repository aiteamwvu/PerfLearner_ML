from newspaper import Article
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import numpy as np

    
with open('trainingdata3keywords.pickle', 'rb') as f:
    all_documents, y, i = pickle.load(f)

tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)        
sklearn_tfidf.fit_transform(all_documents)    
   
print('The number of valid training articles in the data set is {i:d}'.format(i=i))
counter = 0
file = open("example_1.json", 'r', encoding = 'utf-8').read()
record = json.loads(file)
for article in record:
    if 'validated' not in article:
        try:
            counter += 1
            art = Article(article['_id'], language='en')  # English
            art.download()
            art.parse()    
            art_content =  art.text
            art_tr = sklearn_tfidf.transform([art_content])  # obtain its transformation

            sess = tf.Session()
            
            saver = tf.train.import_meta_graph('mynet1.meta')
            
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            
            all_vars = tf.get_collection('vars')
            W_1 = all_vars[0]
            W_2 = all_vars[1]
            W_3 = all_vars[2]
            W_4 = all_vars[3]
            W_fin = all_vars[4]
            b_1= all_vars[5]
            b_2 = all_vars[6]
            b_3 = all_vars[7]
            b_4 = all_vars[8]
            b_fin = all_vars[9]
            
            xtest = art_tr.toarray()
            xtest = xtest.astype(np.float32)
            out_1 = tf.nn.sigmoid(tf.matmul(xtest,W_1) + b_1)
            drop_out_1 = tf.nn.dropout(out_1 , keep_prob = 0.5) 


            out_2 = tf.nn.sigmoid(tf.matmul(drop_out_1,W_2) + b_2)
            drop_out_2 = tf.nn.dropout(out_2 , keep_prob = 0.5) 
            
            #layer 3
            out_3 = tf.nn.relu(tf.matmul(drop_out_2,W_3) + b_3)
            drop_out_3 = tf.nn.dropout(out_3 , keep_prob = 0.5) 
            
            #layer 4
            out_4 = tf.nn.sigmoid(tf.matmul(drop_out_3,W_4) + b_4)
            drop_out_4 = tf.nn.dropout(out_4 , keep_prob = 0.5) 
            
            #Output layer
            out_fin = tf.nn.relu(tf.matmul(drop_out_4,W_fin) + b_fin)
            
            # convert output final (out_fin) to probabilities
            out_probs = tf.nn.softmax(out_fin)
            result = sess.run(out_probs)[0]
            print('finished nnet')
            if result[0] > result[1]: #'article matches users interest' otherwise 'article probably doesnt match our users interest'
                article['validated'] = 1
            else:
                article['validated'] = -2
            sess.close()    
            print('finished validation')
            print({"status":"OK"})
        except:
            article['validated'] = -2
            print({"status":"error with file id " + article["_id"]})
            #obtain the tensorflow graph from saved session
        if counter%50 ==0:
            print('\n\n Number of articles processed is {counter:d}\n\n'.format(counter=counter))
        
