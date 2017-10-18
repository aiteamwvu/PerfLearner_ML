from newspaper import Article
from rake_nltk import Rake
import json
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
# parameters are as follows:
# text (string: text to obtain keywords from)
def getKeywords(text, numOfKeywords, title=""):

    # remove words which are not valid keywords or are not useful
    # parameters are as follows:
    # words (list of strings: words to scrub of junk data and irrelevant data)
    def scrubList(words):

        invalidWords = []

        i = 0
        while (i < len(words)):
            # remove everything that's invalid
            # currently it will remove: unicode codes, specified strings in the invalidWords list
            if ('\u' in words[i].encode('raw_unicode_escape') or words[i] in invalidWords):
                del words[i]
            else:  # otherwise increment iteration
                i += 1
        return words

    # clean keywords; some keywords from rake have junk in them
    # parameters are as follows:
    # words (list of strings: words to scrub of junk data and irrelevant data)
    def scrubWord(word):
        word = word.replace(",", "")
        word = word.replace(".", "")

        return word

    rak = Rake()  # english by default

    # extract keywords and store them+their degree in a dictionary
    rak.extract_keywords_from_text(text)
    wordDic = rak.get_word_degrees()

    # use the title if relevant
    if(title != ""):
        # make it a list by splitting on whitespace
        titleWords = title.split()

        # lower everything for accuracy
        for it in range(0, len(titleWords)):
            titleWords[it] = titleWords[it].lower()

        # if a keyword was in the title, double it's weight because it's likely very relevant
        for word in wordDic:
            if(word in titleWords):
                wordDic[word] = wordDic[word]*2

    rankedWords = sorted(wordDic, key=wordDic.get, reverse=True)
    rankedWords = scrubList(rankedWords)
    returnDic = {}

    if (numOfKeywords > len(rankedWords)):
        numOfKeywords = len(rankedWords)

    for it in range(0, numOfKeywords):
        temp = rankedWords[it]
        temp = scrubWord(temp) # scrub the word to be stored without changing it's value in the list

        # add the new, scrubbed word and it's weight
        returnDic[temp.encode('UTF8')] = (wordDic[rankedWords[it]] * (1.0/len(wordDic)))

    return returnDic


##############################################################################
userKeyWords = [{'google', 'internet', 'apple', 'icloud', 'intel',
                'microsoft',  'camera', 'battery', 'intelligence', 'windows',
                'pixel', 'machine', 'android', 'smart', 'logic', 'memory',
                'technology', 'iphone', 'gadgets', 'ibm', 'programming', 
                'cyber', 'malware', 'autonomous', 'networks', 'learning'},
                {'computers', 'computing', 'computer'},
                {'phones', 'phone'},
                {'smartphone', 'smartphones'},
                {'processor','processors', 'processing'},
                {'ai', 'artificial'},
                {'robot', 'robotics', 'robots', 'bot', 'bots'},
                {'wifi', 'wireless'},
                {'algorithm', 'algorithms'}]

conn = pymongo.MongoClient()["test"]["Article"]
all_documents = []

y = []
i = 0
file = open("example.json").read()
arr = json.loads(file)
for article in arr:
    if "content" in article and article['source_content'] is not 'video':
        i += 1
        url = article['_id']
        art = Article(url, language='en')  # English
        art.download()
        art.parse()    
        art_content =  art.text
        
        Keywords = getKeywords(art_content, 20)
        keys = {key for key in Keywords}
        
        count = 0
        count += len(set.intersection(keys, userKeyWords[0]))
        for sets in userKeyWords[1:]:
            if set.intersection(sets, keys):
                count+=1
                
        all_documents.append(art_content)        
        
        record = conn.find_one({"_id":url})
        record['nn_instance'] = 'train'
        if count >=2:
            y.append([1,0])
            record["validated"] = 1                   
        else:
            y.append([0,1]) 
            record["validated"] = -1
        
        conn.save(record)        

y = np.array(y)

tokenize = lambda doc: doc.lower().split(" ")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)        
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)   


# create training data, validation data and test data
Tfid_array = sklearn_representation.toarray()
sz = Tfid_array.shape

like = np.where(y[:,0]==1)
np.random.shuffle(like)
train_num1 = int(0.7*len(like))
train_ind1 = like[: train_num1]
val_ind1    = like[train_num1:]

unlike = np.where(y[:,0]==0)
np.random.shuffle(unlike)
train_num2 = int(0.7*len(unlike))
train_ind2 = like[: train_num1]
val_ind2    = like[train_num2:]

train_ind = np.r_[train_ind1, train_ind2]
val_ind   = np.r_[val_ind1,   val_ind2]


# training data
xtrain = Tfid_array[train_ind,:]
ytrain = y[train_ind:]

# validation data
xval = Tfid_array[val_ind,:]
yval = y[val_ind:]

#train
x = tf.placeholder(tf.float32, [None,  sz[1]])
y_ = tf.placeholder(tf.float32, [None,  2])

# define a generic function to obtain variables and bias for the Neural net
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



nn1 = 20 #number of neurons in layer_1
nn2 = 20 #number of neurons in layer_2
nn3 = 20 #number of neurons in layer_3
nn4 = 20 #number of neurons in layer_2
beta = 0.01 # regularization coefficient

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

#layer 3
W_3 = weight_variable([nn2,nn3])
b_3 = bias_variable([nn3])
out_3 = tf.nn.relu(tf.matmul(drop_out_2,W_3) + b_3)
drop_out_3 = tf.nn.dropout(out_3 , keep_prob = 0.5) 

#Output layer
W_fin = weight_variable([nn4,2])
b_fin = bias_variable([2])
out_fin = tf.nn.relu(tf.matmul(drop_out_3,W_fin) + b_fin)

# convert output final (out_fin) to probabilities
out_probs = tf.nn.softmax(out_fin)

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out_fin))

# Loss function with L2 Regularization with beta=0.01
regularizers = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3) +  tf.nn.l2_loss(W_fin)
loss = tf.reduce_mean(loss + beta * regularizers)

#Training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# predictions and accuracy
correct_prediction = tf.equal(tf.argmax(out_fin,1), tf.argmax(y_,1))
accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#run session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    sess.run(train_step, feed_dict = {x:xtrain, y_: ytrain})

print('The training accuracy is %s %%'% (sess.run(accuracy, feed_dict = {x:xtrain, y_: ytrain})))
print('The validationn aining accuracy is %s %%'% (sess.run(accuracy, feed_dict = {x:xval, y_: yval})))
# predictions on test data
