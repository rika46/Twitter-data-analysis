#import packages
import requests  
import re  
import pandas as pd  

import numpy as np   
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
#install and import tweepy package, An easy-to-use Python library for accessing the Twitter API.
import tweepy as tw
#Authentication is handled by the tweepy.AuthHandler class
from tweepy import OAuthHandler

from nltk.corpus import stopwords
from nltk import trigrams
import string
import nltk
from nltk.stem.porter import *
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
# Visualizing the topics as modeled by LDA
import matplotlib.pyplot as plt

lem_df1 = pd.read_csv('/Users/rika/Documents/TM/dense_csv.csv',index_col=False)

lda_df = pd.read_csv("/Users/rika/Documents/TM/clust_cv.csv", index_col=False) 

mycv1 = CountVectorizer(input = 'content', stop_words = 'english', max_features=5000)
mymat = mycv1.fit_transform(lem_df1['Headline'])
mycols = mycv1.get_feature_names()
mymat = mymat.toarray()

################ LDA ################


# We have 5 different cuisines in our dataset and therefore we will expect LDA to produce 5 topics
num_topics = 4

# Input data frame for LDA
lda_input_df = lda_df
#lda_input_df = lda_input_df.drop(columns=['Label'])

# Instantiate the LDA model with 100 iterations and 5 topics
lda_model_DH = LatentDirichletAllocation(n_components=num_topics,
                                         max_iter=100, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(lda_input_df)
#print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)



# Get the matrix of values which can then be used to obtain top 15 words for each topic
word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()
num_top_words = 15
vocab = mycv1.get_feature_names()
vocab_array = np.asarray(vocab)

# Plot the top 15 words under each topic using matplotlib
fontsize_base = 15
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 2)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)
plt.tight_layout()
plt.show()



# Another Viz for LDA
# implement a print function
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
   

####### call the function above with our model and CountV
print_topics(lda_model_DH, mycv1, 15)

## Print LDA using print function from above
########## Other Notes ####################
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
import pyLDAvis.gensim 
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model_DH, lda_df, mycv1, mds='tsne')
pyLDAvis.show(panel)
panel = pyLDAvis.gensim.prepare(lda_model_DH, mymat, mycv1, mds='tsne')
pyLDAvis.show(panel)


dtm = np.matrix(lda_df)
panel = pyLDAvis.sklearn.prepare(lda_model_DH, dtm, mycv1)
