##########################
#TO DO : ALL DONE
##########################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 20:56:36 2019

@author: c1888251
"""


import numpy as np
import nltk
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from scipy.sparse import csr_matrix as sc
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from time import time
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



############################################
# Defining Methods
############################################

#method name :Review_preprocesss
#parameters: Review as dataframe
#return: cleaned Review as DataFrame
#Description: the purpose of this method to clean a review to raw data by number of steps
    #steps:
    #  1  Lower case
    #  2  Remove frequent words  - (the words appear in both positive and negative reviews)
    #  3  Remove any word less than 3 letters
    #  4  Remove Stopwords 
    #  5  Remove Numbers and Punctuation 
    #  6  Remove whiteSpaces
    #  7  Lemmatization for noun and verb 

def Review_preproccess(df_tweet):
    #df_tweet['text'].dropna(inplace=True)#drop null values
    for i,tweet_lemma in df_tweet.iterrows():
      list_tweet=[]
      str_tweet=''
      splitting=df_tweet.at[i,'text'].split() # split every review to single word
      for token in splitting:
        token=token.lower()
        token=token.replace("'s", '')
        token=token.replace('br', '')
        #those words were hard-coded removed because they appear on both pos and neg (almost) equally
        token=token.replace('character', '')
        token=token.replace('show', '')
        token=token.replace('movie', '')
        token=token.replace('film', '')
        token=token.replace('would', '')
        token=token.replace('even', '')
        token=token.replace('could', '')
        token=token.replace('one', '')
        token=token.replace('make', '')
        token=token.replace('see', '')
        token=token.replace('also', '')
        token=token.replace('think', '')
        token=token.replace('the', '')
        token=token.replace('and', '')
        if (len(token) > 2 ):#remove any word less than 2 letters
         if token not in stop_words:
             #remove numbers , white specaes and breaklines by Regular Expression
            token=re.sub("([0-9]+)|([^0-9A-Za-z \t])|(<br\s*/><br\s*/>)|(\-)|(\/)|(\w+:\/\/\S+)"," ",token)#regulaer Expression
            token = token.strip()# remove white spaces in the beging of review.
            list_tweet.append(lemmatizer.lemmatize(lemmatizer.lemmatize(token,'v')))#lemmatizing each token, by reyrn it to its base
      str_tweet=' '.join(list_tweet)
      df_tweet.at[i,'text2']=str_tweet # save tokenized cleaned review as list of cleaned reviews
      df_tweet.at[i,'token']=df_tweet.at[i,'text2'].split() # keep tokenized cleaned review it (splitted)        
    return df_tweet
##End Review Preprocessor
   
#2nd method
#method name :get_vocabulary
#parameters: review as dataframe , number of features
#Return : list of most frequent vocabulary
#Description: create dictionary of vocabularies of the most frequent words int train set.
#              return it as list of vocab , the size of list determines by number of features.
#This code was adapted from session 2 posted by Dr Jose Camacho Collados oct-2019
#accessed Nov-2019
#https://learningcentral.cf.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_393342_1&content_id=_5178506_1 
def get_vocabulary(df, num_features):
    dict_word_frequency={}# create dictionary to save each word and its frequent number
    for review in df.token:
      sentence_tokens=review
      for word in sentence_tokens:
        if ( len(word) > 2):
            if word not in dict_word_frequency: dict_word_frequency[word]=1
            else: dict_word_frequency[word]+=1
    #Sort dictionary in descending order to get the highest word frequent numbers on top, then get the top (num_featuers)
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
    # Create dataframe to plot the most 50 frequent words (observing purpose)
    df = pd.DataFrame(columns=['word','freq'])
    i=0
    for word,frequency in sorted_list[:40]:
      df.loc[i,'word'] = word
      df.loc[i,'freq'] = frequency
      i+=1
    #ploting the most 40 frequent words in training set
#    df.plot.bar(x='word',title='Most Frequent words in Training set', figsize=(20,10))
#    plt.xlabel("words")
#    plt.ylabel("Most frequent words")
#    plt.show()
    #Retuern the most freq. words as list of vocab without frequency number.
    vocabulary=[]
    for word,frequency in sorted_list:
      vocabulary.append(word)    
    return vocabulary
##End get_vocabulary
    
#This code was taken from session 2 posted by Dr Jose Camacho Collados oct-2019
#accessed Nov-2019
#https://learningcentral.cf.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_393342_1&content_id=_5178506_1
#3rd method
#method name :get_vector_text
#parameters: list of most frequent vocabulary,review dataFrame
#Return : Extracted vector text
#Description: Match words in vocab list with similar words in review tuple ,
#             if its the case, count number of that word in the tuple to give it more weight. 
def get_vector_text(Vocab_dictionary,review_df):
   vector_text=np.zeros(len(Vocab_dictionary))
   for i, word in enumerate(Vocab_dictionary):
    if word in review_df:
      vector_text[i]=review_df.count(word)
   return vector_text
#End get_vector_text

#4th method
#method name:get_tf_idf
#parameters:review dataframe , number of featuers and stopword list 
#Return : matrix of wighted vocab by TF-DF 
#Description:this mthod to calculate term-frequency times inverse document-frequency wighted terms 
#            
def get_tf_idf(df,num_featuers,stop_words):
    tfidf = TfidfVectorizer(max_features=num_featuers, lowercase=True, analyzer='word', stop_words= stop_words,ngram_range=(2,4))
    tfidf_vocab = tfidf.fit_transform(df['text2'])
    return tfidf_vocab
#end get_tf_idf method
    

#5th method
#method name:get_Hashing
#parameters:review dataframe , number of features and stop-word list 
#Return : matrix of wighted vocab by Hashingvactorizer
#Description:this method to apply  hashing function to word frequency counts 
#
def get_Hashing(df,num_featuers):
    x_hashing=HashingVectorizer(n_features=num_featuers, alternate_sign=False)
    hashing_vocab = x_hashing.fit_transform(df['text2'])
    return hashing_vocab
#end get_Hashing method

#6th method
#method name:train_model
#parameters: 1-data_cleaned: review after cleaning
#            2-list of vocab: here its represnt the first featuer extraction techniuqe
#            3-num featuers: represent featuer numberes used in each featuer extraction technique(most frequent word, TF-IDF amd Hashvactorizer)
#
#Return : traind Machine learning model (classifier)
#Description: this method get train set ,and list of most frequent words in the train set number of features. 
#             we could divide this model to 3 main part first (Extract features from train set, do feature selection , train a model )
def train_model(data_cleaned,vocab,num_featuers):
#This code was adapted from session 2 posted by Dr Jose Camacho Collados Oct-2019
#accessed Nov-2019
#https://learningcentral.cf.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_393342_1&content_id=_5178506_1
    #Apply most frequent words technique
    #extracting  1st dimension of features(most frequent words , also splitting the features from target column ( both are stored as list)
    X_train=[]
    Y_train=[]  
    for i,review in data_cleaned.iterrows():
        vector_review=get_vector_text(vocab,data_cleaned.at[i,'token'])
        X_train.append(vector_review)
        Y_train.append(data_cleaned.at[i,'label'])
    #Convert them to arrays (NumPy libraries)   
    X_train_sentanalysis=np.asarray(X_train)
    Y_train_sentanalysis=np.asarray(Y_train)
    #End adapted code

    #extracting 2nd dimension of features(TF-IDF), then converting it to array using (Scipy library because the returned data type is csr.csr_matrix in Scipy lib )
    X_tfId=get_tf_idf(data_cleaned,num_featuers,stop_words)
    X_tfId=sc.toarray(X_tfId)
    #extracting 3rd dimension of features(HashingVectorizer), then converting it to array using (Scipy library) 
    X_hash= get_Hashing(data_cleaned,num_featuers)
    X_hash=sc.toarray(X_hash)
    
    #Concatenate all 3 dimensions to one matrix 
    X_tfId=np.concatenate((X_tfId,X_hash), axis=1)
    X_train_sentanalysis = np.concatenate((X_train_sentanalysis,X_tfId), axis=1)  
    
    #Define a pipeline contains Feature selection technique and the model and the model
    #Feature selection technique used is selectKbest with chi2 , and the value of k is set to get the half of concatenated features 
    #in first iteration of training/validate model :each feature generate 1000 column (3000 in total) so after feature selection will
    #will reduced to (1500 feature vector)  which are the most wighted feature.
    # the model is logisticRegression, it is a classifier its solver set to sag due to the size of data(large)
    #the motivation behind do them in one pipeline to minimise the steps of fitting and transforming selection the fit again with model,
    #also apply .predict with dev/test (in their stages) without needing to apply (fit_transform)sfeatuer_election separately then predict them.
    model_pipline = Pipeline(steps=[("dimension_reduction", SelectKBest(chi2, k=(int(num_featuers*.5)))),
    ("classifiers", LogisticRegression(solver='sag', max_iter=2000))])#edit the default value of max_iter(100) 
    model_pipline.fit(X_train_sentanalysis,Y_train_sentanalysis)    
    #return the trained model
    return model_pipline
#end train_model method
    



##############################################################################################################
################################################1-DATA LOADING AND PREPROCCESSING #############################
##############################################################################################################

#Get current directory to load data , so if directory is changed no need to retype each path separately
dirpath = os.getcwd()
print("Current directory is : " + dirpath)
foldername = os.path.basename(dirpath)
print("Folder name is : " + foldername)
#structure path for each data file (positive train file, negative train file, positive test file,
# negative test file, positive development file, and negative development path )
train_pos_path= dirpath+ '/IMDb/train/imdb_train_pos.txt'
train_neg_path= dirpath+ '/IMDb/train/imdb_train_neg.txt'

test_pos_path= dirpath+ '/IMDb/test/imdb_test_pos.txt'
test_neg_path= dirpath+ '/IMDb/test/imdb_test_neg.txt'

develop_pos_path= dirpath+ '/IMDb/dev/imdb_dev_pos.txt'
develop_neg_path= dirpath+ '/IMDb/dev/imdb_dev_neg.txt'

print("Data has been uploaded ")

#load data from each path as Pandas dataframe , text column represents review
train_pos=pd.read_csv(train_pos_path, sep = "\t", header=None, names=['text'])
train_neg=pd.read_csv(train_neg_path,  sep = "\t", header=None, names=['text'])

test_pos=pd.read_csv(test_pos_path, sep = "\t", header=None, names=['text'])
test_neg=pd.read_csv(test_neg_path, sep = "\t", header=None, names=['text'])

develop_pos=pd.read_csv(develop_pos_path, sep = "\t", header=None, names=['text'])
develop_neg=pd.read_csv(develop_neg_path, sep = "\t", header=None, names=['text'])

#Add column called 'label' for each of 6 dataframes. it represents the target
# specifying 1: for positive reviews
#           0: for negative reviews
train_pos['label']=1
train_neg['label']=0

test_pos['label']=1
test_neg['label']=0

develop_pos['label']=1
develop_neg['label']=0


##Merge train to a single set contains both positive and negative reviews
##shuffle them to randomize the order of positivity and negativity and reset the index
##do the same for test and developing set separately.

train=shuffle(pd.concat([train_pos,train_neg]))
train=train.reset_index(drop=True)

test=shuffle(pd.concat([test_pos,test_neg]))
test=test.reset_index(drop=True)

dev=shuffle(pd.concat([develop_pos,develop_neg]))
dev=dev.reset_index(drop=True)


#print(train['text'].isnull().sum())
#print(test['text'].isnull().sum())
#print(dev['text'].isnull().sum())

##########################################################################
### to do Give a reason here
#remove duplicates values
if any(train.duplicated(subset=['text'],keep=False)):
    j=train[train.duplicated(subset=['text'],keep=False) == True]
#    print('Triaing has dublicates:'+str(len(j)))
    train.drop_duplicates(subset=['text'], keep = 'first', inplace = True)
if any(test.duplicated(subset=['text'],keep=False)):
    m=test[test.duplicated(subset=['text'],keep=False) == True]
#    print('test has dublicates:'+str(len(m)))
    test.drop_duplicates(subset=['text'], keep = 'first', inplace = True)
if any(dev.duplicated(subset=['text'],keep=False)):
    n=dev[dev.duplicated(subset=['text'],keep=False) == True]
#    print('Development has dublicates:'+str(len(n)))
    test.drop_duplicates(subset=['text'], keep = 'first', inplace = True)



#Define all Global attributes needed in coding as lemmatizing, stop words ..etc
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
#print(str(stop_words))
print('pre-processing Data ...')
#for visualizing purposes, defining new columns: text2 to store list of cleaned review, 
#                                                and token to store token of cleaned data
train['text2']=''
train['token']=''
train_cleaned = Review_preproccess(train)

dev['text2']=''
dev['token']=''
dev_cleaned = Review_preproccess(dev)

test['text2']=''
test['token']=''
test_cleaned = Review_preproccess(test)
print('DONE')


##This code was adapted from DataCamp posted by Dr Duong Vu  8-11-2018
##accessed Dec-2019
##https://learningcentral.cf.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_393342_1&content_id=_5178506_1
#
##this code to generate wordcloud of most frequent words in positive train data , and negative train data , to give a view of data in each group
#need to import WordCloud library
#from wordcloud import WordCloud
#train_wordCloud = train_cleaned.groupby('lable')
#text_positive = " ".join(review for review in train_wordCloud ['text2'].get_group(1))
#text_negative = " ".join(review for review in train_wordCloud ['text2'].get_group(0))
#
##print(text)
#
#wordcloud_positive = WordCloud(max_font_size=75, max_words=50, background_color="white").generate(text_positive)
#wordcloud_negative = WordCloud(max_font_size=75, max_words=50, background_color="white").generate(text_negative)
#plt.figure()
#plt.title('Overview of Positive Train-set')
#plt.imshow(wordcloud_positive, interpolation="bilinear")
#plt.axis("off")
#plt.show()
#plt.figure()
#plt.title('Overview of Negative Train-set')
#plt.imshow(wordcloud_negative, interpolation="bilinear")
#plt.axis("off")
#plt.show()






##############################################################################################################
################################################2-TRAIN & VALIDATE MODEL ####################################
##############################################################################################################


# Now train a model with the different number of features, then validate the model on validation set.
# then pick best model with best number of features depends on the obtained accuracy
list_num_features=[1000,1200,1400]
best_accuracy_dev=0.0
best_num_features=0
best_vocabulary=[]
#Extract most frequent words with highest number of featuers. Inside the loop specify the size dependes on number of featuesrs on that iteraion,
#to prevent extraction many times.
print ("The model will train and validate on 3 diffrent number of featuers which are: "+str(list_num_features))
vocabulary=get_vocabulary(train_cleaned, list_num_features[-1])  

for num_features in list_num_features:
  print('Training model with '+str(num_features)+' features...')
  #send the traing data with modet frequent vocab and number of featuer train model method
  model=train_model(train_cleaned, vocabulary[:num_features], num_features)
  print('Done')
  #
  # Then, we transform  development set to vectors and make the prediction on this set to validate the trained model
  #
  print('Validating model with '+str(num_features)+' features...')
  X_dev_sentanalysis=[]
  Y_dev= []
  for i,review in dev_cleaned.iterrows():
    #extracting  1st dimension of features(most frequent words , also spliting the fetuers from target column ( both are stored as list)
    vector_instance=get_vector_text(vocabulary[:num_features],dev_cleaned.at[i,'token'])
    X_dev_sentanalysis.append(vector_instance)
    Y_dev.append(dev_cleaned.at[i,'label'])
  
  #convert previous list to arrays(NumPy librray) for prediction on the model
  X_dev_sentanalysis=np.asarray(X_dev_sentanalysis)
  Y_dev_gold=np.asarray(Y_dev)  
  
  #extracting 2nd dimenstion of featuers(TF-IDF), then converting it to array using (Scipy library)
  X_dev_TF1=get_tf_idf(dev_cleaned, num_features,stop_words)
  X_dev_TF=sc.toarray(X_dev_TF1)
 
  #extracting 3rd dimenstion of featuers(HashingVectorizer), then converting it to array using (Scipy library )
  X_dev_hash= get_Hashing(dev_cleaned,num_features)
  X_dev_hash=sc.toarray(X_dev_hash)
  
  #Concatenate all 3 dimensions to one matrix 
  X_dev_TF=np.concatenate((X_dev_TF,X_dev_hash), axis=1)
  X_dev = np.concatenate((X_dev_sentanalysis,X_dev_TF), axis=1)

  #######

  #Predicting featuers of Dev set, then calculating the performance measuers
  Y_dev_predictions=model.predict(X_dev)
  print('Done')
#  print('\n')
#This code was adapted from session 3 posted by Dr Jose Camacho Collados Oct-2019
#accessed Nov-2019
#https://learningcentral.cf.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_393342_1&content_id=_5178506_1
  #Applay most frequent words techniuqe
  #perofrmance results of the classifier(model)
  print('Confusion matrix of Development set: ')
  print (str(confusion_matrix(Y_dev_gold, Y_dev_predictions)))
  accuracy_dev=accuracy_score(Y_dev_gold, Y_dev_predictions)
  precision_dev=precision_score(Y_dev_gold, Y_dev_predictions,average='macro')
  recall_dev=recall_score(Y_dev_gold, Y_dev_predictions,average='macro')
  f1_dev=f1_score(Y_dev_gold, Y_dev_predictions,average='macro')
  print("Accuracy: "+str(round(accuracy_dev,4))+",  Precision: "+str(round(precision_dev,4))+"\nRecall: "+str(round(recall_dev,4)) + ",  f1:"+str(round(f1_dev,4))+"\n")
  if accuracy_dev>=best_accuracy_dev:
    best_accuracy_dev=accuracy_dev
    best_num_features=num_features
    best_vocabulary=vocabulary[:num_features]
    best_model=model
#end of code

print('\nBest number of features is : '+str(best_num_features))
  



##############################################################################################################
################################################3-TEST MODEL #################################################
##############################################################################################################


#Testing the model by do the same proccess with development proccess
print('Testing Model')
X_test=[]
Y_test=[]
Y_test_predicted=[]
for i,review in test_cleaned.iterrows():
    vector_review_test=get_vector_text(best_vocabulary,test_cleaned.at[i,'token'])
    X_test.append(vector_review_test)
    Y_test.append(test_cleaned.at[i,'label'])
   
#1st dimension of Feature Extraction
X_test_sentanalysis=np.asarray(X_test)
Y_test_Gold=np.asarray(Y_test)#Gold
#2nd dimension of Feature Extraction
X_test_TF=get_tf_idf(test_cleaned, best_num_features,stop_words)
X_test_TF=sc.toarray(X_test_TF)
#3rd dimension of Feature Extraction
X_test_hash= get_Hashing(test_cleaned,num_features)
X_test_hash=sc.toarray(X_test_hash)
#Concatenate all features
X_test_TF=np.concatenate((X_test_TF,X_test_hash), axis=1)
X_test_sentanalysis = np.concatenate((X_test_sentanalysis,X_test_TF), axis=1)

#predicting on test data
Y_predicted=best_model.predict(X_test_sentanalysis)

#Compute the confusion matrix and evalute measeurs(accuracy ,precision, recall and f1) for testing prediction
print('Confusion matrix of test set')
print(confusion_matrix(Y_test_Gold, Y_predicted))
accuracy=accuracy_score(Y_test_Gold, Y_predicted)
precision=precision_score(Y_test_Gold, Y_predicted,average='macro')
recall=recall_score(Y_test_Gold, Y_predicted,average='macro')
f1=f1_score(Y_test_Gold, Y_predicted,average='macro')
print('Measure of Metrics in Test set')
print("\n Accuracy: "+str(round(accuracy,4))+
",  Precision: "+str(round(precision,4))+"\nRecall: "+str(round(recall,4)) + ",  f1:"+str(round(f1,4)))
   

#This code was adapted from StackAbuse posted by Guest Contributor 25-2-2019
#accessed Nov-2019
#https://stackabuse.com/understanding-roc-curves-with-python/
#Drawing ROC curve methods
#fpr, tpr,_=roc_curve(best_model.predict(X_test_sentanalysis),Y_test_Gold,drop_intermediate=False)
#plt.figure()
#plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve')
#plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
#plt.xlabel('False Positive ')
#plt.ylabel('True Positive ')
#plt.title('ROC curve')
#plt.show()
print('\nDone with IMDB ML model')
##exit()
