import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

import nltk
from nltk.corpus import brown
from nltk import pos_tag
from nltk import pos_tag_sents
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')


def remove_null_values(data):
  """
  removes rows when premises/hypotheses or gold labels are null
  :params a pandas dataframe
  :returns a dataframe with rows removed
  """
  # remove rows with null premises
  data = data[data['sentence1'].notna()]
  # remove rows with null hypotheses
  data = data[data['sentence2'].notna()]
  # remove rows with null gold labels
  data = data[data['gold_label'] != "-"]

  return data

def extract_tfidf(train_sentences):
  """
  :param a list of sentences in the training dataset
  :return a fitted tfidf_vectorizer, which can be used to transform training/test data
  """

  # you might want to play around with the max_features  -  it is taking a long time to run.
  tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features = 1000)
  tfidf_vect.fit(train_sentences)

  return tfidf_vect

def get_cosine_similarity(sents1, sents2):
  """
  :param 2 lists of sentences that are in numerical representatons, such as TFIDF/word-embeddings
  :return a list of cosine similarities between sents1 and sents2
  """
  cosine_similarity_list = []
  for i, item in enumerate(sents1):
      similarity = cosine_similarity(sents1[i].reshape(1, -1), sents2[i].reshape(1, -1))
      cosine_similarity_list.append(similarity)
  cosine_similarity_list = np.concatenate(cosine_similarity_list).ravel().tolist()
  return cosine_similarity_list

def get_bleu_score(sents1, sents2):
  """
  bleu score calculates the number of n-gram overlaps between references and hypotheses
  :params two lists of sentences
  :return  a list of bleu scores between the sentences
  """

  #experiment with different smoothing functions
  smoothie = SmoothingFunction().method4
  score_list = []
  for i, item in enumerate(sents1):
    #lowercase everything
    ref = sents1[i].lower().split()
    hyp = sents2[i].lower().split()
    # the weights determine the weighted geometric mean score from n-grams
    score = sentence_bleu([ref], hyp, weights=(0.75,0.25,0,0), smoothing_function=smoothie) # we use unigram and bigrams
    score_list.append(score)
  return score_list

def get_pos_and_lemma(sents):
  """
  :param a list of sentences
  :return a list of pos, and a list of lemmas
  """

  nlp = spacy.load("en_core_web_sm") # make sure to download before download by python -m spacy download en_core_web_sm
  pos_list = []
  lemma_list = []
  for sent in sents:
    doc = nlp(sent)
    curr_pos = []
    curr_lemma = []
    for token in doc:
      curr_pos.append(token.pos_)
      curr_lemma.append(token.lemma_)
    pos_list.append(curr_pos)
    lemma_list.append(curr_lemma)
  return pos_list, lemma_list

def get_sent_embeddings(sents):
  """
  take a list of sentences and convert it into an arrray of embeddings - 389 dimensions
  """
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
  embeddings = model.encode(sents)
  return embeddings

def svm_model(X_train, y_train, X_test,y_test):
  #svm can handle a great number of features, so it is application if we use TFIDF/sparse data , but not optimal with lots of data points

  model = SVC(max_iter=2) # default is RBF, degree is 3
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  score = accuracy_score(y_test, pred)
  return score

def baggingclassifier(X_train, y_train, X_test,y_test):
  #if not using spare data as features, decision trees might work better

  bag_clf = BaggingClassifier(
      DecisionTreeClassifier(), n_estimators=150, # 150 trees
      max_samples=int(np.ceil(0.7*X_train.shape[0])), #70% of data per tree
      bootstrap=True,n_jobs=3, random_state=42)
  bag_clf.fit(X_train, y_train)
  pred = bag_clf.predict(X_test)
  score = accuracy_score(y_test,pred)
  return score


def LSA_SVM(x_train_sent1_tfidf,x_train_sent2_tfidf,x_test_sent1_tfidf,x_test_sent2_tfidf,train_labels,test_labels):
  print(x_train_sent1_tfidf.shape)
  print(x_train_sent2_tfidf.shape)
  print(x_test_sent1_tfidf.shape)
  print(x_test_sent2_tfidf.shape)

  print("LSA")
  lsa_sent1 = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
  lsa_sent1.fit(x_train_sent1_tfidf)
  lsa_sent1_train = lsa_sent1.transform(x_train_sent1_tfidf)
  lsa_sent1_test = lsa_sent1.transform(x_test_sent1_tfidf)

  
  lsa_sent2 = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
  lsa_sent2.fit(x_train_sent2_tfidf)
  lsa_sent2_train = lsa_sent2.transform(x_train_sent2_tfidf)
  lsa_sent2_test = lsa_sent2.transform(x_test_sent2_tfidf)


  X_train = np.hstack((lsa_sent1_train,lsa_sent2_train))
  X_test = np.hstack((lsa_sent1_test,lsa_sent2_test))
  
  print("SVM")
  model = LinearSVC(C=0.8,max_iter=150) 
  model.fit(X_train, train_labels)
  pred = model.predict(X_test)
  score = accuracy_score(test_labels, pred)
  print("Dev Acc:",score)

def LSA_RandomForest(x_train_sent1_tfidf,x_train_sent2_tfidf,x_test_sent1_tfidf,x_test_sent2_tfidf,train_labels,test_labels):
  print(x_train_sent1_tfidf.shape)
  print(x_train_sent2_tfidf.shape)
  print(x_test_sent1_tfidf.shape)
  print(x_test_sent2_tfidf.shape)

  print("LSA")
  lsa_sent1 = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
  lsa_sent1.fit(x_train_sent1_tfidf)
  lsa_sent1_train = lsa_sent1.transform(x_train_sent1_tfidf)
  lsa_sent1_test = lsa_sent1.transform(x_test_sent1_tfidf)

  
  lsa_sent2 = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
  lsa_sent2.fit(x_train_sent2_tfidf)
  lsa_sent2_train = lsa_sent2.transform(x_train_sent2_tfidf)
  lsa_sent2_test = lsa_sent2.transform(x_test_sent2_tfidf)


  X_train = np.hstack((lsa_sent1_train,lsa_sent2_train))
  X_test = np.hstack((lsa_sent1_test,lsa_sent2_test))
  
  print("Random Forest")
  model = RandomForestClassifier(n_estimators=50,max_depth=7) 
  model.fit(X_train, train_labels)
  pred = model.predict(X_test)
  score = accuracy_score(test_labels, pred)
  print("Dev Acc:",score)

def get_pos_str(pos_sent):
  pos_str = ''
  for i in range(len(pos_sent)):
    curr = pos_sent[i]
    curr_word = curr[0]
    curr_pos = curr[1]
  
    pos_str = pos_str +' '+ curr_pos
    #bi_gram.append(curr_pos+'_'+nxt_pos)
  return pos_str

def POS_SVM(train_data, test_data, x_train_sent1_tfidf,x_train_sent2_tfidf,x_test_sent1_tfidf,x_test_sent2_tfidf):

  print("POS")
  print(train_data['sentence1'].values[0])
  sents1 = [word_tokenize(t) for t in train_data['sentence1'].values]
  sents2 = [word_tokenize(t) for t in train_data['sentence2'].values]

  sents1_test = [word_tokenize(t) for t in test_data['sentence1'].values]
  sents2_test = [word_tokenize(t) for t in test_data['sentence2'].values]

  temp_pd = pd.DataFrame()
  temp_pd['pos1'] = sents1
  temp_pd['pos2'] = sents2

  temp_pd2 = pd.DataFrame()
  temp_pd2['pos1_test'] = sents1_test
  temp_pd2['pos2_test'] = sents2_test
  print(temp_pd['pos1'])

  pos_train_sent1 = temp_pd['pos1'].apply(pos_tag)
  pos_train_sent2 = temp_pd['pos2'].apply(pos_tag)
  pos_test_sent1 = temp_pd2['pos1_test'].apply(pos_tag)
  pos_test_sent2 = temp_pd2['pos2_test'].apply(pos_tag)


  pos_train_sent1 = pos_train_sent1.apply(get_pos_str)
  pos_train_sent2 = pos_train_sent2.apply(get_pos_str)

  pos_test_sent1 = pos_test_sent1.apply(get_pos_str)
  pos_test_sent2 = pos_test_sent2.apply(get_pos_str)
  print(pos_train_sent1)

  pos_sent1_tf = TfidfVectorizer(ngram_range=(1,1))
  pos_sent1_tf.fit(pos_train_sent1)


  pos_sent2_tf = TfidfVectorizer(ngram_range=(1,1))
  pos_sent2_tf.fit(pos_train_sent2)

  pos_sent1_train = pos_sent1_tf.transform(pos_train_sent1)
  pos_sent2_train = pos_sent2_tf.transform(pos_train_sent2)
  pos_sent1_test = pos_sent1_tf.transform(pos_test_sent1)
  pos_sent2_test = pos_sent2_tf.transform(pos_test_sent2)

  print(pos_sent1_train.shape)
  print(pos_sent1_test.shape)

  #temp = x_train_sent1_tfidf + x_train_sent1_tfidf
  x_train_sent1_tfidf_pos = hstack((x_train_sent1_tfidf, pos_sent1_train))
  x_train_sent2_tfidf_pos = hstack((x_train_sent2_tfidf, pos_sent2_train))
  x_test_sent1_tfidf_pos = hstack((x_test_sent1_tfidf, pos_sent1_test))
  x_test_sent2_tfidf_pos = hstack((x_test_sent2_tfidf, pos_sent2_test))
  print(x_train_sent1_tfidf_pos.shape)

  train_labels = train_data["gold_label"]
  test_labels = test_data["gold_label"]
  
  LSA_SVM(x_train_sent1_tfidf_pos,x_train_sent2_tfidf_pos,x_test_sent1_tfidf_pos,x_test_sent2_tfidf_pos,train_labels,test_labels)

  #lsa_sent1 = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
  #lsa_sent1.fit(temp)


  '''
  print("get unique bigrams")
  all_bigrams = []
  for i in range(len(pos_train_sent1.values)):
    #list(np.unique(pos_train_sent1[i]))
    all_bigrams.extend(np.unique(pos_train_sent1[i]))
  
  all_bigrams = np.unique(all_bigrams)
  #get pos n-grams
  print(all_bigrams)
  print(len(all_bigrams))
  '''

def main():
  """
  just a simple starter code for you to start with - I play with 2 types of features,
  and demonstrate how they can be used in SVM and bagging classifiers. Feel free to play around with the features
  """
  #load the data:
  print("loading data")
  train_data = pd.read_csv("snli_1.0/snli_1.0_train.txt", sep="\t")
  test_data = pd.read_csv("snli_1.0/snli_1.0_dev.txt", sep="\t")

  train_data = remove_null_values(train_data)
  test_data = remove_null_values(test_data)
  train_labels = train_data["gold_label"]
  test_labels = test_data["gold_label"]
  
  #example usage of the TFIDF function
  # fit tfidf with sentences in the training data
  
  print("tf-idf")
  train_data['combined_sentences'] = train_data['sentence1'] + " " + train_data['sentence2']
  tfidf_vect = extract_tfidf(train_data['combined_sentences'])

  
  #get tfidf for sentence1 and sentence 2
  x_train_sent1_tfidf = tfidf_vect.transform(train_data['sentence1'])
  x_train_sent2_tfidf = tfidf_vect.transform(train_data['sentence2'])
  x_test_sent1_tfidf = tfidf_vect.transform(test_data['sentence1'])
  x_test_sent2_tfidf = tfidf_vect.transform(test_data['sentence2'])
  
  LSA_SVM(x_train_sent1_tfidf,x_train_sent2_tfidf,x_test_sent1_tfidf,x_test_sent2_tfidf,train_labels,test_labels)
  LSA_RandomForest(x_train_sent1_tfidf,x_train_sent2_tfidf,x_test_sent1_tfidf,x_test_sent2_tfidf,train_labels,test_labels)
  #POS_SVM(train_data, test_data, x_train_sent1_tfidf,x_train_sent2_tfidf,x_test_sent1_tfidf,x_test_sent2_tfidf)

  

if __name__ == '__main__':
  main()