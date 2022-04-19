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
  from sklearn.feature_extraction.text import TfidfVectorizer
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
  import spacy
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
  from sklearn.svm import SVC
  from sklearn.metrics import accuracy_score
  model = SVC() # default is RBF, degree is 3
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  score = accuracy_score(y_test, pred)
  return score

def baggingclassifier(X_train, y_train, X_test,y_test):
  #if not using spare data as features, decision trees might work better
  from sklearn.ensemble import BaggingClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import accuracy_score
  bag_clf = BaggingClassifier(
      DecisionTreeClassifier(), n_estimators=150, # 150 trees
      max_samples=int(np.ceil(0.7*X_train.shape[0])), #70% of data per tree
      bootstrap=True,n_jobs=3, random_state=42)
  bag_clf.fit(X_train, y_train)
  pred = bag_clf.predict(X_test)
  score = accuracy_score(y_test,pred)
  return score

def main():

    """
    just a simple starter code for you to start with - I play with 2 types of features,
    and demonstrate how they can be used in SVM and bagging classifiers. Feel free to play around with the features
    """
    #load the data:
    train_data = pd.read_csv("snli_1.0_train.txt", sep="\t")
    test_data = pd.read_csv("snli_1.0_test.txt", sep="\t")

    train_data = remove_null_values(train_data)
    test_data = remove_null_values(test_data)
    train_labels = train_data["gold_label"]
    test_labels = test_data["gold_label"]

    #example usage of the TFIDF function
    # fit tfidf with sentences in the training data
    train_data['combined_sentences'] = train_data['sentence1'] + " " + train_data['sentence2']
    tfidf_vect = extract_tfidf(train_data['combined_sentences'])

    #get tfidf for sentence1 and sentence 2
    x_train_sent1_tfidf = tfidf_vect.transform(train_data['sentence1'])
    x_train_sent2_tfidf = tfidf_vect.transform(train_data['sentence2'])
    x_test_sent1_tfidf = tfidf_vect.transform(test_data['sentence1'])
    x_test_sent2_tfidf = tfidf_vect.transform(test_data['sentence2'])

    #use tfidf to calculate cosine similarities
    train_tfidf_cosine_similarities = get_cosine_similarity(x_train_sent1_tfidf, x_train_sent2_tfidf)
    test_tfidf_cosine_similarities = get_cosine_similarity(x_test_sent1_tfidf, x_test_sent2_tfidf)

    # use sent embeddings to get cosine similarities
    train_embedding_cosine_similarities = get_cosine_similarity(get_sent_embeddings(train_data["sentence1"]),
                                                                get_sent_embeddings(train_data["sentence2"]))
    test_embedding_cosine_similarities = get_cosine_similarity(get_sent_embeddings(test_data["sentence1"]),
                                                                get_sent_embeddings(test_data["sentence2"]))

    test_data['embedding_cosine_similarity'] = test_embedding_cosine_similarities
    train_data['embedding_cosine_similarity'] = train_embedding_cosine_similarities
    test_data['tfidf_cosine_similarity'] = test_tfidf_cosine_similarities
    train_data['tfidf_cosine_similarity'] = train_tfidf_cosine_similarities

    features = ['tfidf_cosine_similarity', 'embedding_cosine_similarity']
    svm_score = svm_model(train_data[features], train_labels, test_data[features], test_labels)
    bagging_score = baggingclassifier(train_data[features],train_labels, test_data[features], test_labels)

    print('the score from the svm model is: ', svm_score) # 0.289 with only 15 data points in toy set
    print('the score from the bagging model is: ', bagging_score) #0.5 with only 15 data points in toy set 

    #bleu_score = get_bleu_score(train_data['sentence1'], train_data['sentence2'])
    #print('bleu scores: ', bleu_score)

    return svm_score, bagging_score

main()
