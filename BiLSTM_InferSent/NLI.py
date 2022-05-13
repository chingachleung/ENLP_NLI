import nltk
import sys
from InferSent.models import InferSent
import torch
import numpy as np
from keras.layers import Bidirectional, LSTM, Dense, Input, Concatenate
from keras.models import Model
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report


model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

use_cuda = False
model = model.cuda() if use_cuda else model

W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

model.build_vocab_k_words(K=100000)


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
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000)
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

    # experiment with different smoothing functions
    smoothie = SmoothingFunction().method4
    score_list = []
    for i, item in enumerate(sents1):
        # lowercase everything
        ref = sents1[i].lower().split()
        hyp = sents2[i].lower().split()
        # the weights determine the weighted geometric mean score from n-grams
        score = sentence_bleu([ref], hyp, weights=(0.75, 0.25, 0, 0),
                              smoothing_function=smoothie)  # we use unigram and bigrams
        score_list.append(score)
    return score_list


def get_pos_and_lemma(sents):
    """
    :param a list of sentences
    :return a list of pos, and a list of lemmas
    """
    import spacy
    nlp = spacy.load(
        "en_core_web_sm")  # make sure to download before download by python -m spacy download en_core_web_sm
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


def svm_model(X_train, y_train, X_test, y_test):
    # svm can handle a great number of features, so it is application if we use TFIDF/sparse data , but not optimal with lots of data points
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC()  # default is RBF, degree is 3
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    return score


def baggingclassifier(X_train, y_train, X_test, y_test):
    # if not using spare data as features, decision trees might work better
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=150,  # 150 trees
        max_samples=int(np.ceil(0.7 * X_train.shape[0])),  # 70% of data per tree
        bootstrap=True, n_jobs=3, random_state=42)
    bag_clf.fit(X_train, y_train)
    pred = bag_clf.predict(X_test)
    score = accuracy_score(y_test, pred)
    return score


train_data = pd.read_csv("snli_1.0/snli_1.0_train.txt", sep="\t")
test_data = pd.read_csv("snli_1.0/snli_1.0_test.txt", sep="\t")

train_data = remove_null_values(train_data)
test_data = remove_null_values(test_data)
train_labels = train_data["gold_label"]
test_labels = test_data["gold_label"]

# example usage of the TFIDF function
# fit tfidf with sentences in the training data
train_data['combined_sentences'] = train_data['sentence1'] + " " + train_data['sentence2']

# train_data=train_data.sample(frac=0.0001,replace=False)
# test_data=test_data.sample(frac=0.1,replace=False)





temp_df = pd.get_dummies(train_data, columns=['gold_label'])

labels = ['gold_label_contradiction', 'gold_label_entailment', 'gold_label_neutral']
y = np.array(temp_df[labels]).tolist()
y = np.array(y)


embeddings1 = model.encode(train_data['sentence1'], bsize=128, tokenize=False, verbose=True).reshape((len(train_data), 2, 2048))
embeddings2 = model.encode(train_data['sentence2'], bsize=128, tokenize=False, verbose=True).reshape((len(train_data), 2, 2048))
print(train_data.shape)
input1 = Input(shape=(2,2048))
input2 = Input(shape=(2,2048))
conc = Concatenate(axis=2)([input1, input2])
lstm = Bidirectional(LSTM(units=128, return_sequences=False))
rep = lstm(conc)
output = Dense(3, activation='softmax', name='class')(rep)
c_model = Model([input1, input2], output)

from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed

c_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c_model.fit([embeddings1, embeddings2], y, epochs=3, shuffle=True, batch_size=64)
try:
    c_model.save(filepath='./model/model1/model1')
except:
    pass
t_embeddings1 = model.encode(test_data['sentence1'], bsize=128, tokenize=False, verbose=True).reshape((len(test_data), 2, 2048))
t_embeddings2 = model.encode(test_data['sentence2'], bsize=128, tokenize=False, verbose=True).reshape((len(test_data), 2, 2048))
predictions = c_model.predict([t_embeddings1, t_embeddings2])
y_pred = np.argmax(predictions, axis=1)
a = {'contradiction': 0, 'entailment': 1, 'neutral': 2}


print(classification_report(np.array(test_data['gold_label']).tolist(),y_pred,target_names=['contradiction','entailment','neutral']))

