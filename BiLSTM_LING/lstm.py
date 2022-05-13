
import argparse
import io
from utils import *
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Input, Dropout

from tensorflow.keras import Model
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from nltk.corpus import stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words('english'))
from sklearn.metrics import confusion_matrix, classification_report

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-T','--train', type=str, help='Train file', required=True)
args.add_argument('-t','--test', type=str, help='Test or Dev file', required=True)
args.add_argument('-m','--emb_file', type=str, help='Embedding file', required=True)
args = args.parse_args()

EMBEDDING_FILE = args.emb_file
#"glove/glove.6B.200d.txt"
TRAIN_FILE = args.train
#"snli_1.0_train.txt"
TEST_FILE = args.test
#"snli_1.0_test.txt"

# Hyperparameters
VOCAB_SIZE = 20000
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
validation_split = .2
dropout_rate = .3
lr = 0.002


def create_label_mapping():
    return {"neutral":0, "contradiction":1, "entailment": 2}

def lower_case(sentence):
    return sentence.lower()

def create_embedding_matrix(tokenizer):
    embedding_index = {}
    f = io.open(EMBEDDING_FILE, encoding="utf8")
    lines = f.readlines()
    for i, line in enumerate(lines):
        values = line.strip().split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        if i>=50000:
            embedding_index["<OOV>"] = np.asarray(values[1:],dtype='float')
            break
        embedding_index[word] = vec

    f.close()
    vocab_size = len(embedding_index)
    embedding_dim = 200

    # create a weight matrix to initialize the embeddings layer
    embedding_matrix = np.zeros((vocab_size,embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_data():
    train_df = pd.read_csv(TRAIN_FILE, sep="\t", usecols=["sentence1", "sentence2", "gold_label"],
                           dtype={"sentence1": str, "sentence2": str, "gold_label": str})

    train_df = remove_null_values(train_df)
    #print("the number of data points after removing null values:", len(train_df))
    sent1 = train_df['sentence1'].apply(lower_case)
    sent2 = train_df['sentence2'].apply(lower_case)
    combined_sent = sent1.str.cat(sent2, sep= "\t")
    #print("combined sentences: ", combined_sent[1], type(combined_sent))


    # get extra linguisticfeatures
    pos_similarities = combined_sent.apply(get_pos_similarity)
    sent2_negated = sent2.apply(check_if_negated)
    blue_scores = combined_sent.apply(get_bleu_score)
    #print("new bleu scores ", blue_scores )
    sent_len_differences = np.array(compare_sentence_length(sent1, sent2))

    y = train_df["gold_label"].map(create_label_mapping())
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=oov_tok)
    tokenizer.fit_on_texts(sent1 + sent2)

    #turn each text into a sequence of integer IDs
    seq1 = tokenizer.texts_to_sequences(sent1)
    seq2 = tokenizer.texts_to_sequences(sent2)
    # pad sequences into the same length
    X1 = pad_sequences(seq1, maxlen=max_length,padding="post", truncating="post")
    X2 = pad_sequences(seq2, maxlen=max_length, padding="post", truncating="post")

    #get validation split
    train_num = int(len(X1) * (1-validation_split))
    X1_train = X1[:train_num]
    X2_train = X2[:train_num]
    blue_train = blue_scores[:train_num]
    sent2_negated_train = sent2_negated[:train_num]
    pos_train = pos_similarities[:train_num]
    sent_len_diff_train = sent_len_differences[:train_num]
    y_train = y[:train_num]
    X1_val = X1[train_num:]
    X2_val = X2[train_num:]
    blue_val = blue_scores[train_num:]
    sent2_negated_val = sent2_negated[train_num:]
    pos_val = pos_similarities[train_num:]
    sent_len_diff_val = sent_len_differences[train_num:]
    y_val = y[train_num:]

    train_features = np.column_stack((blue_train,sent_len_diff_train,pos_train,sent2_negated_train))
    val_features = np.column_stack((blue_val,sent_len_diff_val,pos_val,sent2_negated_val))

    return (X1_train, X2_train,train_features, y_train), (X1_val,X2_val,val_features,y_val), tokenizer

def get_test_data(fitted_tokenizer):
    train_df = pd.read_csv(TEST_FILE, sep="\t", usecols=["sentence1", "sentence2", "gold_label"],
                           dtype={"sentence1": str, "sentence2": str, "gold_label": str})

    train_df = remove_null_values(train_df[:10])
    #print("the number of data points after removing null values:", len(train_df))
    sent1 = train_df['sentence1'].apply(lower_case)
    sent2 = train_df['sentence2'].apply(lower_case)
    combined_sent = sent1.str.cat(sent2, sep= "\t")
    #print("combined sentences: ", combined_sent[1], type(combined_sent))

    # get extra features
    pos_similarities = combined_sent.apply(get_pos_similarity)
    #print("pos_similarities")
    #print(pos_similarities)
    sent2_negated = sent2.apply(check_if_negated)
    #print("sent2 negated")
    blue_scores = combined_sent.apply(get_bleu_score)
    sent_len_differences = np.array(compare_sentence_length(sent1, sent2))

    y = train_df["gold_label"].map(create_label_mapping())

    #turn each text into a sequence of integer IDs
    seq1 = fitted_tokenizer.texts_to_sequences(sent1)
    seq2 = fitted_tokenizer.texts_to_sequences(sent2)
    # pad sequences into the same length
    X1 = pad_sequences(seq1, maxlen=max_length,padding="post", truncating="post")
    X2 = pad_sequences(seq2, maxlen=max_length, padding="post", truncating="post")

    features = np.column_stack((blue_scores,sent_len_differences,pos_similarities,sent2_negated))

    return X1, X2,features, y

def create_model(label_num,embedding_matrix):
    #create a LSTM model

    #use the two sentences as inputs
    seq1_input = Input(shape=(max_length,),dtype='int32')
    seq2_input = Input(shape=(max_length,), dtype='int32')
    ling_input = Input(shape=(4,))

    #create embedding layer
    input_dim, output_dim = embedding_matrix.shape
    embed = Embedding(input_dim=input_dim, output_dim=output_dim,weights=[embedding_matrix],
                      input_length=max_length,trainable=False)
    emb1 = embed(seq1_input)
    emb2 = embed(seq2_input)

    #fed the embeddings into a BiLSTM layer separately
    seq1_lstm = Bidirectional(LSTM(256))(emb1)
    seq2_lstm = Bidirectional(LSTM(256))(emb2)

    #concatenate the LSTM layers into one
    x=concatenate([seq1_lstm,seq2_lstm,ling_input])
    x= Dropout(dropout_rate)(x)
    x= Dense(30)(x)
    output = Dense(label_num, activation="softmax")(x)
    model = Model(inputs=[seq1_input,seq2_input,ling_input],outputs=output)

    return model


def main():
    num_labels = len(create_label_mapping())
    (X1_train, X2_train, ling_train, y_train), (X1_valid, X2_valid, ling_val, y_valid), tokenizer = get_data()
    X1_test, X2_test, ling_test ,y_test = get_test_data(tokenizer)

    #get some features into the model
    Y_train = to_categorical(y_train, num_labels)
    Y_val = to_categorical(y_valid,num_labels)
    Y_test = to_categorical(y_test,num_labels)

    embed_matrix = create_embedding_matrix(tokenizer)
    model = create_model(label_num=num_labels,embedding_matrix=embed_matrix)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("compiled")
    earlystop = EarlyStopping(monitor="val_accuracy",min_delta=0.001,restore_best_weights=True,
                              patience=1)
    history = model.fit([X1_train, X2_train,ling_train], Y_train, validation_data=([X1_valid, X2_valid, ling_val],Y_val),
                        verbose=1,batch_size=80,callbacks=[earlystop],validation_freq=1,epochs=5)


    loss, acc = model.evaluate([X1_test, X2_test,ling_test],Y_test)
    print("Test loss ", loss, "Test Acc", acc)
    print("printing predictions")
    preds = model.predict(([X1_test, X2_test,ling_test]))
    #print("print out argmax for y")
    print(preds.argmax(axis=1))
    for pred in preds:
        print(pred)
    matrix = pd.DataFrame(confusion_matrix(Y_test.argmax(axis=1), preds.argmax(axis=1), labels = [0,1,2]),
                          index=['gold 0', 'gold 1', 'gold 2'],
                          columns= ['pred 0', 'pred 1', 'pred 2'])
    print(matrix)
    print("classification report")
    print(classification_report(Y_test.argmax(axis=1), preds.argmax(axis=1)))
    model.save("final_model.h5")
    model.summary()

main()
