import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from official.nlp import bert
import official.nlp.bert.tokenization
import os
from keras import backend as K
import random
from sklearn.metrics import classification_report

def sentence_encode(sentence, tokenizer):

    tokens = list(tokenizer.tokenize(str(sentence)))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(train_data, tokenizer):

    sentence_1 = tf.ragged.constant([sentence_encode(sentence, tokenizer) for sentence in np.array(train_data['sentence1'])])
    sentence_2 = tf.ragged.constant([sentence_encode(sentence, tokenizer) for sentence in np.array(train_data['sentence2'])])

    token = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence_1.shape[0]

    input_word_ids = tf.concat([token, sentence_1, sentence_2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    segment_token = tf.zeros_like(token)
    segment_s1 = tf.zeros_like(sentence_1)
    segment_s2 = tf.ones_like(sentence_2)
    input_type_ids = tf.concat([segment_token, segment_s1, segment_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    return inputs

def build_model():

    sequence_length = 125

    input_word_ids = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.layers.Input(shape=(sequence_length,), dtype=tf.int32, name="input_type_ids")

    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)

    pooled_output, _ = bert_layer([input_word_ids, input_mask, input_type_ids])

    X = tf.keras.layers.Dense(units=64, activation='relu')(pooled_output)
    X = tf.keras.layers.Dropout(0.2)(X)

    output_class = tf.keras.layers.Dense(units=3, activation='softmax')(X)

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output_class)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision_result = precision(y_true, y_pred)
    recall_result = recall(y_true, y_pred)
    return 2*((precision_result*recall_result)/(precision_result+recall_result+K.epsilon()))

def accuracy(model, input_bert, data):

  predictions =[np.argmax(i) for i in model.predict(input_bert)]
  acc = np.mean(predictions == data.label_ids.values)
  print(acc)


def main():

    train = pd.read_csv("snli_1.0/snli_1.0_train.txt", sep="\t")
    dev = pd.read_csv("snli_1.0/snli_1.0_dev.txt", sep="\t")
    test = pd.read_csv("snli_1.0/snli_1.0_test.txt", sep="\t")

    train_data = train[['gold_label', 'sentence1', 'sentence2']]
    dev_data = dev[['gold_label', 'sentence1', 'sentence2']]
    test_data = test[['gold_label', 'sentence1', 'sentence2']]

    train_data = train_data[train_data['gold_label'] != '-']
    dev_data = dev_data[dev_data['gold_label'] != '-']
    test_data = test_data[test_data['gold_label'] != '-']

    train_data = train_data[train_data['sentence1'].notna()]
    train_data = train_data[train_data['sentence2'].notna()]

    dev_data = dev_data[dev_data['sentence1'].notna()]
    dev_data = dev_data[dev_data['sentence2'].notna()]

    test_data = test_data[test_data['sentence1'].notna()]
    test_data = test_data[test_data['sentence2'].notna()]

    mask = np.random.rand(len(test_data)) < 0.5
    dev_data_1 = test_data[mask]
    test_data_1 = test_data[~mask]

    pretrained_bert_folder = "uncased_L-12_H-768_A-12"
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(pretrained_bert_folder, "vocab.txt"),
                                                do_lower_case=True)

    model = build_model()
    #print(model.summary())

    input_model = bert_encode(train_data, tokenizer)
    input_dev = bert_encode(dev_data_masked, tokenizer)
    input_test = bert_encode(test_data_masked, tokenizer)

    pd.options.mode.chained_assignment = None
    encoder = LabelEncoder()
    train_data['label_ids'] = encoder.fit_transform(train_data['gold_label'])
    dev_data_1['label_ids'] = encoder.transform(dev_data_1['gold_label'])
    test_data_1['label_ids'] = encoder.transform(test_data_1['gold_label'])


    model.fit(input_model, train_data.label_ids.values,
              epochs=2,
              batch_size=64,
              verbose=1)

    print(accuracy(model, input_dev, dev_data_1))
    print(accuracy(model, input_test, test_data_1))

    prediction = model.predict(input_test, batch_size=64, verbose=1)
    predictions = np.argmax(prediction, axis=1)

    print(classification_report(test_data_1.label_ids.values, predictions))

    predicted_labels = predictions
    actual_labels = test_data_1.label_ids.values

    # Bad examples
    counter = 0
    for i in range(len(actual_labels)):
    if actual_labels[i] != predicted_labels[i]:
        print("Index: ", i, "Actual: ", actual_labels[i], "Predicted: ",predicted_labels[i])
        counter = counter + 1
    print(counter)
    print("Sentence 1: ", test_data_1['sentence1'].iloc[0], "\nSentence 2: ", test_data_1['sentence2'].iloc[0])
    print("Sentence 1: ", test_data_1['sentence1'].iloc[4], "\nSentence 2: ", test_data_1['sentence2'].iloc[4])
    print("Sentence 1: ", test_data_1['sentence1'].iloc[9], "\nSentence 2: ", test_data_1['sentence2'].iloc[9])


    # Good examples
    counter = 0
    for i in range(len(actual_labels)):
    if actual_labels[i] == predicted_labels[i]:
        print("Index: ", i, "Actual: ", actual_labels[i], "Predicted: ",predicted_labels[i])
        counter = counter + 1
    print(counter)
    print("Sentence 1: ", test_data_1['sentence1'].iloc[1], "\nSentence 2: ", test_data_1['sentence2'].iloc[1])
    print("Sentence 1: ", test_data_1['sentence1'].iloc[2], "\nSentence 2: ", test_data_1['sentence2'].iloc[2])
    print("Sentence 1: ", test_data_1['sentence1'].iloc[3], "\nSentence 2: ", test_data_1['sentence2'].iloc[3])

    
if __name__ == '__main__':
    main()