# ENLP_NLI

Please go to https://nlp.stanford.edu/projects/snli/ to download the relevant datasets. 

## statistical.modelling.py
This file contains helper funnctions to extract linguistic features for our first TF-IDF+LSA+SVM baseline model.

## TF-IDF+LSA+SVM baseline

To run training for this model, in the svm folder, run 'python3 statistical_modeling.py'

## BiLSTM_LING
A folder includes `utils.py` and `lstm.py`. utils.py includes helper functions for pre-processing and feature extraction. 

`lstm.py` takes in 3 argument: training data file, test data file and embedding file. Example usage below:
`python3 lstm.py --emb_file <your embedding file> --train <you train file> --test <your test file>`

## BiLSTM with BERT embeddings
To run training for this model, in the bilistm_bert_emd folder, run 'python3 train.py'

That scripts creates dataloaders and initiales a training loop to train the model and save it after every epoch.

## BiLSTM with InferSent
To run the model, in BiLSMT_InferSent folder, run NLI.py.

1. Make sure NLTK tokenizer is installed
  ```python
  import nltk
  nltk.download('punkt')
  ```
2. Download our InferSent models trained with GloVe
  ```
  mkdir encoder
  curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
  ```
3. Download SNLI dataset

## Fine-tuning BERT

To run this model, you need the follwoing:

1. Download the dataset.

2. Install offical package, https://github.com/tensorflow/models/tree/master/official. Using this command, ```!pip install tf-models-official```

3. Download BERT base pretrained model (uncased), https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4

4. Google Colab file was run using Colab Pro. (GPU preferred) 
