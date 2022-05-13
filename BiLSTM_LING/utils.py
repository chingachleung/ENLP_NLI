


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words('english'))


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


def get_sentence_similarity(sent1, sent2):
    #take 2 sentences as arguments
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    similarities = []

    for s1, s2 in zip(sent1, sent2):
        e1 = model.encode(s1)
        e2 = model.encode(s2)
        similarity = cosine_similarity(e1.reshape(1,-1),e2.reshape(1,-1))
        similarities.append(similarity[0][0])

    return similarities

def get_sentence_similarity2(combined_sent):
    # received a combined sentece split by tab
    sent1, sent2 = combined_sent.strip.split("\t")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    e1 = model.encode(sent1)
    e2 = model.encode(sent2)
    similarity = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))
    return similarity

def get_word_similarity(w1,w2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    e1 = model.encode(w1)
    e2 = model.encode(w2)
    similarity = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))
    return similarity[0][0]

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
  return pos_list

def get_pos_similarity(combined_sent):
  """
  :param a sentence
  :return a list of pos, and a list of lemmas
  """
  #import spacy
  #nlp = spacy.load("en_core_web_sm") # make sure to download before download by python -m spacy download en_core_web_sm

  sent1, sent2 = combined_sent.strip().split("\t")
  doc1 = nlp(sent1)
  doc2 = nlp(sent2)
  pos1_list = []
  pos2_list = []
  #lemma_list= []
  for token in doc1:
    pos1_list.append(token.pos_)
    #lemma_list.append(token.lemma_)
  for token in doc2:
    pos2_list.append(token.pos_)

 #use bleu score to compare pos overlaps
  smoothie = SmoothingFunction().method1
  score = sentence_bleu([pos1_list], pos2_list, weights=(0.75, 0.25, 0, 0), smoothing_function=smoothie)
  return score

def check_if_similar_verb(sent1, sent2):
    #take in two list to check if they have similar predicates, return a binary list
    verb_similarity = []
    pos1, lemma1 = get_pos_and_lemma(sent1)
    pos2, lemma2 = get_pos_and_lemma(sent2)
    for i, (w1, w2) in enumerate(zip(pos1, pos2)):
        sim_num = 0
        if "VERB" in w1 and "VERB" in w2:
            verb1 = lemma1[i][w1.index("VERB")]
            verb2 = lemma2[i][w2.index("VERB")]
            if verb1 == verb2:
                sim_num = 1
            else:
                #consider the same with similarity higher than 0.6
                if get_word_similarity(verb1,verb2) > 0.6:
                    sim_num =1
        verb_similarity.append(sim_num)
    return verb_similarity

def check_if_negated(sent):
    toks = sent.strip().split()
    if "no" in toks or "not" in toks or "don't" in toks or "didn't" in toks:
        return 1
    else:
        return 0
def get_sent_length(sent):
    return len(sent)

def compare_sentence_length(sent1, sent2):
    sent_len_diff = sent1.apply(get_sent_length) - sent2.apply(get_sent_length)
    print("sentence length differnces", sent_len_diff)
    print("sent length type", type(sent_len_diff))
    return sent_len_diff

def get_bleu_score(combined_sent):
  # take a combined sentences (sent1 \t sent2) as input
  sent1, sent2 = combined_sent.strip().split("\t")
  #experiment with different smoothing functions
  smoothie = SmoothingFunction().method1
  #lowercase everything
  ref = sent1.split()
  hyp = sent2.split()
  # the weights determine the weighted geometric mean score from n-grams
  score = sentence_bleu([ref], hyp, weights=(0.75,0.25,0,0), smoothing_function=smoothie) # we use unigram and bigrams
  return score
