import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def collate_batch(batch):
    sents1 = []
    sents2 = []
    y_vals_1h = []
    y_vals = []
    max_len_sent1 = batch[0][0].size()[1]
    max_len_sent2 = batch[0][1].size()[1]
    
    for exmpl in batch:
        
        sents1.append(exmpl[0])
        sents2.append(exmpl[1])
        y_vals_1h.append(exmpl[2])
        y_vals.append(exmpl[3])
        
        
        max_len_sent1 = max(max_len_sent1, exmpl[0].size()[1])
        max_len_sent2 = max(max_len_sent2, exmpl[1].size()[1])
        

    for i in range(len(sents1)):
        x1,y1,z1 = sents1[i].size()
        last_word_sent1 = sents1[i][x1-1,y1-1,:]
        last_word_sent1 = torch.reshape(last_word_sent1,(1,1,768))
        
        x2,y2,z2 = sents2[i].size()
        last_word_sent2 = sents2[i][x2-1,y2-1,:]
        last_word_sent2 = torch.reshape(last_word_sent2,(1,1,768))
        
        while sents1[i].size()[1] < max_len_sent1:
            sents1[i] = torch.hstack((sents1[i],last_word_sent1))
            
        while sents2[i].size()[1] < max_len_sent2:
            sents2[i] = torch.hstack((sents2[i],last_word_sent2))
        
    sents1 = torch.cat(sents1, 0) 
    sents2 = torch.cat(sents2, 0) 

    
    y_vals_1h = torch.tensor(np.array(y_vals_1h))
    y_vals = torch.tensor(np.array(y_vals))
    
    return sents1, sents2, y_vals_1h, y_vals
    

def map_y_vals(y_val):
    y_dict = {'entailment':2, 'neutral':1, 'contradiction':0}
    return y_dict[y_val]


class NLIDataset(Dataset):
    def __init__(self, data_dir, len_sample=550152,prefix='train'):
        data = pd.read_csv('../ENLP_NLI/snli_1.0/snli_1.0_'+prefix+'.txt', sep='\t')[:len_sample]
        data = data[data['sentence1'].notna()]
        data = data[data['sentence2'].notna()]
        
        self.y = data['label1'].apply(map_y_vals).values
        
        
        self.sent1 = data['sentence1'].values#pickle.load( open( prefix+'_data_sent1_tokens.pkl', 'rb' ) )
        self.sent2 = data['sentence2'].values#pickle.load( open( prefix+'_data_sent2_tokens.pkl', 'rb' ) )
        del data
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        self.bert_model.eval()
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y_one_hot = np.zeros(3)
        y_one_hot[self.y[idx]] = 1
        
        with torch.no_grad():
            marked_text = self.sent1[idx]
            tokenized_text = self.tokenizer.tokenize(marked_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            outputs = self.bert_model(tokens_tensor)
            last_hidden_state_sent1 = outputs[0]
            
        with torch.no_grad():
            marked_text = self.sent2[idx]
            tokenized_text = self.tokenizer.tokenize(marked_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            outputs = self.bert_model(tokens_tensor)
            last_hidden_state_sent2 = outputs[0]
        
        #return self.sent1[idx], self.sent2[idx], y_one_hot, self.y[idx]
        return last_hidden_state_sent1, last_hidden_state_sent2, y_one_hot, self.y[idx]



'''
def collate_batch(batch):
    
    
    sents1 = []
    sents2 = []
    y_vals_1h = []
    y_vals = []
    
    for exmpl in batch:
        sents1.append(exmpl[0])
        sents2.append(exmpl[1])
        y_vals_1h.append(exmpl[2])
        y_vals.append(exmpl[3])
    
    sents1 = tokenizer.batch_encode_plus(sents1, padding='longest')['input_ids'] 
    sents2 = tokenizer.batch_encode_plus(sents2, padding='longest')['input_ids']  
    
    return torch.tensor(sents1), torch.tensor(sents2), torch.tensor(np.array(y_vals_1h)), torch.tensor(np.array(y_vals))
'''