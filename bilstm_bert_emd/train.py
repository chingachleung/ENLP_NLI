import os
import torch
from torch import nn
from torch.utils.data import Dataset

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from NLIModel import SiameseNLI
from NLIDataset import NLIDataset, collate_batch
from sklearn.metrics import accuracy_score
from tqdm import tqdm





def train(train_data_loc,test_data_loc):

    BATCH_SIZE = 64
    LEN_SAMPLE = 550152
    EPOCHS = 100
    LR = 1
    CLASS_WEIGHTS = torch.Tensor(np.ones(3))
    LOSS_FN = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    EVAL_BATCH = 0
    DEVICE = torch.device('cpu') 
    WEIGHT_DECAY = 0.1
    MOMENTUM = 0.0
    LSTM_LAYERS = 1#4
    INPUT_SIZE = 1#768
    EVAL_BATCH = int(LEN_SAMPLE / 3)
    
    HIDDEN_SIZE = 512
    
    print('Device:',DEVICE)

    # Set fixed random number seed
    torch.manual_seed(42)

    # Initialize the MLP
    model = SiameseNLI(input_size=INPUT_SIZE,num_layers=LSTM_LAYERS,hidden_size=HIDDEN_SIZE)
    model = model.to(DEVICE)


    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    

    sftmx = nn.Softmax(dim=1)
    loss_lst = []
    batch_count = 0
    prev_acc = 0.0
    prev_acc_dev = 0.0


    # Run the training loop
    for epoch in range(0, EPOCHS): # 5 epochs at maximum
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        current_loss = 0.0
        all_preds = torch.tensor([])
        all_targets = torch.tensor([])

        #process chunk
        train_dataset = NLIDataset(train_data_loc,len_sample=LEN_SAMPLE)
        # Prepare dataset
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_batch)
        
        #dev_dataset = NLIDataset(test_data_loc,len_sample=BATCH_SIZE*20, prefix='dev')
        #devloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_batch)

        with tqdm(total=int(LEN_SAMPLE/BATCH_SIZE)) as pbar:
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                model.train()

                # Get inputs
                sents1, sents2, targets_1h, targets  = data
                
                sents1 = sents1.to(DEVICE)
                sents2 = sents2.to(DEVICE)
                targets = targets.to(DEVICE)

                sents1 = torch.reshape(sents1,(len(sents1), -1,INPUT_SIZE))
                sents2 = torch.reshape(sents2,(len(sents2), -1,INPUT_SIZE))
                

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(sents1.float(),sents2.float())

                loss = LOSS_FN(outputs, targets_1h.float())
                preds = torch.argmax(sftmx(outputs.detach()),dim=1)

                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets.detach()))


                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics

                current_loss += loss.item()
                batch_count += 1
                #print('Current train loss:',current_loss/(i+1))


                f = open('loss_train.txt', 'a')
                f.write(str(current_loss/(i+1))+'\n')
                f.close()

                f = open('loss_train_curr.txt', 'a')
                f.write(str(loss.item())+'\n')
                f.close()
                
                if i % 500 == 0:
                    torch.save(model.state_dict(), 'models/nli_siamese_'+str(epoch)+'.pt')
                
                pbar.update(1)
                

        '''
        with torch.no_grad():
            curr_loss_dev = 0.0

            all_preds_dev = torch.tensor([])
            all_targets_dev = torch.tensor([])

            for j, data_dev in enumerate(devloader):
                sents1_dev, sents2_dev, targets_1h_dev, targets_dev  = data_dev

                outputs_dev = model(sents1_dev.float(),sents2_dev.float())
                loss_dev = LOSS_FN(outputs_dev, targets_1h_dev.float())

                preds_dev = torch.argmax(sftmx(outputs_dev.detach()),dim=1)

                curr_loss_dev += loss_dev.item()

                all_preds_dev = torch.cat((all_preds_dev, preds_dev))
                all_targets_dev = torch.cat((all_targets_dev, targets_dev.detach()))

            curr_loss_dev = curr_loss_dev / (j+1)
            print('Epoch dev loss:',curr_loss_dev)
            
            acc_dev = accuracy_score(all_targets_dev.numpy(), all_preds_dev.numpy())
            print('Dev accuracy:', acc_dev)

            f = open('loss_dev.txt', 'a')
            f.write(str(curr_loss_dev)+'\n')
            f.close()
            '''
                    
                  
                    
        #scheduler.print_lr()    
        #scheduler.step()
        print('Epoch train loss:',current_loss/(LEN_SAMPLE/BATCH_SIZE))
        acc = accuracy_score(all_targets.numpy(), all_preds.numpy())
        print('Train accuracy:', acc)
        
        if acc >= prev_acc:
            torch.save(model.state_dict(), 'models/nli_siamese.pt')
            prev_acc = acc
            #prev_acc_dev = acc_dev
            
            f = open('model_saves.txt', 'a')
            f.write(str(epoch)+'\n')
            f.close()
    


if __name__ == '__main__':
    train_data_loc = '../ENLP_NLI/snli_1.0/snli_1.0_train.txt'
    test_data_loc = '../ENLP_NLI/snli_1.0/snli_1.0_dev.txt'
    train(train_data_loc,test_data_loc)