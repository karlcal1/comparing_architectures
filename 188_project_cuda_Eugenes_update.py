#libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import time

#gpu acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#data loading
df= dataframe = pd.read_feather("combined_centroid_data.feather")
x=df["Centroid_X"].values #.values is a pd attribute #gta take .values attribute to remove all the indexing, just returns array
y=df["Centroid_Y"].values
our_data= np.stack([x,y], axis=1) #stack invents a new axis, we tryna couple the [x_i y_i]
our_data = reduced_size= our_data[:20000] #the paper didnt use all the gazilliopn timesteps, theu used 10**5 i think. but it'll take way to long to run for our purposes, like several days.
print("heres what the dataframe looks like btw:")
print(df[:10])
print(x)
print(y)
print(our_data)
print("data length: ",len(our_data))

#hyperparams to keep fixed
epochs= 100 #if it doesnt converge, i gotta adjust
batch_siz= 128 #just depending on speed mostly, big minib. faster
lr=.002
hidden=64 #REMEMBER THIS GOTTA B DIVIDISIBLE B nhead WHICH I PUT AS 4 FOR THRANDFORMER
#lets create a class to return input and output data 

class OurJitterDataset():
    def __init__(self,data,window,gap):
        self.data=data
        self.window=window
        self.gap=gap

    def __len__(self): #i use this s.t. i can run .random_split down below
        return len(self.data)-self.window-self.gap #also this makes it so we dont go outside the data!
    
    def __getitem__(self,i):
        input= torch.tensor(self.data[i:i+self.window], dtype=torch.float32) #thisll b 600,2
        output=torch.tensor( self.data[i+self.window+self.gap], dtype=torch.float32) #thisll b 2,. the output prediction
        input = input.to(device) # if GPU is available, move to GPU
        output = output.to(device)
        return input,output
    
# MODELS ########################
#### LSTM
#lets roll out the model
class OurLSTM(nn.Module):
    def __init__(self, hidden, num_layers):
        super().__init__()
        self.lstm=nn.LSTM(2,hidden_size=hidden, num_layers=num_layers,batch_first=True)
        self.out_proj=nn.Linear(hidden,2)

    def forward(self,x):
        output_of_the_lstm,(final_hidden_state,more_irrelevant_stuff)= self.lstm(x)  #
        return self.out_proj(final_hidden_state[-1]) #last layer hidden state, equiv to [0] for us cuz we just have 1 layer
#### LSTM+CNN
class OurCNNLSTM(nn.Module):
    def __init__(self, hidden, num_layers):
        super().__init__()
        self.conv= nn.Conv1d(2,32,kernel_size=5,padding=2)  
        self.lstm= nn.LSTM(32,hidden_size=hidden,num_layers=num_layers,batch_first=True)
        self.out_proj= nn.Linear(hidden, 2)
    
    def forward(self, x):
        x= x.permute(0,2,1)        
        x= torch.relu(self.conv(x))  
        x= x.permute(0,2,1)      
        output_of_the_lstm,(final_hidden_state,more_irrelevant_stuff)= self.lstm(x)
        return self.out_proj(final_hidden_state[-1])
#### TRANSF

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # Obtained from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1/
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class OurTransformer(nn.Module):
    def __init__(self, hidden, num_layers):
        super().__init__()
        self.input_proj=nn.Linear(2,hidden)
        self.positional_encoder = PositionalEncoding(dim_model = hidden, dropout_p = 0.1, max_len=5000)
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=4,batch_first=True)
        self.transformer= nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.out_proj= nn.Linear(hidden,2)
    
    def forward(self,x):
        x= self.input_proj(x)         
        x = self.positional_encoder(x)
        x= self.transformer(x)         
        return self.out_proj(x[:, -1])   #last timestep
                    


#the 3 hyperparams to vary
for gap in [5,10,20]:
    for window in [150,300,600]:

        #load up the data. we can put it here bc it only depends on gap and window, not num layers in the model
        our_dataset = OurJitterDataset(our_data,window,gap)
        train_size=int(.8*len(our_dataset))
        train_data = OurJitterDataset(our_data[0:train_size],window,gap)
        test_data = OurJitterDataset(our_data[train_size:len(our_dataset)-train_size])
        #train_data,test_data= torch.utils.data.random_split(our_dataset,[train_size,len(our_dataset)-train_size])


        train_loader= DataLoader(train_data,batch_size=batch_siz,shuffle=True)
        test_loader= DataLoader(test_data,batch_size=batch_siz,shuffle=True)

        for num_layers in [1,2,3]:
        # for num_layers in [3,2,1]:
            for model_class in [OurLSTM, OurCNNLSTM , OurTransformer]:
            # for model_class in [OurTransformer, OurLSTM, OurCNNLSTM]:

                our_model = model_class(hidden,num_layers) #instantiating and initializing the model based on the model class which we loop over
                our_model = our_model.to(device) # passing model to GPU if available; else, CPU
                torch.cuda.synchronize() # for good measure
                # -------------------------------------------
                ###########################################here!!!!!!!!!!
                optimizer=torch.optim.Adam(our_model.parameters(),lr=lr)
                loss_fn = nn.MSELoss()

                train_rmse_list_x=[]
                train_rmse_list_y=[]
                test_rmse_list_x=[]
                test_rmse_list_y=[]
                train_loss_list=[]
                test_loss_list=[]

                #NOTICE! a thing to be mindful of here is that the model mse is taken per batch whereas rmse x list is taken per epoch... i will keep it like this because we dont need the model mse anyway, its just extra information, and rmse is more interpretable for our task anyway!

                start = time.time()
                print(f"model class {model_class.__name__} with gap = {gap}, window = {window}, num_layers = {num_layers} started {time.asctime(time.localtime())}")
                for epoch in range(epochs):
                    our_model.train()
                    torch.cuda.synchronize() # for good measure
                    for inp,target in train_loader:
                        pred=our_model(inp)
                        # print("pred.device", pred.device)
                        # print("target.device", target.device)
                        l=loss_fn(pred, target)
                        train_loss_list.append(l.item()) #.item() gets the value... bc remember, l will have gradient attched to it
                        optimizer.zero_grad()
                        l.backward()
                        optimizer.step()
                

                    our_model.eval() #so we disable dropout
                    
                    train_errors=[] #this is rmse
                    test_errors=[] #this is rmse

                    #lets look at the rmse
                    with torch.no_grad(): #just makes the runs a little faster by disabling the whole computational graph stuff. it matters alot actually when u run big models.
                        for inp,target in train_loader:
                            pred= our_model(inp)
                            train_errors.append((pred-target).detach().cpu().numpy())
                    train_errors=np.concatenate(train_errors)
                    rmse_x= np.sqrt((train_errors[:,0]**2).mean())
                    rmse_y= np.sqrt((train_errors[:,1]**2).mean())
                    train_rmse_list_x.append(rmse_x)
                    train_rmse_list_y.append(rmse_y)


                    with torch.no_grad():
                        for inp,target in test_loader:
                            pred= our_model(inp)
                            test_errors.append((pred-target).detach().cpu().numpy())
                            test_loss_list.append(loss_fn(pred,target).item())
                    test_errors=np.concatenate(test_errors)
                    rmse_x= np.sqrt((test_errors[:,0]**2).mean())
                    rmse_y= np.sqrt((test_errors[:,1]**2).mean())
                    test_rmse_list_x.append(rmse_x)
                    test_rmse_list_y.append(rmse_y)
                    if epoch%10==0:
                        epoch_time = time.time() - start
                        print(f"time = {epoch_time:.1f} s, epoch number {epoch+1}, test rmse_x = {rmse_x:.3f} μm, test rmse_y = {rmse_y:.3f} μm")

                end = time.time()
                print(f"done {time.asctime(time.localtime())}")
                print(f"runtime: {(end-start) // 60:.0f}:{(end-start) % 60:.0f}") 
                fig=plt.figure()
                ax=fig.add_subplot()
                ax.plot(train_rmse_list_x,label="train x")
                ax.plot(train_rmse_list_y,label="train y")
                ax.plot(test_rmse_list_x,label="test x")
                ax.plot(test_rmse_list_y,label="test y")
                ax.legend()
                ax.set_xlabel("training epoch")
                ax.set_ylabel("test loss rmse")
                ax.set_title(f"rmse of testing data for x and y \n {model_class.__name__}, gap = {gap}, window = {window}, num_layers= {num_layers}")
                
                #all the plots!!!! all plots are gonna get saved into the directory, so the folder we store this file in 
                plt.savefig(f"outputs/plot_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}")                        #turns out u gotta save before showing lest u get an empty plot. also use the name dunders to avoid those classic ugly brackets u get othewise
                #lets save the trained model. lemme just give it unmistakeable names:
                torch.save(our_model.state_dict(), f"outputs/params_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}")


                np.save(f"outputs/train_rmse_x_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", train_rmse_list_x)
                np.save(f"outputs/train_rmse_y_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", train_rmse_list_y)
                np.save(f"outputs/test_rmse_x_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", test_rmse_list_x)
                np.save(f"outputs/test_rmse_y_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", test_rmse_list_y)
                np.save(f"outputs/train_loss_list_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", train_loss_list) #we can also plot this later shud we find it interesting. but rn lets not overpower ourselves, rmse is what ewe care about anyway. also these guys are gonna be per batchw whereas the rmse guys are per epoch
                np.save(f"outputs/test_loss_list_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", test_loss_list)

                # plt.show()