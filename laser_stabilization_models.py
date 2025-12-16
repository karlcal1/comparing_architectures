from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import pyarrow.feather as feather


## Eugene's visualization code
visualize_Fig2 = False # Set to True to visualize Fig2
df= dataframe = pd.read_feather("combined_centroid_data.feather")
df = reduced_size= df[:10000]
X_calib = 6.9  # um per pixel
Y_calib = 6.9  # um per pixel
sample_rate = 1000 # 1 ms time step, 1 kHz laser
x=df["Centroid_X"].values
y=df["Centroid_Y"].values
time = df["Time (ms)"].values

x_offset = min(x)
y_offset = min(y)
x_data = (x - x_offset) * X_calib
y_data = (y - y_offset) * Y_calib

# Do some FFTs
x_fft = np.fft.fft(x_data)
y_fft = np.fft.fft(y_data)

n_samples = len(x_data)
freq_array = np.fft.fftfreq(n_samples, 1/sample_rate) # For 1 ms time step

# keep only positive frequencies
positive_freqs = freq_array[:n_samples//2]
x_fft_magnitude = np.abs(x_fft[:n_samples//2])
y_fft_magnitude = np.abs(y_fft[:n_samples//2])
window_samples = 500 # Rolling average window size
x_rolling_mean = pd.Series(x_data).rolling(window=window_samples, center=True, min_periods=1).mean()
x_rolling_std = pd.Series(x_data).rolling(window=window_samples, center=True, min_periods=1).std()
y_rolling_mean = pd.Series(y_data).rolling(window=window_samples, center=True, min_periods=1).mean()
y_rolling_std = pd.Series(y_data).rolling(window=window_samples, center=True, min_periods=1).std()

if visualize_Fig2 == True:
    plt.figure(figsize=(20, 18))
    plt.subplot(3, 2, 1) # First plot
    plt.plot(time, x_data, label=f'Raw Centroid x', alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Centroid X (μm)')
    plt.title(f'Raw Centroid X')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2) # Second plot
    plt.plot(time, y_data, label=f'Raw Centroid y', alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Centroid Y (μm)')
    plt.title(f'Raw Centroid Y')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.subplot(3, 2, 3)
    # Plot sigma bands first (so they appear behind the mean line)
    plt.fill_between(time, 
                    x_rolling_mean - 2*x_rolling_std, 
                    x_rolling_mean + 2*x_rolling_std, 
                    alpha=0.2, color='red', label=f'±2σ={2*np.std(x_data):.1f} μm')
    plt.fill_between(time, 
                    x_rolling_mean - x_rolling_std, 
                    x_rolling_mean + x_rolling_std, 
                    alpha=0.3, color='orange', label=f'±1σ={np.std(x_data):.1f} μm')
    # Plot the running mean on top
    plt.plot(time, x_rolling_mean, 'b-', linewidth=2, 
            label=f'500 ms Running Avg')
    plt.xlabel('Time (ms)')
    plt.ylabel('Centroid X (μm)')
    plt.title(f'Running Average Centroid X (500ms window)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    # Plot sigma bands first (so they appear behind the mean line)
    plt.fill_between(time, 
                    y_rolling_mean - 2*y_rolling_std, 
                    y_rolling_mean + 2*y_rolling_std, 
                    alpha=0.2, color='red', label=f'±2σ={2*np.std(y_data):.1f} μm')
    plt.fill_between(time, 
                    y_rolling_mean - y_rolling_std, 
                    y_rolling_mean + y_rolling_std, 
                    alpha=0.3, color='orange', label=f'±1σ={np.std(y_data):.1f} μm')
    # Plot the running mean on top
    plt.plot(time, y_rolling_mean, 'b-', linewidth=2, 
            label=f'500 ms Running Avg')
    plt.xlabel('Time (ms)')
    plt.ylabel('Centroid Y (μm)')
    plt.title(f'Running Average Centroid Y (500ms window)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # FFT Centroid X plot
    plt.subplot(3, 2, 5)
    plt.semilogy(positive_freqs, x_fft_magnitude, 'b-', alpha=0.8, 
                label=f'FFT Centroid X')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude (arb)')
    plt.title(f'FFT Centroid X')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)  # Nyquist frequency limit

    # FFT Centroid X plot
    plt.subplot(3, 2, 6)
    plt.semilogy(positive_freqs, y_fft_magnitude, 'r-', alpha=0.8, 
                label=f'FFT Centroid Y')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FFT Amplitude (arb)')
    plt.title(f'FFT Centroid Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)  # Nyquist frequency limit
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.0)

    plt.savefig("centroid_analysis_1_14_2022_3_31_39_PM.png", dpi=300, bbox_inches='tight')
    plt.show()


# End of Eugene's visualization code

# Beginning Vincent's MLP code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import random
DATA_FILE = "combined_centroid_data.feather"

def train_eval(axis, data, gap, window):
    np.random.seed(67)
    """
    This part takes 10 random 10 second (10k datapoint) samples from the entire dataset, and ensures there is no overlap
    Then it creates a x set of window-sized consecutive segemnts within each chunk
    The y is then a gap ahead of the end of each x list
    """ 
    start = []
    while len(start)<10:
        new = random.randrange(0, len(data)-10000)
        for i in start:
            if abs(i-new)<10000:
                continue
        start.append(new)
    x_train = []
    y_train = []
    x_test = []
    y_test=[]
    for count, i in enumerate(start):
        chunk = data[i:i+10000]
        chunk = chunk-np.mean(chunk)
        
        x = [chunk[l:l+window] for l in range(10000-window-gap)]
        y = [chunk[l+window+gap] for l in range(10000-window-gap)]
        if count < 8:
            x_train.extend(x)
            y_train.extend(y)
        else:
            y_test.extend(y)
            x_test.extend(x)

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test=np.array(y_test, dtype=np.float32)

    """
    after the data is seperated, it is then scaled using a standardscaler
    We then use the scikit MLPRegressor, with a higher alpha value, as there were overfitting issues with the default
    """

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    mlp=MLPRegressor(hidden_layer_sizes=(128, 64), alpha = 0.05, early_stopping = False)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    natural_jitter = np.std(y_test)
    gain = 100 *(1-rmse/natural_jitter)
    y_train_pred = mlp.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    """
    The model weights and biases are then output through saving to a joblib
    The other results are returned in a dictionary so it can be easily accessed in the future
    """

    joblib.dump(mlp, f"{axis}_{gap}.joblib")
    return {'Axis': axis, 'Window': window,  'Gap':gap,'Train RMSE': train_rmse, 'Test RMSE' : rmse, 'Gain' : gain, 'Train_Loss' : mlp.loss_curve_}


def plot(results):
    """
    This function makes two loss curve plots, one for the jitter along x and one for the jitter along y
    """
    fig, axes = plt.subplots(1, 2, figsize = (10, 6))
    for i, res in enumerate(results):
        ax = axes[i]
        ax.set_xlabel('epochs')
        ax.set_ylabel("training loss")
        ax.plot(res['Train_Loss'])
        ax.set_title(f"{res['Axis']}")

    fig.tight_layout()
    fig.savefig(f"{results[0]['Window']}_gap{results[0]['Gap']}_loss_cuve.png")
    #plt.show()

if __name__=="__main__":
    """
    This code runs the functions we defined above, saves all outputs to a csv file
    """
    df = pd.read_feather(DATA_FILE)
    for gap in [5,10,20]:
        for window in [150,300,600]:
            results = [train_eval(ax, df[ax].values, gap, window) for ax in ['Centroid_X', 'Centroid_Y']]
            pd.DataFrame(results).to_csv(f"{window}_{gap}_summary.csv", index=False)
            print(gap, window, results[0]['Train RMSE'], results[0]['Test RMSE'], results[1]['Train RMSE'], results[1]['Test RMSE'], results[0]['Gain'], results[1]['Gain'])
            plot(results)
    
## End of Vincent's MLP code

## Beginning of Karl's advanced models code

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
our_data = reduced_size= our_data[:25000] #the paper didnt use all the gazilliopn timesteps, theu used 10**5. but it'll take way to long to run for our purposes, like several days for sure on cpu, and probably over a day on gpu.
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


#lets create a class to return input and output data. not necessary, but i think it lookw pretty clean
class OurJitterDataset():
    def __init__(self,data,window,gap):
        self.data=data
        self.window=window
        self.gap=gap

    def __len__(self): #i use this s.t. i can run .random_split down below 
        return len(self.data)-self.window-self.gap #also this makes it so we dont go outside the data!
    
    def __getitem__(self,i):
        input= torch.tensor(self.data[i:i+self.window], dtype=torch.float32) #thisll b 600,2 or howevr long we want our window to be.
        output=torch.tensor( self.data[i+self.window+self.gap], dtype=torch.float32) #thisll b 2,. the output prediction
        input = input.to(device) # if GPU is available, move to GPU
        output = output.to(device)
        return input,output
    
# MODELS ########################
#### LSTM
#lets roll out the model
class OurLSTM(nn.Module):
    def __init__(self, hidden, num_layers,window): #im just including window here bc it makes it easier in the class loop below b.c. the transformer does need this argument
  
        super().__init__()
        self.lstm=nn.LSTM(2,hidden_size=hidden, num_layers=num_layers,batch_first=True)
        self.out_proj=nn.Linear(hidden,2)

    def forward(self,x):
        output_of_the_lstm,(final_hidden_state,more_irrelevant_stuff)= self.lstm(x)  #LSTM outputs the output (which is all the h's, but we dont need all), as well as the short term memory h and the cell c. but we only care about the last short term mem bc thats whats actually needed for prediction 
        return self.out_proj(final_hidden_state[-1]) #final_hidden_staete gives us last hidden state of each layer, so equiv to [0] in the case where we just have 1 layer. 
#### LSTM+CNN
class OurCNNLSTM(nn.Module): 
    #combine cnn and lstm. however, realize nn.Conv1d doesnt take in (batch,seq_len,feature). 
    #instead, cnn's have the feature dim and the seq_len dim flipped. that's why im permuting the order of the data inside the forward pass
    #btw feature_vectors is called "channels" by cnn ppl bc u have 3 "channels" in RGB
    #anyway, having the kernel to be 5 is pretty standard, and its pretty common to just pad up the data so u have the same amount of inputs as outputs
    #altho in this case it doesnt matter bc the lstm will eat it up regardless of whether the output has a different seq_len
    def __init__(self, hidden, num_layers,window):
        super().__init__() 
        self.conv= nn.Conv1d(2,32,kernel_size=5,padding="same")   
        self.lstm= nn.LSTM(32,hidden_size=hidden,num_layers=num_layers,batch_first=True)
        self.out_proj= nn.Linear(hidden, 2)
    
    def forward(self, x):
        x= x.permute(0,2,1)        
        x= torch.relu(self.conv(x))   #you have to do an actuvation fn after nn.Conv1d bc nn.Conv1d just has sliding dot products, which is a linear operation. 
        x= x.permute(0,2,1)      
        output_of_the_lstm,(final_hidden_state,more_irrelevant_stuff)= self.lstm(x)
        return self.out_proj(final_hidden_state[-1])
#### TRANSF
class OurTransformer(nn.Module):
    def __init__(self, hidden, num_layers,window):
        super().__init__()
        self.input_proj=nn.Linear(2,hidden)
        self.positional_embedding = nn.Parameter( torch.randn(  1,window,hidden  ) *.01 )      # heres what i forgot before. now i fixed this s.t. theres positional embedding. also i added .01* s.t. this positional embedding doesnt totally overshadow the x values. u cud also use nn.Embedding. also i wrote 1 for ther first entry in torch.randn bc itll broadcast to however many samples is fed to the model anyway
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden,nhead=4,batch_first=True)
        self.transformer= nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.out_proj= nn.Linear(hidden,2)
    
    def forward(self,x):
        x= self.input_proj(x)   + self.positional_embedding  # i pushed in the positional embedding. so thats now baked into the transformer encoder, s.t. the attn can account for positional encoding, st the final step of the attn has encoded WHEN the sequence's other values occured 
        x= self.transformer(x)         
        return self.out_proj(x[:, -1])   #last timestep
                 


#the 3 hyperparams to vary
for gap in [5,10,20]:
    for window in [150,300,600]:

        #load up the data. we can put it here bc it only depends on gap and window, not num layers in the model
        our_dataset = OurJitterDataset(our_data,window,gap)
        train_size=int(.8*len(our_dataset))
            
        train_data = torch.utils.data.Subset(our_dataset, range(train_size)) #take in  the indices. this'll work w the class i made, unlike our_dataset[:train_size], since we gotta __getitem__ as integer indices, not slice
        test_data = torch.utils.data.Subset(our_dataset, range(train_size,len(our_dataset)))
        ###BAD###train_data,test_data= torch.utils.data.random_split(our_dataset,[train_size,len(our_dataset)-train_size])


        train_loader= DataLoader(train_data,batch_size=batch_siz,shuffle=True)
        test_loader= DataLoader(test_data,batch_size=batch_siz,shuffle=True)

        for num_layers in [1,2,3]:
        # for num_layers in [3,2,1]:
            for model_class in [OurLSTM, OurCNNLSTM , OurTransformer]:
            # for model_class in [OurTransformer, OurLSTM, OurCNNLSTM]:

                our_model = model_class(hidden,num_layers,window) #instantiating and initializing the model based on the model class which we loop over
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
                    with torch.no_grad(): #just makes the runs a little faster by disabling the whole computational graph gradient tracking. it matters alot actually when u run big models.
                        for inp,target in train_loader:
                            pred= our_model(inp)
                            train_errors.append((pred-target).detach().cpu().numpy())
                    train_errors=np.concatenate(train_errors)
                    rmse_x= np.sqrt((train_errors[:,0]**2).mean())
                    rmse_y= np.sqrt((train_errors[:,1]**2).mean())
                    train_rmse_list_x.append(rmse_x)
                    train_rmse_list_y.append(rmse_y)

                    #this is all pretty standard stuff. its just checking errors. and the .no_grad() is because we dont need to waste memory on tracking gradients when were not running any backward pass anyway
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
                plt.close()
                #lets save the trained model. lemme just give it unmistakeable names:
                torch.save(our_model.state_dict(), f"outputs/params_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}")


                np.save(f"outputs/train_rmse_x_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", train_rmse_list_x)
                np.save(f"outputs/train_rmse_y_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", train_rmse_list_y)
                np.save(f"outputs/test_rmse_x_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", test_rmse_list_x)
                np.save(f"outputs/test_rmse_y_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", test_rmse_list_y)
                np.save(f"outputs/train_loss_list_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", train_loss_list) #we can also plot this later shud we find it interesting. but rn lets not overpower ourselves, rmse is what ewe care about anyway. also these guys are gonna be per batchw whereas the rmse guys are per epoch
                np.save(f"outputs/test_loss_list_{model_class.__name__}_gap{gap}_window{window}_num_layers{num_layers}", test_loss_list)

                # plt.show()


## End of Karl's advanced models code

## Beginning of Eugene's results hyperparameter analysis code, this is for making Figure 11, I adjusted the parameters manually to get the other figures...

import numpy as np
import matplotlib.pyplot as plt
folder_path = 'outputs'
model_class = 'OurTransformer' # Choose btw OurTransformer, OurCNNLSTM, OurLSTM
# gap = 5
# window = 150
# num_layers = 2

my_best_rmse_x = 100000
my_best_rmse_y = 100000
gap_lst = [5]
window_lst = [600]
num_layers_lst = [1, 2, 3]
fig, ax1 = plt.subplots() 
ax1.tick_params(axis ='y') 
ax1.set_xlabel("Layer #")
ax1.set_ylabel("Test RMSE / um")
ax1.set_xticks(ticks=[1, 2, 3])
# ax1.set_xticks(ticks=[150, 300, 600])
ax2 = ax1.twinx() 
# for gap in gap_lst: # [5,10,20]
for gap in gap_lst: # [5,10,20]
    test_rmse_x_lst_for_all = []
    test_rmse_y_lst_for_all = []
    train_rmse_x_lst_for_all = []
    train_rmse_y_lst_for_all = []
    
    for window in window_lst: # [150,300,600]
        for num_layers in num_layers_lst: # [1,2,3]

        # for window in window_lst: # [150,300,600]
            test_rmse_list_x = np.load(f"{folder_path}/test_rmse_x_{model_class}_gap{gap}_window{window}_num_layers{num_layers} (1).npy")
            test_rmse_list_y = np.load(f"{folder_path}/test_rmse_y_{model_class}_gap{gap}_window{window}_num_layers{num_layers} (1).npy")
            train_rmse_list_x = np.load(f"{folder_path}/train_rmse_x_{model_class}_gap{gap}_window{window}_num_layers{num_layers} (1).npy")
            train_rmse_list_y = np.load(f"{folder_path}/train_rmse_y_{model_class}_gap{gap}_window{window}_num_layers{num_layers} (1).npy")
            # print(f"For {model_class}, gap {gap}, window size {window}, num_layers {num_layers}, RMSE X & Y:", test_rmse_list_x[-1], test_rmse_list_y[-1])
            if len(test_rmse_x_lst_for_all) > 0:
                if my_best_rmse_x > test_rmse_list_x[-1]:
                    my_best_rmse_x = test_rmse_list_x[-1]
                    print(f"For {model_class}, gap {gap}, window size {window}, num_layers {num_layers} better RMSE X found:", my_best_rmse_x)
                if my_best_rmse_y > test_rmse_list_y[-1]:
                    my_best_rmse_y = test_rmse_list_y[-1]
                    print("Note: current RMSE_x is", test_rmse_list_x[-1])
                    print(f"For {model_class}, gap {gap}, window size {window}, num_layers {num_layers} better RMSE Y found:", my_best_rmse_y)
            test_rmse_x_lst_for_all.append(test_rmse_list_x[-1])
            test_rmse_y_lst_for_all.append(test_rmse_list_y[-1])
            train_rmse_x_lst_for_all.append(train_rmse_list_x[-1])
            train_rmse_y_lst_for_all.append(train_rmse_list_y[-1])
            print(gap, window, num_layers, test_rmse_list_x[-1], test_rmse_list_y[-1], train_rmse_list_x[-1], train_rmse_list_y[-1])
    # ax1.plot(gap_lst, test_rmse_x_lst_for_all, label='Test X, layer # '+str(num_layers))
    # ax1.plot(gap_lst, test_rmse_y_lst_for_all, label='Test Y, layer # '+str(num_layers))
    # ax2.plot(gap_lst, train_rmse_x_lst_for_all, label='Train X, layer # '+str(num_layers), linestyle='dashed')
    # ax2.plot(gap_lst, train_rmse_y_lst_for_all, label='Train Y, layer # '+str(num_layers), linestyle='dashed')
    # ax1.plot(num_layers_lst, test_rmse_x_lst_for_all, label='Test X, layer # '+str(num_layers))
    # ax1.plot(num_layers_lst, test_rmse_y_lst_for_all, label='Test Y, layer # '+str(num_layers))
    # ax2.plot(num_layers_lst, train_rmse_x_lst_for_all, label='Train X, layer # '+str(num_layers), linestyle='dashed')
    # ax2.plot(num_layers_lst, train_rmse_y_lst_for_all, label='Train Y, layer # '+str(num_layers), linestyle='dashed')
    ax1.plot(num_layers_lst, test_rmse_x_lst_for_all, label='Test X')
    ax1.plot(num_layers_lst, test_rmse_y_lst_for_all, label='Test Y')
    ax2.plot(num_layers_lst, train_rmse_x_lst_for_all, label='Train X', linestyle='dashed')
    ax2.plot(num_layers_lst, train_rmse_y_lst_for_all, label='Train Y', linestyle='dashed')

# Create legend after all plots are done
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_lines = lines1 + lines2
all_labels = labels1 + labels2
ax2.legend(all_lines, all_labels, loc='upper right')

print("Best RMSE X:", my_best_rmse_x)
print("Best RMSE Y:", my_best_rmse_y)
ax2.set_ylabel("Train RMSE / um")
ax2.tick_params(axis ='y')
plt.title(f'RMSE for Transformer with gap size {gap}, window size {window}')
plt.show()
