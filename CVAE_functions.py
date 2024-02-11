import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader 

import matplotlib.pyplot as plt
import numpy as np


class SignalDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][0] #load pair x0/x1
        y = self.data[idx][1]
        return x, y  
    
def closeness_penalty(recon_x):
    # Compute pairwise distances between points
    pairwise_distances = torch.cdist(recon_x.view(recon_x.size(0), -1), recon_x.view(recon_x.size(0), -1), p=2)
    # Penalize closeness (e.g., inverse distance)
    closeness_penalty = torch.reciprocal(pairwise_distances + 1e-6)  # Add epsilon to avoid division by zero
    return closeness_penalty

# def distance_threshold_penalty(recon_x, threshold):
#     # Calculate mean distance between points
#     mean_distance = torch.mean(torch.cdist(recon_x.view(recon_x.size(0), -1), recon_x.view(recon_x.size(0), -1), p=2))
#     # Penalize deviation from threshold
#     distance_deviation = torch.abs(mean_distance - threshold)
#     return distance_deviation

def even_distribution_penalty(recon_x, lower_limit, upper_limit):
    # Calculate the number of points outside the desired distribution range
    num_points_outside = torch.sum((recon_x < lower_limit) | (recon_x > upper_limit))
    # Penalize deviation from even distribution
    distribution_deviation = torch.abs(num_points_outside)
    return distribution_deviation

    
# define the loss function
def loss_function(recon_x, x, cond_data, mu, logvar, beta, wx, wy,lamb, fun_list):
    
    # Calculate penalties
    closeness_penalty_val = closeness_penalty(recon_x)
#     distance_threshold_penalty_val = distance_threshold_penalty(recon_x, threshold=0.1)
    even_distribution_penalty_val = even_distribution_penalty(recon_x, lower_limit=0, upper_limit=1)
    
    closeness_penalty_mean = torch.mean(closeness_penalty_val.float())
#     distance_threshold_penalty_sum = torch.sum(distance_threshold_penalty_val)
    even_distribution_penalty_sum = torch.sum(even_distribution_penalty_val.float())

    # Combine penalties
    similarity_penalty = closeness_penalty_mean + even_distribution_penalty_sum

    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    
    #recon_loss_fn = torch.nn.L1Loss(reduction='mean')
    # recon_loss_fn = torch.nn.L1Loss(reduction='sum')
    recon_loss_fn = torch.nn.MSELoss()
    x_loss  = recon_loss_fn(x, recon_x)
    
    # Calculate the next-wise-element functions in fun_list
    results_list = []
    x0 = recon_x[:,0,:,0].cpu().detach().numpy().flatten()
    x1 = recon_x[:,0,:,1].cpu().detach().numpy().flatten()
    
    for fun in fun_list:
        result = fun(x0, x1)
        results_list.append(result)
    
    Nw = recon_x.size(-2)
    recon_cond_data = np.vstack([results_list]).T.reshape(len(cond_data), Nw*len(fun_list))
    recon_cond_data = torch.Tensor(np.array(recon_cond_data)).type(torch.float)    
#     if torch.cuda.is_available():
    recon_cond_data = recon_cond_data.cuda()
#     if torch.backends.mps.is_available():
#         recon_cond_data = recon_cond_data.to(torch.device('mps'))
    y_loss =  recon_loss_fn(cond_data, recon_cond_data)
  
      
    total_loss = (beta * KLD + wx * x_loss + wy * y_loss + lamb * similarity_penalty).mean()
    
    return total_loss, KLD, x_loss, y_loss, similarity_penalty


def train_cvae(cvae, train_loader, optimizer, beta, wx, wy,lamb, epoch, fun_list, step_to_print=1):
    cvae.train()
    train_loss = 0.0
    KLD_loss = 0.0
    recon_loss = 0.0
    cond_loss = 0.0
    similarity_penalty = 0.0

    for batch_idx, (data, cond_data) in enumerate(train_loader):
        Nw = data.size(-2)
        cond_data = torch.reshape(cond_data, (len(cond_data), Nw * len(fun_list)))
#         if torch.backends.mps.is_available():
#             cond_data = cond_data.to(torch.device('mps'))
#             data = data.to(torch.device('mps'))
#         if torch.cuda.is_available():
        cond_data = cond_data.cuda()
        data = data.cuda()
            

        # ===================forward=====================
        recon_data, z_mean, z_logvar = cvae(data, cond_data)
        loss, loss_KLD, loss_x, loss_y, similarity= loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy,lamb ,fun_list)
        
        train_loss += loss.item()
        KLD_loss += loss_KLD.item()
        recon_loss += loss_x.item()
        cond_loss += loss_y.item()
        similarity_penalty += similarity.item()
        
        
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    KLD_loss /= len(train_loader)
    recon_loss /= len(train_loader)
    cond_loss /= len(train_loader)
    similarity_penalty /=len(train_loader)

    result_dict = {
        "epoch": epoch,
        "average_loss": train_loss,
        "KLD_loss": KLD_loss,
        "x_loss": recon_loss,
        "y_loss": cond_loss,
        "Similarity_penalty" : similarity_penalty,
    }

    if epoch % step_to_print == 0:
        print('Train Epoch {}: Average Loss: {:.6f}, KLD: {:3f}, x_loss: {:3f}, y_loss: {:3f}, similarity_penalty: {:3f}'.format(epoch, train_loss,
                                                                                                    KLD_loss,
                                                                                                    recon_loss, cond_loss, similarity_penalty))

    return result_dict
    
    

    
def test_cvae(cvae, test_loader, beta, wx, wy, lamb, fun_list):
    cvae.eval()
    test_loss = 0.0

    with torch.no_grad():
        
        for batch_idx, (data, cond_data) in enumerate(test_loader):
            Nw = data.size(-2)
            cond_data =  torch.reshape(cond_data, (len(cond_data), Nw* len(fun_list)))
#             if torch.cuda.is_available():
            cond_data =  cond_data.cuda()
            data = data.cuda()
            cond_data = cond_data.cuda()
#             if torch.backends.mps.is_available():
#                 cond_data = cond_data.to(torch.device('mps'))
#                 data = data.to(torch.device('mps'))

            recon_data, z_mean, z_logvar = cvae(data, cond_data)

            loss,_,_,_,_= loss_function(recon_data, data, cond_data, z_mean, z_logvar, beta, wx, wy, lamb, fun_list)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print('Test Loss: {:.6f}'.format(test_loss))
    return test_loss
    
    

    
def generate_samples(cvae, num_samples, given_y, input_shape, zmult = 1):
    
    cvae.eval()
    samples = []
    givens = []
    
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random latent vector
            z_rand = (torch.randn(*input_shape)*zmult).cuda()
#             if torch.cuda.is_available():
#                 z_rand = z_rand.cuda()
#             if torch.backends.mps.is_available():
#                 z_rand = z_rand.to(torch.device('mps'))
                
            num_args = cvae.encoder.forward.__code__.co_argcount
            if num_args > 2 :
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0), given_y))
            else: 
                z = cvae.sampling(*cvae.encoder(z_rand.unsqueeze(0)))
            # Another way to generate random latent vector
            #z = torch.randn(1, latent_dim).cuda()
            
            # Set conditional data as one of the given y 
            # Generate sample from decoder under given_y
            sample = cvae.decoder(z, given_y)
            samples.append(sample)
            givens.append(given_y)

    
    samples = torch.cat(samples, dim=0)   
    givens = torch.cat(givens, dim=0) 
    return samples, givens


def plot_samples(x, y, num_samples , n_cols = 10, fig_size = 2): 
    
    x = x[0:num_samples]
    y = y[0:num_samples]
    n_rows = round(len(x)/n_cols)
    
    plt.rcdefaults()
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(1.25*n_cols*fig_size, n_rows*fig_size))

     
    for j, ax in enumerate(axarr.flat):
        x0 = x[j,0,:,0].cpu().detach().numpy().flatten()
        x1 = x[j,0,:,1].cpu().detach().numpy().flatten()
        y0 = y[j,0,:,0].cpu().detach().numpy().flatten()
        y1 = y[j,0,:,1].cpu().detach().numpy().flatten()
        
        
        #y_gen = x0*x1
        
        ax.plot(range(50),x0)
        ax.plot(range(50),x1)
        #ax.plot(range(50),y_gen)
        ax.plot(range(50),y0, color = 'r', linestyle = 'dotted')
        ax.plot(range(50),y1, color = 'b', linestyle = 'dotted') 
        
        ax.set_xticks([])
        ax.set_yticks([])
        

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show() 
    
    

def plot_samples_stacked(x_given, x, y, fun_list, n_cols = 4, fig_size = 3): 
   
    plt.rcdefaults()
    
    x_num = x.size(-1)
    y_num = y.size(-1)
    n_cols = x_num + y_num 
    
    
    f, axs = plt.subplots(1, n_cols, figsize=(1.25*n_cols*fig_size, fig_size))
     
    
    for j in range(len(x)):
        
        for i in range(x_num + y_num):
            if i < x_num :
                x_i = x[j,0,:,i].cpu().detach().numpy().flatten()
                x_i_given = x_given[:,0,:,i].cpu().detach().numpy().flatten()
                axs[i].plot(range(50), x_i)
                axs[i].plot(range(50), x_i_given, color = 'r')
                axs[i].set_title(f'X{i}') 
                axs[i].set_ylim(0,1)
            else: 
                
                y0 = y[j,0,:,i-x_num].cpu().detach().numpy().flatten()
                
                axs[i].plot(range(50), y0, color = 'r')
                axs[i].set_title(f'Y{i-x_num}') 
                x0 = x[j,0,:,0].cpu().detach().numpy().flatten()
                x1 = x[j,0,:,1].cpu().detach().numpy().flatten()
                fun = fun_list[i-x_num]
                ygen = fun(x0,x1)
                axs[i].plot(range(50),ygen)
                    
        
            axs[i].set(xticks=[], yticks=[])    
    
    
    
    plt.show()           