import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
import imageio
import itertools
import numpy as np
import struct
import argparse
from GAN import Generator
from GAN import Discriminator

parser = argparse.ArgumentParser()
## add this line
parser.add_argument('-f')
parser.add_argument("--epoch", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--adj_bar", type=float, default=0.001, help="adj bar")
parser.add_argument("--adj_bar_2", type=bool, default=False, help="use adj bar")
parser.add_argument("--init_dim", type=int, default=7, help="dimensionality of the latent code")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--Ghid', type=int, default=64, help='Hidden feature number of G.')
parser.add_argument('--Dhid', type=int, default=128, help='Hidden feature number of D.')
parser.add_argument('--feature_num', type=int, default=6, help='num of conditions')
parser.add_argument('--take_place', type=bool, default=True, help='G and D take place during training')
parser.add_argument('--train_set', type=int, default=30, help='total number of epoch for each G and D')
parser.add_argument('--lr_2', type=int, default=3, help='fix learning rate')


opt = parser.parse_args()
print(opt)
print(opt.adj_bar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

if not os.path.exists('./spatial_gan_train/newp'):
    os.mkdir('./spatial_gan_train/newp')

if not os.path.exists('./spatial_gan_train/death'):
    os.mkdir('./spatial_gan_train/death')
	

################################ Data cleaning ##########################################
########################################################################################

region_width = 10
region_length = 10

x = np.loadtxt(open('./mobility_new_list.csv', "r"), delimiter=",", skiprows=0)
x = np.vstack([x[:109200, :], x[120120: , :]])
#x = x[54600:, :]
x = x.reshape(-1, region_width * region_length, 1)
#print('x shape:\n', x.shape)

## normalize X
## 

x_max_test = x.max()

x = x / (x.max())
x = (x - 0.5) / 0.5
#print(x.max())
#print(x.min())
#print('first two x samples:\n', x[0:2])

#x = x[:12012, :, :]
#x = np.vstack([x[:10920, :, :], x[12012: , :, :]])
#x = np.delete(x, excludex, 0)
print(x.shape)


## load y and normalize
y = np.loadtxt(open('./feature_new_two.csv', "r"), delimiter=",", skiprows=0)
y = np.vstack([y[:1092000, :], y[1201200: , :]])
y = np.hstack([y[:, :2], y[:, 3:7]])
# y2 = np.loadtxt(open('./12_weeks.csv', "r"), delimiter=',', skiprows=0)
# #poi = np.loadtxt(open('./poi_week_one_col.csv', "r"), delimiter=",", skiprows=0)

# old_pop = y2[:, 3]
# print(old_pop[0])

# old_pop_col = old_pop.reshape(-1)

# y[:, 3] = old_pop_col
#print(y.shape)
def min_max_normal(xfunc):
    max_x = xfunc.max()
    min_x = xfunc.min()
    xfunc = (xfunc - min_x) / (max_x - min_x)
    return xfunc

#y = np.hstack([y[:, :1], y[:, 2:7]])
#print(y.shape)
y_norm = np.zeros(shape=(y.shape[0],opt.feature_num))
#print(y_norm.shape)

for i in [0, 1, 2, 3, 4, 5]:
    y_norm[:, i] = min_max_normal(y[:, i])

#y_norm[:, 6] = (y_norm[:, 6] - 0.5) / 0.5
	
y = np.copy(y_norm)
##y = np.hstack((x,y))

#y = np.vstack([y[:1092000, :], y[1201200: , :]])
#y = np.hstack([y[:, 0].reshape(-1,1), y[: , 2:7]])
##y = np.hstack((x,y))

#y = np.vstack([y[:1092000, :], y[1201200: , :]])

y = y.reshape(-1, region_width * region_length, opt.feature_num)
#y = np.delete(y, excludex, 0)
print(y[0:2,:,:])


print('y shape:\n', y.shape)
print('first two samples:\n' ,y[0:2, :, :])

x = torch.tensor(x)
y = torch.tensor(y)

#print(y[0])


#cosine = np.loadtxt(open('./cosine_week.csv', "r"), delimiter=",", skiprows=0)
#y = np.ones(shape=(4998,100,100), dtype=int)
#cosine = np.divide(cosine,100)

#cosine_list_new = []
#for i in range(0,119):
#	cosine_list_new.append(cosine)
#new_cosine = np.concatenate(cosine_list_new)
#adj = new_cosine.reshape(-1, region_width * region_length, region_width * region_length)

#print(adj.shape)
#print(adj[0])
adj = np.zeros(shape=(12012,100,100))

print(adj.shape)

#adj = cosine.reshape(-1, 100, 100)
#print(adj.shape)
#print(adj[0])
#print(sum(adj[0,1, :]))
adj = torch.tensor(adj)
#print('adj shape:\n', adj.shape)

dataset = Data.TensorDataset(x, y, adj)
train_loader = Data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)

D = Discriminator(1 + opt.feature_num, opt.Dhid, 1).to(device)
G = Generator(opt.feature_num + opt.init_dim, opt.Ghid, 1).to(device)

opt_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
opt_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

################################ Loss histgram ##########################################
########################################################################################
def show_train_hist(hist, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(path)


################################ Training ##########################################
########################################################################################
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []

# training
for epoch in range(opt.epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    ##if epoch  == 10 or epoch == 20:
    #if epoch % opt.train_set == 0: #or epoch == 20:
    if epoch == 60 or epoch ==180 or (epoch >= 300 and (epoch % 150 == 0)): #== 180 or (epoch>=300 and (epoch % 200 ==0)): #or epoch == 200 or (epoch>200 and (epoch % 100 ==0)):
        opt_G.param_groups[0]['lr'] /= 10
        opt_D.param_groups[0]['lr'] /= 10
        torch.save(G.state_dict(), './spatial_gan_train/death/G_params_' + str(opt.init_dim) + str(opt.feature_num) + str(opt.lr) + str(opt.take_place) + str(opt.train_set) + str(opt.adj_bar_2) + '_' + str(epoch) + '.pkl')   # save parameters
        torch.save(D.state_dict(), './spatial_gan_train/death/D_params_' + str(opt.init_dim) + str(opt.feature_num) + str(opt.lr) + str(opt.take_place) + str(opt.train_set) + str(opt.adj_bar_2) + '_' + str(epoch) + '.pkl')

        # print('learning rate:\n',opt_G.param_groups[0]['lr'])
		
    counterprint = 0
	
    for step, (b_x, b_y, b_adj) in enumerate(train_loader):
        counterprint = counterprint + 1
		######################### Train Discriminator #######################
        D.zero_grad()
        num_img = b_x.size(0) # batch size

        real_img = Variable(b_x.to(device)).float()     # put tensor in Variable
        img_label = Variable(b_y.to(device)).float()
		
        img_adj = Variable(b_adj.to(device)).float()
        prob_real_img_right_pair = D(real_img, img_adj, img_label)
		
		
        noise = torch.randn(num_img, opt.init_dim * region_length * region_width).view(num_img, region_length * region_width, opt.init_dim)
        noise = Variable(noise.to(device))  # randomly generate noise

        fake_img = G(noise, img_adj, img_label)
        selected_indices = torch.LongTensor([3])
        selected_indices = Variable(selected_indices.to(device))
        img_index = torch.index_select(img_label, 2, selected_indices)
        t0 = Variable(torch.tensor(0).to(device))
        t1 = Variable(torch.tensor(1).to(device))
        tn1 = Variable(torch.tensor(-1).to(device))
        img_mask = torch.where(img_index > t0, t1, tn1).float()
        fake_img = torch.min(fake_img, img_mask)

        prob_fake_img_pair = D(fake_img, img_adj, img_label)


        # sample real imgs from database(just shuffle this batch imgs)
        shuffled_row_idx = torch.randperm(num_img)
        real_shuffled_img = b_x[shuffled_row_idx]
        real_shuffled_img = Variable(real_shuffled_img.to(device)).float()
        shuffled_adj = b_adj[shuffled_row_idx]
        shuffled_adj = Variable(shuffled_adj.to(device)).float()

        prob_real_img_wrong_pair = D(real_shuffled_img, shuffled_adj, img_label)
        
        # if epoch % 30 == 0 and counterprint == 1:
        #     print('img_label:\n', img_label)
        #     print('prob_real_img_right_pair:\n', prob_real_img_right_pair)
        #     print('prob_fake_img_pair:\n', prob_fake_img_pair)
        #     print('prob_real_img_wrong_pair:\n', prob_real_img_wrong_pair)
			
        D_loss = - torch.mean(torch.log(prob_real_img_right_pair) + 
			                              torch.log(1. - prob_fake_img_pair) + torch.log(1. - prob_real_img_wrong_pair))

        D_loss.backward()
        
        # if (epoch % 60 <= 30) and (epoch % 60 > 0):
        #     D_loss.backward()
            
        opt_D.step()

        D_losses.append(D_loss.item())

        ########################### Train Generator ############################# 
        G.zero_grad()
        # compute loss of fake_img
        noise2 = torch.randn(num_img, opt.init_dim * region_length * region_width).view(num_img, region_length * region_width, opt.init_dim)
        noise2 = Variable(noise2.to(device))  # randomly generate noise

        # create random label
        y_real = Variable(torch.ones(num_img).to(device))
        G_result = G(noise2, img_adj, img_label)
        selected_indices = torch.LongTensor([3])
        selected_indices = Variable(selected_indices.to(device))
        g_index = torch.index_select(img_label, 2, selected_indices)
        t0 = Variable(torch.tensor(0).to(device))
        t1 = Variable(torch.tensor(1).to(device))
        tn1 = Variable(torch.tensor(-1).to(device))
        g_mask = torch.where(g_index > t0, t1, tn1).float()
        G_result = torch.min(G_result, g_mask)


        D_result = D(G_result, img_adj, img_label).squeeze()
        # print("D score: ", D_result.cpu().data.numpy())

        G_loss = BCE_loss(D_result, y_real)

        G_loss.backward()

        # if (epoch % 60 > 30) or (epoch % 60 == 0):
        #     G_loss.backward()
            
        opt_G.step()

        G_losses.append(G_loss.item())
		
        # if epoch % 30 == 0 and counterprint == 1:
        #     print('G_result:\n', G_result)
        #     print('D_result:\n', D_result)
        
        if counterprint % 50 == 0:
            #print('Epoch [{}/{}], D_loss: {:.6f}, G_loss: {:.6f} '.format(epoch + 1, opt.epoch, D_loss.item(), G_loss.item()))
            print(epoch + 1, opt.epoch, D_loss.item(), G_loss.item())

        if epoch % 50 == 0 and counterprint % 50 == 0:
          sampleid = 1
          np.set_printoptions(suppress=True)
          G_array = G_result.cpu().data.numpy()
          sample = G_array[sampleid].reshape(10,10)
          sample = sample*0.5 + 0.5
          print("G:\n", sample*x_max_test)
          true_array = real_img.cpu().data.numpy()
          sample_true = true_array[sampleid].reshape(10,10)
          sample_true = sample_true*0.5 + 0.5
          print("G:\n", sample_true*x_max_test)

        np.set_printoptions(suppress=False)



    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    
	
show_train_hist(train_hist, path='./spatial_gan_train/death/train_loss_hist_' + str(opt.feature_num) + str(opt.lr) + str(opt.train_set) + str(opt.adj_bar_2) + '.png')
torch.save(G.state_dict(), './spatial_gan_train/death/G_params_' + str(opt.init_dim) + str(opt.feature_num) + str(opt.lr) + str(opt.take_place) + str(opt.train_set) + str(opt.adj_bar_2) + '.pkl')   # save parameters
torch.save(D.state_dict(), './spatial_gan_train/death/D_params_' + str(opt.init_dim) + str(opt.feature_num) + str(opt.lr) + str(opt.take_place) + str(opt.train_set) + str(opt.adj_bar_2) + '.pkl')

########################Test###############################
## 2989, 1092
num_img_new = 1092
num_img = 1092
number = 1092
# num_img = excludex
# number = len(excludex)

testresult = np.zeros(shape=(number, 10, 10))

for testid in range(0, 10):
  noise2 = torch.randn(number, opt.init_dim * region_length * region_width).view(number, region_length * region_width, opt.init_dim)
  noise2 = Variable(noise2.to(device))  # randomly generate noise

  img_label = y[num_img_new*11:num_img_new*11+num_img, :, :]
  #img_label = y[excludex]
  img_label = torch.tensor(img_label)
  img_label = Variable(img_label.to(device)).float()

  img_adj = np.zeros(shape=(number, 100, 100))
  img_adj = torch.tensor(img_adj)
  img_adj = Variable(img_adj.to(device)).float()

  # img_ha = ha_new[num_img_new*11:num_img_new*11+num_img, :, :]
  # img_ha = torch.tensor(img_ha)
  # img_ha = Variable(img_ha.to(device)).float()

  G.eval()

  G_result = G(noise2, img_adj, img_label)

  # create random label
  y_real = Variable(torch.ones(number).to(device))

  selected_indices = torch.LongTensor([4])
  selected_indices = Variable(selected_indices.to(device))
  img_index = torch.index_select(img_label, 2, selected_indices)
  t0 = Variable(torch.tensor(0).to(device))
  t1 = Variable(torch.tensor(1).to(device))
  tn1 = Variable(torch.tensor(-1).to(device))
  # G_result = G_result + img_ha
  # G_result[G_result > 1] = 1
  img_mask = torch.where(img_index > t0, t1, tn1).float()
  #G_result = (G_result + img_ha)/2
  G_result = torch.min(G_result, img_mask)


  D_result = D(G_result, img_adj, img_label).squeeze()
  # print("D score: ", D_result.cpu().data.numpy())
  testresult = testresult + G_result.cpu().data.numpy().reshape(-1, region_width, region_length)

  G_loss = BCE_loss(D_result, y_real)
  if testid % 2 == 0:
    print('testid: ', testid)

testresult = testresult/10

np.set_printoptions(suppress=True)
#print("D score: ", D_result.cpu().data.numpy())
print("G_loss: ", G_loss.item())
print(np.mean(D_result.cpu().data.numpy()))
testresult_real = (testresult*0.5 + 0.5) * x_max_test 
# testresult_real = (testresult)*x_max_test 
#print(testresult[0])
print(testresult_real[0]) ## print one image
#print(x_test_show[11232])
x_test_show

x_select = x_test_show[num_img_new*11:num_img_new*11+num_img, :, :]
#x_select = x_test_show[excludex[0]]

x_sample = x_select
print(x_sample[0].reshape(10,10))

#########reshape##########
one = np.zeros(shape=(58,70))
two = np.zeros(shape=(58,70))

for i in range(0,58):
  for j in range(0,70):
    one[i][j] = i
for i in range(0,58):
  for j in range(0,70):
    two[i][j] = j

from skimage.util.shape import view_as_windows
one_sub = view_as_windows(one, (10,10),1)
two_sub = view_as_windows(two, (10,10),1)

one_li = []
for i in range(0,49):
  for j in range(0,61):
    one_li.append(one_sub[i][j])
one_list = np.concatenate(one_li, axis=0)


two_li = []
for i in range(0,49):
  for j in range(0,61):
    two_li.append(two_sub[i][j])
two_list = np.concatenate(two_li, axis=0)

final = np.zeros(shape=(58,70), dtype=float)
count = np.zeros(shape=(58,70), dtype=float)
for i in range(0, len(one_list)):
  for j in range(0, len(one_list[1])):
    k = int(one_list[i][j])
    l = int(two_list[i][j])
    final[k][l] = final[k][l] + results[i][j]
    count[k][l] = count[k][l]+1