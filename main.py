import torch.nn as nn
import torch
import argparse
import random
import numpy as np
import pdb
from torch.autograd import Variable
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--num_epochs', type = int, default = 20000)
parser.add_argument('--learning_rate', type = float, default = 0.05)
parser.add_argument('--num_bits', type = int, default = 10)
parser.add_argument('--num_units', type = int, default = 100)

parser.add_argument('--t_o',type=int,default=1,help='T_0')
parser.add_argument('--t_mul',type=int,default=2,help='T_mul')
parser.add_argument('--lr_min',type=float,default=0.005,help='lr_min')#0.005
parser.add_argument('--lr_max',type=float,default=0.5,help='lr_max')#0.5
parser.add_argument('--num_samples', type = int, default = 924)

class cos_anneal(object):
    def __init__(self,args):
        self.args = args
        if args.num_samples % args.batch_size == 0:
            self.T_i = args.t_o * (args.num_samples // args.batch_size)
        else:
            self.T_i = args.t_o * (args.num_samples // args.batch_size + 1)
        self.T_mul = args.t_mul
        self.lr_min = args.lr_min
        self.lr_max = args.lr_max
        self.get_frac_list(self.T_i)
        self.iter = 0
    
    def get_frac_list(self,num):
        self.frac_lst = [1. * i/(num-1) for i in range(num)]

    def get_lr(self):
        lr = self.lr_min + 0.5*(self.lr_max-self.lr_min)*(1.+np.cos(self.frac_lst[self.iter]*np.pi)) 
        self.iter += 1
        if self.iter == len(self.frac_lst):
            self.T_i = self.T_i*self.T_mul
            self.get_frac_list(self.T_i)
            self.iter = 0
        self.curr_lr = lr
        return lr


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def check(x):
    if x % 3 == 0 and x % 5 == 0:
        return 0
    if x % 3 == 0:
        return 1
    if x % 5 == 0:
        return 2
    return 3

def get_data(min,max):
    X = np.array([i for i in range(min,max+1)])
    np.random.shuffle(X)
    y = Variable(torch.LongTensor(np.array([check(x) for x in X]))).cuda()
    X = Variable(torch.FloatTensor(np.array([binary_encode(i, args.num_bits) for i in X]))).cuda()
    X_batches = list(torch.split(X,args.batch_size,dim=0)) 
    y_batches = list(torch.split(y,args.batch_size,dim=0))
    data = zip(X_batches,y_batches)
    return data,X,y
    
def train(data):
    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fb_model.parameters(), lr = scheduler.get_lr(), momentum = 0.1, nesterov = True)
    loss1 = 0
    num = 0
    for X_batch, y_batch in data:
        _,logits = fb_model(X_batch)
        loss = crit(logits,y_batch.view(-1)).view(-1,1)
        optimizer.zero_grad()    
        loss.backward()
        torch.nn.utils.clip_grad_norm(fb_model.parameters(), 0.25)
        optimizer.step()
        loss1 += loss.data
        num += 1
    loss1 = loss1.cpu()[0][0]
    return 1.*loss1/num

def validate(data):
    crit = nn.CrossEntropyLoss()
    loss1 = 0
    num = 0
    fb_model.eval()
    for X_batch, y_batch in data:
        _,logits = fb_model(X_batch)
        loss = crit(logits,y_batch.view(-1)).view(-1,1)
        loss1 += loss.data
        num += 1
    loss1 = loss1.cpu()[0][0]
    return 1.*loss1/num

def test1(X,y):
    #num = 0
    #for X_batch, y_batch in data:
    fb_model.eval()
    probs,_ = fb_model(X)
    sel = probs.max(1)[1].data.cpu()
    y = y.data.cpu()
    #pdb.set_trace()
    acc = 1.*torch.eq(sel,y).sum()/sel.size()[0]
    #num += 1
    
    return acc

def test2(data):
    num = 0
    acc = 0
    fb_model2.eval()
    for X_batch, y_batch in data:
        probs,_ = fb_model2(X_batch)
        sel = probs.max(1)[1].data.cpu()
        y = y_batch.data.cpu()
        #pdb.set_trace()
        acc += 1.*torch.eq(sel,y).sum()/sel.size()[0]
        num += 1
    
    return 1.*acc/num

set_seed(101)
args = parser.parse_args()

train_data,X,y = get_data(101,2**args.num_bits)
test_data,_,_ = get_data(1,100)
fb_model = Model(args)
fb_model.cuda()

best_loss = np.inf

scheduler = cos_anneal(args)
for epoch in range(args.num_epochs):
    train_loss = train(train_data)
    valid_loss = validate(test_data)
    print('| epoch %03d | trainloss %.5f | validloss %.5f | lr %.4f' % (epoch, train_loss, valid_loss, scheduler.curr_lr))
    print('-' * 60)
    if valid_loss < best_loss:
         fb_model2 = fb_model
         with open('model.th','wb') as f:
             torch.save(fb_model,f)

print('test acc = ',test2(test_data))


    
        
