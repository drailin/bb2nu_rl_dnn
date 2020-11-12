import torch.nn as nn
import torch as torch
import horovod.torch as hvd
import os, glob
import csv
import numpy as np
import hvd_util as hu

##from larcv import larcv_interface
from larcv.distributed_queue_interface import queue_interface
from collections import OrderedDict

import torch.optim as optim
import sparseconvnet as scn

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


hvd.init()
#seed = 314159
seed = 12188
print("hvd.size() is: " + str(hvd.size()))
print("hvd.local_rank() is: " + str(hvd.local_rank()))
print("hvd.rank() is: " + str(hvd.rank()))

# Horovod: pin GPU to local rank.
#torch.cuda.set_device(hvd.local_rank())

os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
torch.cuda.manual_seed(seed)

global_Nclass = 2 # bkgd, 0vbb, 2vbb
global_n_iterations_per_epoch = 800
global_n_iterations_val = 80
global_n_epochs = 10
global_batch_size = hvd.size()*128

# filter
#bg_wt_trn = 0.65
#bg_wt_tst = 0.44
# no weights
bg_wt_trn = 1.0
bg_wt_tst = 1.0

nepoch0 = 0

weight_decay = 1e-3 # 1e-6
learning_rate = 0.01 # 0.011
learning_rate = learning_rate*(0.9)**((nepoch0-1)//5)

max_voxels= '10000'
producer='sparse3d_voxels_group'

dimension = 3
nPlanes = 1

modelfilepath = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/nextnew/models/'

modelname = 'model-nextnew-lucyr.pkl'
historyname = 'history-nextnew-lucyr.csv'
scorename = 'scoreeval-nextnew-lucyr.csv'


def plot_classes_preds(images, labels, probs):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels
    X,Y,Z,E,I = images
    fig = plt.figure(figsize=(12, 6))
    for idx in np.arange(4):
        ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
        plt.scatter(X[I == idx], Y[I == idx], c=E[I == idx], marker='s', edgecolor='None', alpha=0.3)
        ax.set_title("{0:.3f}\n(label: {1})".format(
            probs[idx],
            labels[idx]),
            color=("green" if round(probs[idx])==labels[idx].item() else "red"))
        ax = fig.add_subplot(2, 4, idx+5, xticks=[], yticks=[])
        plt.scatter(X[I == idx], Z[I == idx], c=E[I == idx], marker='s', edgecolor='None', alpha=0.3)
    return fig
    
def larcvsparse_to_scnsparse_3d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    n_dims = input_array.shape[-1]
    split_tensors = np.split(input_array, n_dims, axis=-1)

    # To map out the non_zero locations now is easy:
    non_zero_inds = np.where(split_tensors[-1] != -999)

    # The batch dimension is just the first piece of the non-zero indexes:
    batch_size  = input_array.shape[0]
    batch_index = non_zero_inds[0]

    # Getting the voxel values (features) is also straightforward:
    features = np.expand_dims(split_tensors[-1][non_zero_inds],axis=-1)
    # normalize event energy to 1.0
    fsum = np.zeros(batch_size)
    for i in range(batch_size):
        bidx = batch_index == i
        fsum[i] = np.sum(features[bidx])
        features[bidx] = features[bidx]/fsum[i]

    # Lastly, we need to stack up the coordinates, which we do here:
    dimension_list = [0]*(len(split_tensors)-1)
    for i in range(len(split_tensors) - 1):
        dimension_list[i] = split_tensors[i][non_zero_inds]

    # Tack on the batch index to this list for stacking:
    dimension_list.append(batch_index)

    # And stack this into one np array:
    dimension = np.stack(dimension_list, axis=-1)
    #coords = np.array([dimension_list[iwt] for iwt in range(len(dimension_list))])

    output_array = (dimension, features, batch_size,)
    return output_array,fsum

def to_torch(minibatch_data):
    for key in minibatch_data:
        if key == 'entries' or key =='event_ids':
            continue
        if key == 'image':
            minibatch_data['image'] = (
                    torch.tensor(minibatch_data['image'][0]).long(),
                    torch.tensor(minibatch_data['image'][1], device=torch.device('cuda')).float(),
                    minibatch_data['image'][2],
                )
        else:
            minibatch_data[key] = torch.tensor(minibatch_data[key],device=torch.device('cuda'))
    
    return minibatch_data

'''
Model below is an example, inspired by 
https://github.com/facebookresearch/SparseConvNet/blob/master/examples/3d_segmentation/fully_convolutional.py
Not yet even debugged!
EC, 24-March-2019
'''

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.inputLayer = scn.InputLayer(dimension, spatial_size=512, mode=3)
        self.initialconv = scn.SubmanifoldConvolution(dimension, nPlanes, 64, 7, False)
        self.residual = scn.Identity()
        self.add = scn.AddTable()
        self.sparsebl11 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 64, 64, 3, False)).add(
            scn.BatchNormLeakyReLU(64)).add(
            scn.SubmanifoldConvolution(dimension, 64, 64, 3, False))
        self.sparsebl12 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 64, 64, 3, False)).add(
            scn.BatchNormLeakyReLU(64)).add(
            scn.SubmanifoldConvolution(dimension, 64, 64, 3, False))
        self.sparsebl21 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 128, 128, 3, False)).add(
            scn.BatchNormLeakyReLU(128)).add(
            scn.SubmanifoldConvolution(dimension, 128, 128, 3, False))
        self.sparsebl22 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 128, 128, 3, False)).add(
            scn.BatchNormLeakyReLU(128)).add(
            scn.SubmanifoldConvolution(dimension, 128, 128, 3, False))
        self.relu1 = scn.LeakyReLU(64)
        self.relu2 = scn.LeakyReLU(128)
        self.downsample1 = scn.Sequential().add(
            scn.Convolution(dimension, 64, 64, [2,2,2], [2,2,2], False)).add(
            scn.BatchNormLeakyReLU(64))
        self.downsample2 = scn.Sequential().add(
            scn.Convolution(dimension, 64, 128, [2,2,2], [2,2,2], False)).add(
            scn.BatchNormLeakyReLU(128))
        self.downsample3 = scn.Sequential().add(
            scn.Convolution(dimension, 128, 64, [4,4,4], [4,4,4], False)).add(
            scn.BatchNormLeakyReLU(64))
        self.downsample4 = scn.Sequential().add(
            scn.Convolution(dimension, 64, 2, [4,4,4], [4,4,4], False)).add(
            scn.BatchNormLeakyReLU(2))
        self.sparsetodense = scn.SparseToDense(dimension, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2*8*8*8, 2)
        self.linear3 = nn.Linear(2,1)

    def forward(self,x):
        x = self.inputLayer(x)
        x = self.initialconv(x)
        x = self.downsample1(x)
        # first resnet block 1
        res = self.residual(x)
        x = self.sparsebl11(x)
        x = self.add([x,res])
        x = self.relu1(x)
        # first resnet block 2
        res = self.residual(x)
        x = self.sparsebl12(x)
        x = self.add([x,res])
        x = self.relu2(x)
        # downsample convolution
        x = self.downsample2(x)
        # second resnet block 1
        res = self.residual(x)
        x = self.sparsebl21(x)
        x = self.add([x,res])
        x = self.relu2(x)
        # second resnet block 2
        res = self.residual(x)
        x = self.sparsebl22(x)
        x = self.add([x,res])
        x = self.relu2(x)
        # downsample convolution
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.sparsetodense(x)
        x = x.view(-1, 2*8*8*8)
        x = self.dropout1(x)
        x = nn.functional.elu(self.linear2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.linear3(x))
        return x

def load_model():
    net = Model().cuda()
    try:
        print ("Reading weights from file")
        net.load_state_dict(torch.load(modelfilepath+modelname))
        #net.eval()
        print("Succeeded.")
    except:
        print ("Failed to read pkl model. Proceeding from scratch.")

    return net

def init_optimizer(net):
    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(net.parameters(), lr=learning_rate * hvd.size(),
                          momentum=0.9, weight_decay=weight_decay)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate * hvd.size(), weight_decay=weight_decay)
    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=net.named_parameters())
    
    # This moves the optimizer to the GPU:
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    
    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs

    return lr_step, optimizer

def load_data():
    ''' initialize data loading '''
    # config files
    main_fname = os.environ['HOME']+'/bb2nu_rl_dnn/larcvconfig_train_lr.txt'
    aux_fname = os.environ['HOME']+'/bb2nu_rl_dnn/larcvconfig_test_lr.txt'
    # initilize io
    root_rank = hvd.size() - 1
    #_larcv_interface = queue_interface( random_access_mode="serial_access" )
    #_larcv_interface = queue_interface( random_access_mode="random_events" )
    _larcv_interface = queue_interface( random_access_mode="random_blocks" )
    
    # Prepare data managers:
    io_config = {
        'filler_name' : 'TrainIO',
        'filler_cfg'  : main_fname,
        'verbosity'   : 1,
        'make_copy'   : True
    }
    aux_io_config = {
        'filler_name' : 'TestIO',
        'filler_cfg'  : aux_fname,
        'verbosity'   : 1,
        'make_copy'   : True
    }
    # Build up the data_keys:
    data_keys = OrderedDict()
    data_keys['image'] = 'data'
    data_keys['label'] = 'label'
    aux_data_keys = OrderedDict()
    aux_data_keys['image'] = 'test_data'
    aux_data_keys['label'] = 'test_label'
    
    _larcv_interface.prepare_manager('train', io_config, global_batch_size, data_keys, color = 0)
    _larcv_interface.prepare_manager('test', aux_io_config, global_batch_size, aux_data_keys, color = 0)

    return _larcv_interface

def init_logger():
    historyfilepath = os.environ['PROJWORK']+'/nph133/nextnew/csvout/'
    fieldnames = ['Training_Validation', 'Iteration', 'Epoch', 'Loss',
                  'Accuracy', "Learning Rate"]
    if hvd.rank()==0:
        filename = historyfilepath+historyname
        csvfile = open(filename,'a')
        history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        history_writer.writeheader()

        writer = SummaryWriter(historyfilepath+'tblogs/')

    return history_writer,writer,csvfile
   
def prepare_batch(_larcv_interface,mode,bg_wt):
    _larcv_interface.prepare_next(mode)
    minibatch_data = _larcv_interface.fetch_minibatch_data(mode, pop=True, fetch_meta_data=False)
    minibatch_dims = _larcv_interface.fetch_minibatch_dims(mode)
    
    for key in minibatch_data:
        if key == 'entries' or key == 'event_ids':
            continue
        minibatch_data[key] = np.reshape(minibatch_data[key], minibatch_dims[key])
    
    # Strip off the primary/aux label in the keys:
    for key in minibatch_data:
        new_key = key.replace('aux_','')
        minibatch_data[new_key] = minibatch_data.pop(key)            
    
    minibatch_data['image'],fsum = larcvsparse_to_scnsparse_3d(minibatch_data['image'])
    minibatch_data = to_torch(minibatch_data)

    weights = torch.ones(len(minibatch_data['label'])).cuda() - minibatch_data['label'].transpose(dim0=1,dim1=0)[0]*(1.0 - bg_wt)

    return minibatch_data,minibatch_dims,fsum,weights
 

def train_fullysupervised():
   
    net = load_model() 
    lr_step,optimizer = init_optimizer(net) 
    _larcv_interface = load_data()
    if hvd.rank()==0:
        history_writer,writer,csvfile = init_logger()
    else:
        history_writer = None
        csvfile = None

    train_loss = hu.Metric('train_loss')
    train_accuracy = hu.Metric('train_accuracy')
    val_loss = hu.Metric('val_loss')
    val_accuracy = hu.Metric('val_accuracy')

    metrics = train_loss,train_accuracy,val_loss,val_accuracy
    
    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    Niterations = min(tr_epoch_size,global_n_iterations_per_epoch+1)

    for epoch in range(nepoch0, global_n_epochs + nepoch0):

        '''
        Run training per epoch
        '''
        lr_step.step()
        net.train()
        for iteration in range(tr_epoch_size):
            net.train()
            optimizer.zero_grad()
   
            minibatch_data,minibatch_dims,fsum,weights = prepare_batch(_larcv_interface,'train',bg_wt_trn)

            yhat = net(minibatch_data['image'])
            values, target = torch.max(minibatch_data['label'], dim=1)

            if hvd.rank() == 0:
                Xs = minibatch_data['image'][0].transpose(1,0)[0][minibatch_data['image'][0].transpose(1,0)[-1] < 4].cpu()
                Ys = minibatch_data['image'][0].transpose(1,0)[1][minibatch_data['image'][0].transpose(1,0)[-1] < 4].cpu()
                Zs = minibatch_data['image'][0].transpose(1,0)[2][minibatch_data['image'][0].transpose(1,0)[-1] < 4].cpu()
                Es = minibatch_data['image'][1].transpose(1,0)[0][minibatch_data['image'][0].transpose(1,0)[-1] < 4].cpu()
                Is = minibatch_data['image'][0].transpose(1,0)[-1][minibatch_data['image'][0].transpose(1,0)[-1] < 4].cpu()
                writer.add_figure('training images HL', plot_classes_preds((Xs,Ys,Zs,Es,Is), target[:4].cpu().detach().numpy(), yhat[:4].cpu().detach().numpy().T[0]), global_step=epoch+nepoch0)

            criterion = torch.nn.BCELoss(weight=weights.view(-1,1)).cuda()
            # backprop and optimizer step
            loss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1)) * len(weights) / weights.sum() #\
            #     + torch.abs(nposy/bagsize - labelprop)
            loss.backward()
            optimizer.step()

            # below is to keep this from exceeding 4 hrs
            if iteration > global_n_iterations_per_epoch:
                break

        '''
        Run evaluation per epoch
        '''
        # give network metrics to tensorboard
        if hvd.rank() == 0:
            writer.add_histogram('initial conv weights HL', net.initialconv.weight.data, epoch+nepoch0)
            writer.add_histogram('downsample4 weights HL', (net.downsample4)[0].weight.data, epoch+nepoch0)
            writer.add_histogram('linear2 weights HL', net.linear2.weight.data, epoch+nepoch0)

        eval_epoch(net,optimizer,_larcv_interface,history_writer,csvfile,metrics,epoch+nepoch0)

        if hvd.rank() == 0:
            writer.add_scalar('Loss/train', metrics[0].avg, epoch+nepoch0)
            writer.add_scalar('Accuracy/train', metrics[1].avg, epoch+nepoch0)
            writer.add_scalar('Loss/val', metrics[2].avg, epoch+nepoch0)
            writer.add_scalar('Accuracy/val', metrics[3].avg, epoch+nepoch0)
        
    '''
    Save model
    '''
    if hvd.rank() == 0:
        torch.save(net.state_dict(), modelfilepath+modelname)

    if hvd.rank() == 0:
        writer.close()
 

def eval_epoch(net,optimizer,_larcv_interface,history_writer,csvfile,metrics,epoch):
    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)

    train_loss,train_accuracy,val_loss,val_accuracy = metrics

    for param_group in optimizer.param_groups:
        lrnow = param_group['lr']
    
    net.eval()

    for iteration in range(tr_epoch_size):
        net.eval()
        minibatch_data,minibatch_dims,fsum,weights = prepare_batch(_larcv_interface,'train',bg_wt_trn) 

        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])

        values, target = torch.max(minibatch_data['label'], dim=1)
        
        criterion = torch.nn.BCELoss(weight=weights.view(-1,1)).cuda()
        acc = hu.accuracy(yhat, target.type(torch.FloatTensor), weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        tloss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1)) * len(weights) / weights.sum()
        train_loss.update(tloss)

        output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),
                  'Accuracy':float(train_accuracy.avg.data), "Learning Rate":lrnow}

        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()

        # below is to keep this from exceeding 4 hrs
        if iteration > global_n_iterations_per_epoch:
            break

    for iteration in range(te_epoch_size):
        net.eval()

        minibatch_data,minibatch_dims,fsum,weights = prepare_batch(_larcv_interface,'test',bg_wt_tst) 

        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])
        
        values, target = torch.max(minibatch_data['label'], dim=1)

        acc = hu.accuracy(yhat, target.type(torch.FloatTensor), weighted=True, nclass=global_Nclass)
        val_accuracy.update(acc)

        criterion = torch.nn.BCELoss(weight=weights.view(-1,1)).cuda()
        vloss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1)) * len(weights) / weights.sum()
        val_loss.update(vloss)

        print("Val.Epoch: {}, Iteration: {}, Train,Val Loss: [{:.4g},{:.4g}], *** Train,Val Accuracy: [{:.4g},{:.4g}] ***".format(epoch, iteration,float(train_loss.avg), val_loss.avg, train_accuracy.avg, val_accuracy.avg ))

        output = {'Training_Validation':'Validation','Iteration':iteration, 'Epoch':epoch, 
                  'Loss':float(val_loss.avg), 'Accuracy':float(val_accuracy.avg), "Learning Rate":lrnow}

        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()
        if iteration>=global_n_iterations_val:
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        csvfile.flush()

def score_new_events():

    net = load_model()
    _larcv_interface = load_data()

    #scfieldnames = ['Iteration', 'Class', 'Score0', 'Score1']
    scfieldnames = ['Iteration', 'Class', 'Score0','X','Y','Z','E']
    historyfilepath = os.environ['PROJWORK']+'/nph133/nextnew/csvout/'
    
    if hvd.rank()==0:
        scfilename = historyfilepath+scorename
        sccsvfile = open(scfilename,'w')
        score_writer = csv.DictWriter(sccsvfile, fieldnames=scfieldnames)
        score_writer.writeheader()
    
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    print('te_epoch_size: %s'%te_epoch_size)
    print('larcv size: %s'%_larcv_interface.size('test'))
    print('global_batch_size: %s'%global_batch_size)
    net.eval()
    for iteration in range(te_epoch_size):
        net.eval()
   
        minibatch_data,minibatch_dims,fsum,weights = prepare_batch(_larcv_interface,'test',bg_wt_tst)
    
        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])
        
        values, target = torch.max(minibatch_data['label'], dim=1)
   
        evtidc = torch.transpose(minibatch_data['image'][0],0,1)[-1] 
        for ievt in range(len(target)):
            targ = int(target[ievt])
            scr0 = float(yhat[ievt])
            #scr1 = float(yhat[ievt][1])
            img = torch.transpose(minibatch_data['image'][0][np.where(evtidc == ievt)],0,1).type(torch.float)
            val = minibatch_data['image'][1][np.where(evtidc == ievt)].type(torch.float)
            xmn = float(img[0].mean())
            ymn = float(img[1].mean())
            zmn = float(img[2].mean())*fsum[ievt]
            emn = float(fsum[ievt])
    
            #output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'Score1':scr1}
            output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'X':xmn, 'Y':ymn, 'Z':zmn, 'E':emn}
    
            if hvd.rank()==0:        
                score_writer.writerow(output)
            if iteration>=global_n_iterations_val:
                break # Just check val for 4 iterations and pop out
    
        if hvd.rank()==0:        
            sccsvfile.flush()


def score_new_data_events():

    net = load_model()
    _larcv_interface = load_data()

    #scfieldnames = ['Iteration', 'Class', 'Score0', 'Score1']
    scfieldnames = ['Iteration', 'Class', 'Score0','X','Y','Z','E']
    historyfilepath = os.environ['PROJWORK']+'/nph133/nextnew/csvout/'
    
    if hvd.rank()==0:
        scfilename = historyfilepath+scorename
        sccsvfile = open(scfilename,'w')
        score_writer = csv.DictWriter(sccsvfile, fieldnames=scfieldnames)
        score_writer.writeheader()
    
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    print('te_epoch_size: %s'%te_epoch_size)
    print('larcv size: %s'%_larcv_interface.size('test'))
    print('global_batch_size: %s'%global_batch_size)
    net.eval()
    for iteration in range(te_epoch_size):
        net.eval()
   
        minibatch_data,minibatch_dims,fsum,weights = prepare_batch(_larcv_interface,'test',bg_wt_tst)
    
        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])
        
        values, target = torch.max(minibatch_data['label'], dim=1)
   
        evtidc = torch.transpose(minibatch_data['image'][0],0,1)[-1] 
        for ievt in range(len(target)):
            targ = int(target[ievt])
            scr0 = float(yhat[ievt])
            #scr1 = float(yhat[ievt][1])
            img = torch.transpose(minibatch_data['image'][0][np.where(evtidc == ievt)],0,1).type(torch.float)
            val = minibatch_data['image'][1][np.where(evtidc == ievt)].type(torch.float)
            xmn = float(img[0].mean())
            ymn = float(img[1].mean())
            zmn = float(img[2].mean())
            deltaz = float(img[2].max() - img[2].min())
            emn = float(fsum[ievt]) * 1./(1 - 2.76e-4*deltaz)
    
            #output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'Score1':scr1}
            output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'X':xmn, 'Y':ymn, 'Z':zmn, 'E':emn}
    
            if hvd.rank()==0:        
                score_writer.writerow(output)
            if iteration>=global_n_iterations_val:
                break # Just check val for 4 iterations and pop out
    
        if hvd.rank()==0:        
            sccsvfile.flush()


