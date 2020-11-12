"""Utilities for working with Horovod, just to make the coding look better."""
import horovod.torch as hvd
import torch
import torch.nn as nn
import csv
import os
import time
import numpy as np


def printh(*args, **kwargs):
    """Print from the head rank only."""
    if hvd.rank() == 0:
        print(*args, **kwargs)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.rsum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum = hvd.allreduce(val.detach().cpu(), name=self.name)
        self.rsum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    def reset(self):
        self.sum = torch.tensor(0.)
        self.rsum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    @property
    def avg(self):
        return self.sum #/ self.n
    @property
    def add(self):
        return self.rsum * self.n


def save_checkpoint(model, optimizer, epoch, path):
    if hvd.rank() == 0:
        filepath = os.path.join(path,
                                "checkpoint_epoch{epoch}.h5".format(epoch=epoch + 1))
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, average=False, name=name)
    return avg_tensor.item()


def accuracy(output, target, weighted=True, nclass=2):
    # get the index of the max log-probability
    #pred = output.max(1, keepdim=True)[1].cpu()
    pred = torch.round(output).cpu()
    if(weighted):
        w_acc = 0.
        nnonzero = 0
        for cl in range(nclass):
            icl = np.where(target.view_as(pred).cpu() == cl)
            if len(icl[0]) != 0:
                w_acc += (pred[icl]).eq(target.cpu().view_as(pred)[icl]).cpu().float().sum()/float(len(icl[0]))
                nnonzero += 1
        return w_acc/float(nnonzero)
    else:
        return pred.eq(target.view_as(pred)).cpu().float().mean()


def pdist(feats_1, feats_2, fsum_1, fsum_2, mb_1=[], mb_2=[]):
    erg_1 = np.zeros(len(fsum_1))
    erg_2 = np.zeros(len(fsum_2))
    erange_1 = []
    erange_2 = []
    pd = 0.
    N_1 = torch.tensor(0)
    N_2 = torch.tensor(0)
    if len(mb_1) != 0:
        erange_1 = [1.55, 1.63, 1.72, 1.80]
        evtidc = torch.transpose(mb_1,0,1)[-1] 
        for ievt in range(len(fsum_1)):
            img = torch.transpose(mb_1[np.where(evtidc == ievt)],0,1).type(torch.float)
            deltaz = float(img[2].max() - img[2].min())
            erg_1[ievt] = float(fsum_1[ievt]) * 1./(1 - 2.76e-4*deltaz)
    else:
        erange_1 = [1.49, 1.57, 1.65, 1.73]
        erg_1 = fsum_1
    if len(mb_2) != 0:
        erange_2 = [1.55, 1.63, 1.72, 1.80]
        evtidc = torch.transpose(mb_2,0,1)[-1] 
        for jevt in range(len(fsum_2)):
            img = torch.transpose(mb_2[np.where(evtidc == jevt)],0,1).type(torch.float)
            deltaz = float(img[2].max() - img[2].min())
            erg_2[jevt] = float(fsum_2[jevt]) * 1./(1 - 2.76e-4*deltaz)
    else:
        erange_2 = [1.49, 1.57, 1.65, 1.73]
        erg_2 = fsum_2
       
    for ievt in range(len(fsum_1)):
        if (erange_1[0] < erg_1[ievt] < erange_1[1]) or (erange_1[2] < erg_1[ievt] < erange_1[3]):
            N_1 += 1
            for jevt in range(len(fsum_2)):
                if (erange_2[0] < erg_2[jevt] < erange_2[1]) or (erange_2[2] < erg_2[jevt] < erange_2[3]):
                    if int(N_1) == 1:
                        N_2 += 1
                    pd += ((feats_1[ievt] - feats_2[jevt])**2).mean()
    return pd,N_1,N_2
    
    

def log_batch(rank, epoch, batch_idx, data_size, local_loss, global_loss,
              local_accuracy, global_accuracy, home_dir, fieldnames,
              total_log_lines):
    filename = os.path.join(home_dir, 'training_rank{:03d}.csv'.format(rank))
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writerow(dict(time=time.time(), epoch=epoch, rank=rank,
                             batchno=batch_idx,
                             data_size=data_size,
                             local_loss=float("{:.6f}".format(local_loss)),
                             global_loss=float("{:.6f}".format(global_loss)),
                             local_acc=float("{:.6f}".format(local_accuracy)),
                             global_acc=float("{:.6f}".format(global_accuracy))
                             )
                        )
    # only print this to file if there will be less than 100 total lines
    # so twenty five per gpu
    last_epoch_last_batch = ((batch_idx * epoch) == total_log_lines)
    first_epoch_first_batch = ((batch_idx * epoch) == 0)
    batch_epoch_lt_25 = ((batch_idx * epoch) <= 25)
    if not batch_epoch_lt_25:
        batch_epoch_div_by_25 = \
            (batch_idx * epoch) % (total_log_lines // 25) == 0
    else:
        batch_epoch_div_by_25 = False
    if last_epoch_last_batch or first_epoch_first_batch or \
            batch_epoch_lt_25 or batch_epoch_div_by_25:
        logstring = "Epoch {epoch:02d} | Rank {rank:02d} | " \
            .format(epoch=epoch, rank=rank)
        logstring += "batch {batch_idx:03d} | # images {datasize:02d}" \
            .format(batch_idx=batch_idx, datasize=data_size)
        logstring += " | loc_loss {loc_loss:.6f} | glob_loss {loss:.6f}" \
            .format(loc_loss=local_loss, loss=global_loss)
        logstring += " | loc_acc {loc_acc:.6f} | glob_acc {acc:.6f}" \
            .format(loc_acc=local_accuracy, acc=global_accuracy)
        print(logstring)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx, train_loader, optimizer, train_set,
                         batches_per_allreduce, warmup_epochs,
                         base_lr):
    if epoch < warmup_epochs:
        epoch += float(batch_idx + 1) / float(train_set.n_batches)
        lr_adj = 1. / hvd.size() \
            * (epoch * (hvd.size() - 1) / warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr * hvd.size() * batches_per_allreduce \
            * lr_adj


class JNet(nn.Module):
    def __init__(self):
        super(JNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        self.fc1 = nn.Linear(256*4*4, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = nn.functional.elu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 5, 5)
        x = nn.functional.elu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 5, 5)
        x = nn.functional.elu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 5, 5)
        x = nn.functional.elu(self.conv4(x))
        x = nn.functional.max_pool2d(x, 5, 5)
        x = x.view(-1, 256*4*4)
        x = nn.functional.elu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


class JishNet(nn.Module):
    def __init__(self):
        super(JishNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, 4, padding=2)
        self.conv4 = nn.Conv2d(128, 256, 4, padding=2)
        self.fc1 = nn.Linear(256*9*9, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = nn.functional.elu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 5, 5)
        x = nn.functional.elu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 5, 5)
        x = nn.functional.elu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 4, 4)
        x = nn.functional.elu(self.conv4(x))
        x = nn.functional.max_pool2d(x, 4, 4)
        x = x.view(-1, 256*9*9)
        x = nn.functional.elu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class SCNet(nn.Module):
    def __init__(self):
        import sparseconvnet as scn
        super(SCNet, self).__init__()

        self.input_tensor = scn.InputLayer(dimension=2,
                                           spatial_size=(3600, 3600))
        self.elu = scn.ELU()
        self.conv1 = scn.SubmanifoldConvolution(dimension=2, nIn=1, nOut=10,
                                                filter_size=5, bias=False)
        self.conv2 = scn.SubmanifoldConvolution(dimension=2, nIn=10, nOut=64,
                                                filter_size=5, bias=False)
        self.conv3 = scn.SubmanifoldConvolution(dimension=2, nIn=64, nOut=128,
                                                filter_size=5, bias=False)
        self.conv4 = scn.SubmanifoldConvolution(dimension=2, nIn=128, nOut=256,
                                                filter_size=5, bias=False)
        self.maxp = scn.MaxPooling(dimension=2, pool_size=5, pool_stride=5)
        self.maxp2 = scn.MaxPooling(dimension=2, pool_size=4, pool_stride=4)

        N = 256
        self.sparse_to_dense = scn.SparseToDense(dimension=2, nPlanes=N)
        self.fc1 = nn.Linear(N*9*9, 32)
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.input_tensor(x)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.elu(x)
        x = self.maxp2(x)
        x = self.conv4(x)
        x = self.elu(x)
        x = self.maxp2(x)
        x = self.sparse_to_dense(x)
        x = x.view(-1, 256*9*9)
        x = nn.functional.elu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x
