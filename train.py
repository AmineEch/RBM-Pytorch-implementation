from model.RBM import *
from data.load_data import load_mnistdataloader
import torch.optim as optim
from tensorboardX import SummaryWriter

# Directories

dataset_dir = './data/data'
model_dir = './checkpoints'
# global values

n_v = 784
n_h = 1024
batch_size = 128
num_epochs = 10
gibbs_itter = 10

# Initiate model
rbm = RBM(n_v, n_h)
optimizer = optim.SGD(rbm.params,lr=0.001,momentum=0.9)

# TensorboardX
writer = SummaryWriter(log_dir=model_dir)

# dataset

mnist = load_mnistdataloader(dataset_dir)

def train_fn():
    global_step = 0
    for k in range(num_epochs):
        for i, batch in enumerate(mnist):
            batch[0] = batch[0].view([batch_size, -1])
            err_l2 = 0
            err_CE = 0
            for j in range(batch[0].shape[0]):
                x = batch[0][j,:].view([-1])
                if k == 0 and i == 0:
                    x_t = x
                    h_t = rbm.sample_h_knowing_x(x_t)
                    x_t, h_t = rbm.gibbs_sampling(gibbs_itter, x_t, h_t)
                else:
                    x_t, h_t = rbm.gibbs_sampling(gibbs_itter, x_t, h_t)
                err_l2 += torch.norm(x_t - x)
                err_CE += F.binary_cross_entropy(x_t,x)
                loss = rbm.forward(x, x_t, h_t)
                loss.backward()

            rbm.params[0].grad, rbm.params[1].grad, rbm.params[2].grad = rbm.params[0].grad / batch_size, \
                                                          rbm.params[1].grad / batch_size, rbm.params[2].grad / batch_size
            optimizer.step()
            optimizer.zero_grad()
            print("[*] reconstruction (error l2, CE) batch "+str(i)+" : "+str(err_l2/batch_size) + str(err_CE/batch_size))
            writer.add_scalar('L2',err_l2/batch_size,global_step)
            writer.add_scalar('CE', err_CE / batch_size, global_step)
            writer.add_image('sample', x.view([1,1,28,28]), global_step)
            writer.add_image('reconstruction', x_t.view([1,1,28,28]),global_step)
            writer.add_histogram('weights',rbm.params[0],global_step)
            global_step += 1

if __name__ == '__main__':
    train_fn()