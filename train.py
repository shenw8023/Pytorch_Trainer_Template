from cpu import Trainer
from model import Net
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse
import time

from cpu import set_random_seed

set_random_seed(10)

"""
        model = ...
        optimizer = ...
        lr_scheduler = ...
        data_loader = ...
        # train 100 epochs
        trainer = Trainer(model, optimizer, lr_scheduler, data_loader, max_epochs=100)
        trainer.train()
"""

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()



device = torch.device("cuda:1")
model = Net()
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)




class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_one_iter(self):
        iter_start_time = time.perf_counter()
        lr_this_iter = self.lr

        start = time.perf_counter()
        data, target = next(self._data_iter_train)
        data, target = data.to(self.device), target.to(self.device)

        data_time = time.perf_counter() - start

        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss_dict = {"loss_dict":loss}


        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        # self._log_iter_metrics(loss_dict, data_time, time.perf_counter() - iter_start_time, lr_this_iter)

        # 记录任意指标，全都会被写入tensorboard，部分会被写入console
        self.log(self.cur_iter, lr=lr_this_iter, smooth=False)
        self.log(self.cur_iter, train_loss=loss.detach().cpu().item())




trainer = MyTrainer(model = model, 
                    optimizer=optimizer, 
                    lr_scheduler=scheduler, 
                    train_data_loader=train_loader, 
                    eval_data_loader=test_loader, 
                    max_epochs=30,
                    max_num_checkpoints=3,
                    eval_period = 200,
                    early_stop_patience = 2,
                    early_stop_monitor = "eval_acc",
                    device = device, 
                    )  


trainer.train()