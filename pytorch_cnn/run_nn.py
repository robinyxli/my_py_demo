# Run Convolutional Neural Network
import torch
dtype = torch.float
device = torch.device("cuda:0") # Uncommon this to run on GPU
# device = torch.device("cpu") # Uncommon this to run on CPU
import torch.nn as nn
import torch.nn.functional as F

set_seed = 0
import numpy as np
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channel_1 = 50
        self.out_channel_2 = 10
        self.out_channel_3 = 5
        self.kernel_size_1 = (300, 200)
        self.kernel_size_2 = 50
        self.kernel_size_3 = (2, 2)
        self.conv1 = nn.Conv2d(1, self.out_channel_1, self.kernel_size_1)
        self.conv2 = nn.Conv2d(self.out_channel_1, self.out_channel_2, self.kernel_size_2)
        self.conv3 = nn.Conv2d(self.out_channel_2, self.out_channel_3, self.kernel_size_3)


    @staticmethod
    def activation_func(type, x):
        if type == 'tanh':
            return F.max_pool2d(F.tanh(x), (2,1))
        elif type == 'relu':
            return F.max_pool2d(F.relu(x), (2,1))
        elif type == 'sigmoid':
            # return F.max_pool2d(F.sigmoid(x), (2,1))
            return F.sigmoid(x)

    def forward(self, x):
        x = self.activation_func('sigmoid',self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        m1 = nn.Linear(x[0].shape[0], 25)
        m2 = nn.Linear(25, 2)
        x = F.relu(m1(x))
        x = m2(x)
        x = F.softmax(x, dim=1)
        return x

model = Net()