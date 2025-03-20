# Run Optimization for CNN
import torch.optim as optim
import torch

dtype = torch.float
device = torch.device("cuda:0")  # Uncommon this to run on GPU
# device = torch.device("cpu") # Uncommon this to run on CPU
from run_nn import Net, model
import torch.nn as nn
import numpy as np
import sim
import random
import warnings
warnings.filterwarnings("ignore")
import os
import sys

sys.path.append(os.getcwd())
set_seed = 0
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed(set_seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_model = True
load_model = False


class run_optim:
    def __init__(self, alpha, miu, X_y_set):
        self.optimizer = optim.SGD(model.parameters(), lr=alpha, momentum=miu, weight_decay=0)
        self.alpha = alpha
        self.miu = miu
        self.X_train = X_y_set[0]
        self.y_train = X_y_set[1]
        self.X_test = X_y_set[2]
        self.y_test = X_y_set[3]
        self.batch_size = round(len(self.X_train) / 1.1)

    @staticmethod
    def loss_function(x):
        if x == 'MSE':
            return nn.MSELoss()
        elif x == 'cross entropy':
            return nn.CrossEntropyLoss()

    def model_train(self):
        optimizer = self.optimizer
        batch_size = self.batch_size
        X_train = self.X_train
        y_train = self.y_train

        print("Batch Size is " + str(self.batch_size))

        epochs = 1
        for j in range(epochs):
            print("Processing Epoch " + str(j) + " out of Epoches " + str(epochs-1))
            for i in range(0, len(X_train) - batch_size + 1):
                print("Processing Batch " + str(i) + " out of Batch " + str(len(X_train) - batch_size))
                optimizer.zero_grad()
                loss = run_optim.loss_function('MSE')(model(X_train[i:i + batch_size]), y_train[i:i + batch_size])
                loss.backward()
                optimizer.step()

        print("Loss for training set is")
        print(loss)

        if save_model:
            file_path = os.getcwd() + '/pretrained/'
            if not os.path.exists(file_path):
                os.makedirs(file_path)


            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'epochs': epochs}
            torch.save(checkpoint, file_path + str(self.miu) + '.pth')

        #print("Model's state_dict:")
        #for param_tensor in model.state_dict():
            #print(param_tensor, "\t", model.state_dict()[param_tensor])

        #print("Optimizer's state_dict:")
        #for var_name in optimizer.state_dict():
            #print(var_name, "\t", optimizer.state_dict()[var_name])

    def model_test(self):
        if not load_model:
            run_optim.model_train(self)
        else:
            file_path = os.getcwd() + '/pretrained/' + str(self.miu) + '.pth'

            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model'])
            model.eval()

        X_test = self.X_test
        y_test = self.y_test

        correct = 0
        with torch.no_grad():
            for i in range(0, len(X_test)):
                y_hat = model(X_test[i].view(-1, 1, N, time_step + 1))
                y = y_test[i]
                if torch.argmax(y_hat) == torch.argmax(y):
                    correct += 1

        print("The accuracy is")
        print(correct / len(y_test))


        return correct / len(y_test)



if __name__ == "__main__":
    time_step = 252
    # Number of Simulations
    N = 500
    # Number of Samples
    M = 500
    ls = []
    X = torch.Tensor([])
    y = torch.Tensor([])
    random.seed(0)
    for i in range(M):
        p = random.random()
        if p < 0.5:
            obj = sim.BS_asset_sim(miu=0.05,
                                sigma=np.array([0, 0.2]),
                                rho=0,
                                J_apt=[random.uniform(-0.01, 0), random.uniform(0, 0.01)],
                                _lambda=5, S_0=400, T=1, t=0, time_step=time_step, N=N)
            X = torch.cat((X, torch.Tensor(obj.exact()).view(-1, 1, N, time_step + 1)), 0)
            y = torch.cat((y, torch.Tensor([1, 0]).view(1, 2)), 0)
        else:
            obj = sim.BS_asset_sim(miu=0.05,
                                sigma=np.array([0, 0.2]),
                                rho=0,
                                J_apt=[random.uniform(-1, 0), random.uniform(0, 1)],
                                _lambda=5, S_0=400, T=1, t=0, time_step=time_step, N=N)
            X = torch.cat((X, torch.Tensor(obj.exact()).view(-1, 1, N, time_step + 1)), 0)
            y = torch.cat((y, torch.Tensor([0, 1]).view(1, 2)), 0)

    train_split = round(len(X) * 0.7)
    X_train, X_test = X[:train_split], X[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]
    X_y_set = X_train, y_train, X_test, y_test
    run_optim(alpha=0.05, miu=0.75, X_y_set=X_y_set).model_test()
