import torch
import torch.nn as nn


class nnBaseModel(nn.Module):
    def __init__(self, faceData, label, netList):
        super(nnBaseModel, self).__init__()
        torch.set_default_dtype(torch.float64)

        self.data = faceData
        self.label = label
        self.layers = nn.Sequential()
        for i in range(len(netList) - 1):
            self.layers.add_module('lin-'+str(i), nn.Linear(netList[i], netList[i+1]))
            if i == len(netList) - 1:
                pass
            else:
                self.layers.add_module('sigm-'+str(i), nn.ReLU())

    def forward(self, x):
        ypred = self.layers(x)

        return ypred

    def loss_CrossEntropy(self):
        crossEntropyLossFunc = nn.CrossEntropyLoss()
        return crossEntropyLossFunc(self.forward(self.data), self.label)

    def train_adam(self, niteration=10, lr=0.005):  # here X is N*1 and Y is N*d
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.loss_CrossEntropy()

            loss.backward()
            optimizer.step()
            print('loss_nnl:', loss.item())

    def train_bfgs(self, niteration=10, lr=0.001):
        # LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)  # lr is very important, lr>0.1 lead to failure
        for i in range(niteration):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                # self.update()
                loss = self.loss_CrossEntropy()
                loss.backward()
                print('nll:', loss.item())
                return loss

            # optimizer.zero_grad()
            optimizer.step(closure)

