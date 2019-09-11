import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.autograd import Variable
import torch.nn.functional as F

# unsqueeze 将一维数组转化为二维数组
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data(tensor), shape(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # noisy data(tensor), shape(100,1)

# torch can only train on Variable, so convert then to Variable
# The code below id deprecated in Pytorch 0.4. Now, autograd dirctly supports tensors

# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)

        return x


net = Net(n_feature=1, n_hidden=10, n_output=1) # define the network
print(net)

plt.ion() # something about plotting
# plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x) # input x and predict based on x

    loss = loss_func(prediction, y) # must be (1. nn output, 2. target)

    optimizer.zero_grad() # clear gradientd for next train
    loss.backward()       # backpropration, compute gradients
    optimizer.step()      # apply gradients

    if t%5 == 0:
        # plot and show learning proces
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy(), fontdict={'size':20,'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()







