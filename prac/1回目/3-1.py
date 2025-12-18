import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(2, 16) #この数値を16から4にしたら精度は上がる？
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
def main():

    import numpy as np
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0], [1], [1], [0]])

    num_epochs = 10000

    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    net = Net()

    net.train()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # start to train
    epoch_loss = []
    for epoch in range(num_epochs):
        # forward
        outputs = net(x_tensor)

        # calculate loss
        loss = criterion(outputs, y_tensor)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save loss of this epoch
        epoch_loss.append(loss.data.numpy().tolist())

    print(net(torch.from_numpy(np.array([[0, 0]])).float()))  #tensor([[0.1525]], grad_fn=<SigmoidBackward0>)
    print(net(torch.from_numpy(np.array([[1, 0]])).float()))  #tensor([[0.7982]], grad_fn=<SigmoidBackward0>)
    print(net(torch.from_numpy(np.array([[0, 1]])).float()))  #tensor([[0.6603]], grad_fn=<SigmoidBackward0>)
    print(net(torch.from_numpy(np.array([[1, 1]])).float()))  #tensor([[0.3350]], grad_fn=<SigmoidBackward0>)

if __name__ == "__main__":
    main()
