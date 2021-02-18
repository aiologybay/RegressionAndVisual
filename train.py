import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(10)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1).to(device)
#print(x)
y = x.pow(3) + 0.1 * torch.randn(x.size()).to(device)
#print(y)

plt.cla()
plt.scatter(x.data.cpu(), y.data.cpu())
plt.pause(0.5)


class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.predict(out)
        return out


model = Net(1, 20, 1).to(device)
#print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()
epochs = 1000
#if not os.path.exists('checkpoint.pt'):
for t in range(1, epochs+1):
    prediction = model(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch:{},Loss:{:.4f}'.format(t, loss.data))
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': t,
        'loss': loss
    }, 'checkpoint.pt')
    plt.cla()
    plt.scatter(x.data.cpu(), y.data.cpu())
    plt.plot(x.data.cpu(), prediction.data.cpu(), 'r-', lw=5) 
    plt.text(0.5, 0, 'Loss={:.4f}'.format(loss.data), fontdict={'size': 20, 'color': 'red'})
    if t % 10 == 0:
        plt.savefig('./visual/train_epoch_{}.png'.format(t))
        print('Epoch {} done! Saving train_epoch_{}.png!'.format(t, t))
    plt.pause(0.01) 
plt.ioff()
plt.pause(3)
'''
else:
    checkpoint=torch.load('checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch=checkpoint['epoch']
    print('Success to load epoch {}!'.format(start_epoch))
for i in range(start_epoch+1, epochs):
    prediction = model(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch:{},Loss:{:.4f}'.format(i, loss.data))
    plt.cla()
    plt.scatter(x.data.cpu(), y.data.cpu())
    plt.plot(x.data.cpu(), prediction.data.cpu(), 'r-', lw=5)
    plt.text(0.5, 0, 'Loss={:.4f}'.format(loss.data), fontdict={'size': 20, 'color': 'red'})
    plt.savefig('train_epoch_{}.png'.format(i))
plt.ioff()
plt.pause(3)
'''
