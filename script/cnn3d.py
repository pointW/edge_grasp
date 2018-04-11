# python
import os

# torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

# numpy
import numpy as np

# pyplot
import matplotlib.pyplot as plt


f = os.path.dirname(__file__) + '/../data/npy/normal_3d.npy'
print f
data_normal = np.load(f)

f = os.path.dirname(__file__) + '/../data/npy/normal1_3d.npy'
data = np.load(f)
data_normal = np.vstack((data_normal, data))
print 'normal data: ' + str(data_normal.shape[0])

np.random.shuffle(data_normal)
test_normal, training_normal = data_normal[:100, :], data_normal[100:, :]


f = os.path.dirname(__file__) + '/../data/npy/edge_table1_3d.npy'
data_edge = np.load(f)

f = os.path.dirname(__file__) + '/../data/npy/edge_table2_3d.npy'
data = np.load(f)
data_edge = np.vstack((data_edge, data))

f = os.path.dirname(__file__) + '/../data/npy/edge_shelf_3d.npy'
data = np.load(f)
data_edge = np.vstack((data_edge, data))
print 'edge data: ' + str(data_edge.shape[0])

np.random.shuffle(data_edge)
test_edge, training_edge = data_edge[:100, :], data_edge[100:, :]


training_data = torch.cat((torch.FloatTensor(training_normal),
                           torch.FloatTensor(training_edge))).unsqueeze(1)
training_label = torch.cat((torch.zeros(training_normal.shape[0]),
                            torch.ones(training_edge.shape[0]))).type(torch.LongTensor)

test_data = torch.cat((torch.FloatTensor(test_normal),
                       torch.FloatTensor(test_edge))).unsqueeze(1)
test_label = torch.cat((torch.zeros(test_normal.shape[0]),
                        torch.ones(test_edge.shape[0]))).type(torch.LongTensor)

training_set = Data.TensorDataset(training_data, training_label)
training_loader = Data.DataLoader(
    dataset=training_set,
    batch_size=64,
    shuffle=True,
)

test_set = Data.TensorDataset(test_data, test_label)
test_loader = Data.DataLoader(
    dataset=test_set,
    batch_size=64,
)


class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv3d(32, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 6 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


cnn = CNN3D()
criterion = F.cross_entropy
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    step_each_ep = 0
    steps = []
    training_loss = []
    correct_rate = []
    test_loss = []
    eps = []
    ticks = []

    for ep in range(epoch):
        for step, (batch_x, batch_y) in enumerate(training_loader):
            cnn.train()
            inputs, labels = Variable(batch_x), Variable(batch_y)
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print '[%d, %d] loss: %.3f' % (ep + 1, step + 1, loss.data[0])

            if ep == 0:
                steps.append(step)
                step_each_ep += 1
            else:
                steps.append(step + ep * step_each_ep)
            training_loss.append(loss.data[0])
        print 'finished epoch ' + str(ep)
        c, t = test()
        correct_rate.append(c)
        test_loss.append(t)
        eps.append(ep+1)
        ticks.append(steps[-1])

    plot_train_test(steps, training_loss, eps, test_loss, correct_rate, ticks)


def test():
    cnn.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs, volatile=True), Variable(labels)
        outputs = cnn(inputs)
        test_loss += criterion(outputs, labels, size_average=False).data[0]
        prediction = outputs.data.max(1, keepdim=True)[1]
        correct += prediction.eq(labels.data.view_as(prediction)).long().cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / len(test_loader.dataset), test_loss


def plot_train_test(steps, training_loss, eps, test_loss, correct_rate, ticks):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    lns1 = ax1.plot(steps, training_loss, 'r', label='training loss', alpha=0.5)
    ax1.set_xlabel('steps')
    ax1.set_xlim(0, steps[-1])

    ax2.set_xlabel('episodes')
    ax2.set_xlim(0, eps[-1])
    lns2 = ax2.plot(eps, test_loss, 'x-', label='test loss')
    lns3 = ax2.plot(eps, correct_rate, 'x-', label='correct rate')

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='best')
    plt.title('training process of 3D CNN', y=1.09)

    plt.show()


if __name__ == '__main__':
    train(20)
