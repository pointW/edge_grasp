# python
import os

# torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# numpy
import numpy as np

f = os.path.dirname(__file__) + '/../data/train_normal_1.npy'
data_normal = np.load(f)
np.random.shuffle(data_normal)
test_normal, training_normal = data_normal[:20, :], data_normal[20:, :]

f = os.path.dirname(__file__) + '/../data/train_edge_1.npy'
data_edge = np.load(f)
np.random.shuffle(data_edge)
test_edge, training_edge = data_edge[:20, :], data_edge[20:, :]

training_data = map(lambda x: [x, 0], training_normal) + map(lambda x: [x, 1], training_edge)
print 'train data: ' + str(len(training_data))
np.random.shuffle(training_data)
test_data = map(lambda x: [x, 0], test_normal) + map(lambda x: [x, 1], test_edge)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


cnn = CNN()
criterion = F.cross_entropy
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)


def train(epoch):
    cnn.train()
    running_loss = 0.0
    for i, data in enumerate(training_data):
        inputs, labels = data
        inputs, labels = torch.FloatTensor(inputs), torch.LongTensor([labels])
        inputs = torch.unsqueeze(inputs, 0)
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if i % 20 == 19:
            print '[%d, %d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 20)
            running_loss = 0.0
    print 'finished'


def test():
    cnn.eval()
    test_loss = 0
    correct = 0
    for data, target in test_data:
        data, target = torch.FloatTensor(data), torch.LongTensor([target])
        data = torch.unsqueeze(data, 0)
        data, target = Variable(data, volatile=True), Variable(target)
        output = cnn(data)
        test_loss += criterion(output, target).data[0]
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).long().sum()
    test_loss /= len(test_data)
    print test_loss
    print correct


for epoch in range(4):
    train(epoch)
    test()

