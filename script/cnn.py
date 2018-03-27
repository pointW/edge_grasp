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
test_edge, training_edge = data_edge[:20, :], data_edge[20:, :]
DATA = map(lambda x: [x, 0], training_normal) + map(lambda x: [x, 1], training_edge)
np.random.shuffle(DATA)


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


for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(DATA):
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
            print '[%d, %d] loss: %.3f' % (epoch+1, i+1, running_loss / 20)
            running_loss = 0.0

print 'finished'