from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        #initializing batch normalization after Fully Connected Layer 1
        self.dropout1 = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        #Adding a new Fully Connected layer 
        self.newfc = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.newbn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = self.pool(x)
    #     x = F.relu(self.conv3(x))
    #     x = F.relu(self.conv4(x))
    #     x = self.pool(x)
    #     x = x.view(-1, self.num_flat_features(x))
    #     x = F.relu(self.fc1(x))
    #     #adding Batch Normalization after FC1
    #     x = self.bn1(x)
    #     #new FC layer after Batch normalization
    #     x = self.newfc(x)
    #     x = self.fc2(x)
    #     return x
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #adding Batch Normalization after FC1
        x = self.bn1(x)
        #new FC layer after Batch normalization
        x = self.newfc(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def plot(self, x_axis, train_accuracy, test_accuracy, x_label, llegend, llegendy, xstep = 1.0, file = None):
        
        #fig, axes = plt.subplots()
        #fig.canvas.draw()

        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, xstep))

        plt.plot(x_axis, train_accuracy, '-r', label='Training')
        plt.plot(x_axis, test_accuracy, '-b', label='Test')
        
        #labels = [item.get_text() for item in axes.get_xticklabels()]
        #labels = ['1e-1', '5e-2', '3e-2', '2e-2', '1e-2', '7e-3', '5e-3', '3e-3', '1e-3', '1e-4']
        #labels = ['250', '500', '1000', '2000', '2500']
        #axes.set_xticklabels(labels)
        
        plt.ylabel(llegendy)
        plt.xlabel(x_label)
        plt.title(llegend)
        plt.legend(loc='best')
        if file is not None:
            plt.savefig('figures/%s' % file, bbox_inches='tight')
        plt.gcf().clear()


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(size_average=False)
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.data[0]
    net.train() # Why would I do this?
    return total_loss / total, correct / total

if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 22 #maximum epoch to train

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(size=[32,32], padding=4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    #pretrained_dict = torch.load('mytraining1024.pth')
    net = Net().cuda()
    #net.load_state_dict(pretrained_dict, strict = False)
    net.train() # Why would I do this?
    
    
    #model_dict = net.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #model_dict.update(pretrained_dict)
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    #Using ADAM Optimizer
    #Adding l2 regularization using weight_decay
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    training_acc = []
    testing_acc = []
    training_loss = []
    testing_loss = []
    x_axis = []

    print('Start training...')
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        if(epoch % 5 == 0):
            training_acc.append(train_acc)
            testing_acc.append(test_acc)
            training_loss.append(train_loss)
            testing_loss.append(test_loss)
            x_axis.append(epoch)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'mytraining1024.pth')
    net.plot(x_axis, training_acc, testing_acc, 'Number of Epochs', 'Accuracy vs #Epochs [Using Data Augmentation using RandomHorizontalFlip]', 'Accuracy', 5.0, 'bestaccuracy1.png')
    net.plot(x_axis, training_loss, testing_loss, 'Number of Epochs', 'Accuracy vs #Epochs [Using Data Augmentation using RandomHorizontalFlip]', 'Loss', 5.0, 'bestloss1.png')

#https://pytorch.org/tutorials/beginner/saving_loading_models.html
#https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
#http://cs231n.github.io/transfer-learning/
#https://jhui.github.io/2017/03/16/CNN-Convolutional-neural-network/
#Activation Functions: https://pytorch.org/docs/0.3.1/nn.html