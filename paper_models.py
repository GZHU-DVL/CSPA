###
###This file include victim models on mnist, cifar10
###
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001
#########################MNIST#######################
class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def predict(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True).view(1, 1, 28, 28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict

#########################CIFAR10#######################
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG_plain(nn.Module):
    def __init__(self, vgg_name, nclass, img_width=32):
        super(VGG_plain, self).__init__()
        self.img_width = img_width
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

    def predict(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True).view(1, 3, 32, 32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        print(output)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        _, predict = torch.max(output.data, 1)
        return predict

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def predict(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True).view(1, 3, 32, 32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image, 0, 1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict


def load_mnist_data(test_batch_size=500):
    """ Load MNIST data from torchvision.datasets
        input: None
        output: minibatches of train and test sets
    """
    # MNIST Dataset
    test_dataset = dsets.MNIST(root='/home/ranyu/workspace/Project/FastSignOpt-mnist-Avgl2/data/mnist', download=True,
                               train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    return test_loader


def load_cifar10_data(test_batch_size=500):
    """ Load MNIST data from torchvision.datasets
        input: None
        output: minibatches of train and test sets
    """
    # CIFAR10 Dataset
    test_dataset = dsets.CIFAR10('/data/ranyu/dataset/CIFAR10', download=True,
                                 train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    return test_loader

def load_cifar100_data(test_batch_size=500):
    """ Load MNIST data from torchvision.datasets
        input: None
        output: minibatches of train and test sets
    """
    # CIFAR10 Dataset
    test_dataset = dsets.CIFAR100('/data/ranyu/dataset/CIFAR100', download=True,
                                 train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    return test_loader

def load_TinyImageNet_data(data_root=None, test_batch_size=500):
    test_dataset = dsets.ImageFolder(data_root, transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    return test_loader


def train_mnist(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, loss.data.item()))


def test_mnist(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct / total))


def cross_entropy(log_input, target):
    product = log_input * target
    loss = torch.sum(product)
    loss *= -1 / log_input.size()[0]
    return loss


def train_cifar10(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        if epoch % 10 == 0 and epoch != 0:
            lr = lr * 0.95
            momentum = momentum * 0.95
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, loss.data.item()))
    return model


def test_cifar10(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100.0 * correct / total))





def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)


def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))


if __name__ == '__main__':
    # print('train mnist' + '=' * 66)
    # train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    # net = MNIST()
    # if torch.cuda.is_available():
    #     net.cuda()
    #     # net = torch.nn.DataParallel(net)
    # train_mnist(net, train_loader)
    # save_model(net,'./models/mnist.pth')
    # test_mnist(net, test_loader)

    print('test mnist' + '=' * 66)
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    net = MNIST()
    if torch.cuda.is_available():
        net.cuda()
        # net = torch.nn.DataParallel(net)
    load_model(net, 'models/mnist.pth')
    test_mnist(net, test_loader)







