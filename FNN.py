# I want to implement a fuzzy neural network
#いまのとこただのCNN

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from torchvision.transforms import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define Fuzzy Layer
class FuzzyLayer(nn.Module):
    def __init__(self, in_features, out_features, num_mfs):
        super(FuzzyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_mfs = num_mfs
        self.mfs = nn.Parameter(torch.rand((in_features, num_mfs)))
        self.fc = nn.Linear(in_features * num_mfs, out_features)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = torch.exp(-torch.sum((x - self.mfs)**2, dim=1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 画像データ識別のために，CNNをPyTorchで実装する
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(net, opt, criterion, num_epochs=10):
    """
    データの繰り返し処理を実行し、入力データをネットワークに与えて最適化します
    """
    # 学習経過を格納するdict
    history = {"loss":[], "accuracy":[], "val_loss":[], "val_accuracy":[]}

    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        valid_loss = 0.
        valid_acc = 0.
        train_total = 0
        valid_total = 0

        # 学習
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            opt.zero_grad() # 勾配情報をリセット
            pred = net(inputs)  # モデルから予測を計算(順伝播計算)：tensor(BATCH_SIZE, 確率×10)
            loss = criterion(pred, labels) # 誤差逆伝播の微分計算
            train_loss += loss.item() # 誤差(train)を格納
            loss.backward()
            opt.step()  # 勾配を計算
            _, indices = torch.max(pred.data, axis=1)  # 最も確率が高いラベルの確率と引数をbatch_sizeの数だけ取り出す
            train_acc += (indices==labels).sum().item() # labelsと一致した個数
            train_total += labels.size(0) # データ数(=batch_size)

        history["loss"].append(train_loss)  # 1epochあたりの誤差の平均を格納
        history["accuracy"].append(train_acc/train_total) # 正解数/使ったtrainデータの数

        # 学習ごとの検証
        with torch.no_grad():
            for data in trainloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                pred = net(inputs)  # モデルから予測を計算(順伝播計算)：tensor(BATCH_SIZE, num_class)
                loss = criterion(pred, labels) # 誤差の計算
                valid_loss += loss.item()  # 誤差(valid)を格納
                values, indices = torch.max(pred.data, axis=1)  # 最も確率が高い引数をbatch_sizeの数だけ取り出す
                valid_acc += (indices==labels).sum().item()
                valid_total += labels.size(0) # データ数(=batch_size)

        history["val_loss"].append(valid_loss)  # 1epochあたりの検証誤差の平均を格納
        history["val_accuracy"].append(valid_acc/valid_total) # 正解数/使ったtestデータの数
        # 5の倍数回で結果表示
        if (epoch+1)%5==0:
            print(f'Epoch：{epoch+1:d} | loss：{history["loss"][-1]:.3f} accuracy: {history["accuracy"][-1]:.3f} val_loss: {history["val_loss"][-1]:.3f} val_accuracy: {history["val_accuracy"][-1]:.3f}')
    return net, history



def plot_fig(history):
    plt.figure(1, figsize=(13,4))
    plt.subplots_adjust(wspace=0.5)

    # 学習曲線
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="valid")
    plt.title("train and valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    # 精度表示
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="valid")
    plt.title("train and valid accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()

    plt.show()



if __name__ == "main":
    # CIFAR-10データセットを使って，CNNを学習させる

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform, )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    criterion = nn.CrossEntropyLoss()
    # opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    opt = optim.Adam(model.parameters(), lr=0.001)
    net, history = train(net=model, opt=opt, criterion=criterion)
    plot_fig(history=history)

    dataiter = iter(trainloader)
    inputs, labels = next(dataiter)
    inputs, labels = inputs.to(device), labels.to(device)

    # 画像と正解ラベルの表示
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))