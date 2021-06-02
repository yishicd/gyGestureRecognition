import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets


class HandsDataset(Dataset):
    def __init__(self):
        super(HandsDataset, self).__init__()
        imgs = []

        for ind, img in enumerate(os.listdir('dataset/hands')):
            img = cv2.imread('dataset/hands' + '/' + str(ind) + '.jpg', cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img) / 255

            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
            img = torch.tensor(img)
            img = img.to(torch.float32)
            imgs.append([int(ind / 10), img])

        self.len = len(imgs)
        self.imgs = imgs

    def __getitem__(self, item):
        return self.imgs[item][1], self.imgs[item][0]

    def __len__(self):
        return self.len


dataset = HandsDataset()
tran_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)


class HandsNetModel(torch.nn.Module):
    def __init__(self):
        super(HandsNetModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=11)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=5)
        self.conv4 = torch.nn.Conv2d(30, 40, kernel_size=3)

        self.pooling = torch.nn.MaxPool2d(2)


        self.fc = torch.nn.Linear(37440, 5)

    def forward(self, x):
        batch_size = x.size(0)

        x = torch.relu(self.pooling(self.conv1(x)))
        x = torch.relu(self.pooling(self.conv2(x)))
        x = torch.relu(self.pooling(self.conv3(x)))
        x = torch.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size, -1)
        # print(x.size())
        x = self.fc(x)
        return x


model = HandsNetModel()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    running_loss = 0.0
    for batch_ind, data in enumerate(tran_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # print(inputs)
        # print(inputs.size())
        # print(target)
        # print(type(target))
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':

    for epoch in range(100):
        train(epoch)
    torch.save({'net': model.state_dict()}, 'model_02.pt')
