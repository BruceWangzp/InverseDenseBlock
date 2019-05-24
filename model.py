import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transform
import torchvision.datasets as datasets
import os
import time
# 导入数据
data_dir = 'InverseDenseBlock/data/hymenoptera_data'
data_transform = transform.Compose([transform.Resize(256), transform.RandomResizedCrop(224),
                                    transform.RandomHorizontalFlip(), transform.ToTensor(),
                                    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transform)
# print(image_datasets)
batch_size = 32
dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
data_transform_val = transform.Compose([transform.Resize(256), transform.RandomResizedCrop(224),transform.RandomHorizontalFlip(),transform.ToTensor(),
                                    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
image_datasets_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transform_val)

dataloader_val = torch.utils.data.DataLoader(image_datasets_val, batch_size=batch_size, shuffle=True, num_workers=0)
print(len(image_datasets))
dropout_rate =0.5
# 定义基本的层
class inverted_layer(nn.Module):
    def __init__(self, in_channels, growth_rate=12, stride=1):
        super(inverted_layer, self).__init__()
        self.stride = stride
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1),
                                     nn.BatchNorm2d(4 * growth_rate),
                                     nn.ReLU(True))
        self.dwconv3x3 = nn.Sequential(nn.Conv2d(4 * growth_rate, 4 * growth_rate, kernel_size=3,
                                                 stride=self.stride, padding=1, groups= 4 * growth_rate),
                                       nn.BatchNorm2d(4 * growth_rate),
                                       nn.ReLU(4 * growth_rate))
        self.conv1x1_2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1x1(x)

        x = self.dwconv3x3(x)

        x = self.conv1x1_2(x)
        if self.stride == 2:
            identity = self.avgpool(identity)

            out  = torch.cat((x, identity), 1)
            out = self.channel_shuffle(out)
            return out
        out = torch.cat((x, identity), 1)
        out = self.channel_shuffle(out)

        return out

    def channel_shuffle(self, x):
        n = x.size(1) // 2
        x1 = x
        x = x.view(x1.size(0), 2, n, x1.size(2), x1.size(3))
        x = x.transpose(1, 2)
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        x = x.contiguous()
        x = x.view(x1.size(0), -1, x1.size(2), x1.size(3))
        return x

class inverted_block(nn.Module):
    def __init__(self, in_channels, growth_rate, num_repeate):
        super(inverted_block, self).__init__()
        self.block = []
        for i in range(num_repeate):
            if i == 0:
                self.block.append(inverted_layer(in_channels + i * growth_rate, growth_rate, stride=2))
            if i > 0:
                self.block.append(inverted_layer(in_channels + i * growth_rate, growth_rate, stride=1))
        self.Block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.Block(x)
        return x

class InvertedDenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(InvertedDenseNet, self).__init__()
        self.in_channels = in_channels
        self.feature = nn.Sequential(OrderedDict([('Conv1', nn.Conv2d(3, self.in_channels, kernel_size=3, stride=2, padding=1)),
                                                  ('BN1', nn.BatchNorm2d(self.in_channels)),
                                                  ("relu1", nn.ReLU(True)),
                                                  ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

        cfg = [2, 4, 2]
        for i, num in enumerate(cfg):
            self.feature.add_module('InvertedDense%d' % i, inverted_block(self.in_channels, growth_rate, num))
            self.in_channels = self.in_channels + growth_rate * num
            if i < len(cfg)-1:
                self.feature.add_module('bottleneck%d' % i,
                                        nn.Conv2d(self.in_channels, self.in_channels + growth_rate, kernel_size=1))
                self.in_channels = self.in_channels + growth_rate
                self.feature.add_module('bottleneck%d_BN' % i, nn.BatchNorm2d(self.in_channels))
                self.feature.add_module('bottlebeck%d_relu' % i, nn.ReLU(True))

            if i == len(cfg)-1:
                self.feature.add_module('bottleneck%d' % i,
                                        nn.Conv2d(self.in_channels, 512, kernel_size=1))
                self.feature.add_module('bottleneck%d_BN' % i, nn.BatchNorm2d(512))
                self.feature.add_module('bottlebeck%d_relu' % i, nn.ReLU(True))

        self.feature.add_module('avgpool', nn.AvgPool2d(kernel_size=7))
        self.fc1 = nn.Linear(512, 512)
        self.fc = nn.Linear(512,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, dropout_rate)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, dropout_rate)
        out = torch.sigmoid(self.fc(x))
        return out

model = InvertedDenseNet(64, 12)
# print(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
lr = 0.01


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01
    lr = lr * (0.1 ** (epoch // 200))
    if epoch%200==0:
      print('The learning_rate of epoch %d:' % epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
weight_decay=0.0005
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoches = 500
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
# print('weight decay:', weight_decay)
print('optimizer: Adam')
print('dropout_rate:', dropout_rate)
print('batchsize:', batch_size)
# Train
for i in range(epoches):
    since = time.time()
    model.train()
#     if i%50==0:
#         adjust_learning_rate(optimizer, i)
    running_loss = 0.0
    for inputs, labels in dataloader:
        labels = labels.float().view(labels.size(0), -1)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    time_elapsed = time.time() - since
    if i==0:
      print("epoch time", time_elapsed, 's')

      
    model.eval()
    eval_loss=0
    for inputs_eval, labels_eval in dataloader_val:
        labels1 = labels_eval.float().view(labels_eval.size(0), -1)
        inputs_eval, labels1 = inputs_eval.to(device), labels1.to(device)
        output1 = model(inputs_eval)
        loss1 = criterion(output1, labels1)
        eval_loss += loss1.item()
        
    if i%10==0:
      print('epoch: %d Loss:' % i, running_loss/(698/batch_size))
      print('epoch: %d eval_Loss:' % i, eval_loss/20)
    


      correct = 0
      total = 0

      with torch.no_grad():
          for inputs, labels in dataloader:
              inputs, labels = inputs.to(device), labels.to(device)
              output = model(inputs).squeeze()
              labels = labels.byte()
              output = (output > 0.5)
              correct += (output == labels).sum().item()
              total += labels.size(0)
          print('correct:', correct, 'total:', total,'Accuracy of train: ', (correct/total))


          model.eval()
          correct = 0
          total = 0
          for i in range(10):
            for inputs_val, labels_val in dataloader_val:
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                output = model(inputs_val).squeeze()
                labels = labels_val.byte()
                output = (output > 0.5)
                correct += (output == labels).sum().item()
                total += labels.size(0)
          print('correct:', correct, 'total:', total,'Accuracy of val: ', (correct/total))
       
