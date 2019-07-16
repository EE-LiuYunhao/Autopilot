import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as tud
import pickle

BATCH_SIZE = 32

class AutoPilotCNN(nn.Module):
    def __init__(self):
        super(AutoPilotCNN, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(1,32,3,1,1),
             nn.ReLU(),
             nn.MaxPool2d(2)
         ) # Shape=(BATCH, 50, 50, 32)
        self.conv2 = nn.Sequential(
             nn.Conv2d(32,32,3,1,1),
             nn.ReLU(),
             nn.MaxPool2d(2)
         )# Shape=(BATCH, 25, 25, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# Shape=(BATCH, 12, 12, 64)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# Shape=(BATCH, 6, 6, 64)
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# Shape=(BATCH, 3, 3, 128)
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# Shape=(BATCH, 1, 1, 128)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1*1*128, 1024),
            nn.Linear(1024, 256),
            nn.Linear(256,64),
            nn.Linear(64,1)
        )# Shape = (BATCH,1)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.0001)
        self.loss_func = nn.MSELoss()
    def forward(self, x):
        x = torch.div(x,127.5)
        x = torch.add(x, -1.0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size()[0],-1)
        out = self.fc(x)
        return out, x
    
    def train(self, x_list, y_list):
        datasheet = tud.TensorDataset(x_list, y_list)
        loader = tud.DataLoader(
            dataset=datasheet,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 2
        )
        for step, (batch_x, batch_y) in enumerate(loader):
            out = self(batch_x)[0]
            loss = self.loss_func(out.view_as(batch_y),batch_y)
            if(step%50==0):
                test(out,batch_y)
                print('loss=',loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def test(out, batch_y):
    out = out.view_as(batch_y)
    print(out, '\n', batch_y)


def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels


def main():
    features, labels = loadFromPickle()
    features = features.reshape(features.shape[0], 100, 100, 1)
    
    features_tensor = torch.tensor(features, dtype=torch.float32).permute(0,3,1,2)
#   features_tensor = features_tensor.cuda()
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
#    labels_tensor = labels_tensor.cuda()

    cnn = AutoPilotCNN()
#    cnn = cnn.cuda()

    cnn.train(features_tensor,labels_tensor)

    torch.save(cnn,'Autopilot_V2.pk1')

if __name__ == '__main__':
    main()
