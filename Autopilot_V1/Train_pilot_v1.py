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
        self.conv1= nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3= nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fluc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*5*5, 128)
        )
        self.fluc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
        self.fluc3 = nn.Linear(64,1)
    def forward(self,x): #x is the input
        x = torch.div(x,127.5)
        x = torch.add(x, -1.0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0],-1)
        out = self.fluc1(x)
        out = self.fluc2(out)
        out = self.fluc3(out)
        return out,x #out is the output of entire NN, and x is the one for CNN layers

def train(model, x_list, y_list):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()
    #for batch
    datasheet = tud.TensorDataset(x_list, y_list)
    loader = tud.DataLoader(
        dataset = datasheet,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 2
    )
    for step, (batch_x, batch_y) in enumerate(loader): 
        out = model(batch_x)[0]
        loss = loss_func(out.view_as(batch_y),batch_y)
        if(step%50==0):
            test(out,batch_y)
            print("loss=",loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(out, batch_y):
    out = out.view_as(batch_y)
    print(out,'\n',batch_y)

def loadFromPickle():
    with open("features_40", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels

def augmentData(features, labels):
    features = np.append(features, features[:, :, ::-1], axis=0)
    labels = np.append(labels, -labels, axis=0)
    return features, labels


def main():
    features, labels = loadFromPickle()
    features, labels = augmentData(features, labels)
    features = features.reshape(features.shape[0], 40, 40, 1)

    features_tensor = torch.tensor(features, dtype=torch.float32).permute(0,3,1,2)
#    features_tensor = features_tensor.cuda()
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
#    labels_tensor = labels_tensor.cuda()

    cnn = AutoPilotCNN()
#    cnn.cuda()

    train(cnn, features_tensor, labels_tensor)

    torch.save(cnn, "Autopilot_V1.pk1")

if __name__ == "__main__": 
    main()
