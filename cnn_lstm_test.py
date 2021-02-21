from torch import nn
import torch.nn.functional as F
from dsdetectron.sequence import TextSequenceDataset
import torch


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.fc1 = nn.Linear(6 * 46 * 46, 120)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        x = self.conv1(x)
        print(x.size())
        x = x.view(-1, 6 * 46 * 46)
        x = self.fc1(x)
        return x


class TestCNNLSTM(nn.Module):
    def __init__(self):
        super(TestCNNLSTM, self).__init__()
        self.cnn = BasicCNN()
        self.rnn = nn.LSTM(120, 10, 2, batch_first=True)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out = self.rnn(r_in)
        return F.log_softmax(r_out[0], dim=1)


dset = TextSequenceDataset(
    "/local/ecw/deepscribe-detectron/data_nov_2020/hotspots_val.json",
    "/local/ecw/deepscribe-detectron/data_nov_2020/signs_nov_2020.csv",
    "/local/ecw/deepscribe-detectron/data_nov_2020/images_cropped",
)

imgs, labels = dset[0]

print(imgs.size())

model = TestCNNLSTM()

print(model(imgs).size())