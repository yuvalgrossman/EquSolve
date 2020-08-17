import torch.nn as nn
import torch.nn.functional as F

# define the neural net class
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20,
                               kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=1800, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=1401)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x));  # print(x.shape)
        x = F.max_pool2d(x, 2, 2);  # print(x.shape)
        x = F.relu(self.conv2(x));  # print(x.shape)
        x = F.max_pool2d(x, 2, 2);  # print(x.shape)
        x = x.view(x.size(0), -1);  # print(x.shape)
        x = F.relu(self.fc1(x));  # print(x.shape)
        x = self.fc2(x);  # print(x.shape)
        return x

if __name__ == "__main__":
    net = SimpleClassifier()
    print(net)
