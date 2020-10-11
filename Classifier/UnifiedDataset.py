from torch.utils.data import Dataset

class UnifiedDataset(Dataset):

    def __init__(self, mnist, hasy):
        self.mnist = mnist
        self.hasy = hasy
        # self.ctr = 0

        self.hasy.data.symbol_id += 10 # shift symbols classes number to start after digits

    def __len__(self):
        return len(self.mnist) + len(self.hasy)

    def __getitem__(self, idx):
        # print(self.ctr, idx)
        # self.ctr += 1
        if idx>=len(self.mnist):
            return self.hasy.__getitem__(idx-len(self.mnist))
        return self.mnist.__getitem__(idx)