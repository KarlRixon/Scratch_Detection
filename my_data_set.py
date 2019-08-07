from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


# train_data = MyDataset(txt='tail_train.txt', transform=transforms.ToTensor())
# data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
# print(len(data_loader))


def show_batch(imgs):
    grid = utils.make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


# for i, (batch_x, batch_y) in enumerate(data_loader):
#     if (i < 4):
#         print(i, batch_x.size(), batch_y.size())
#         show_batch(batch_x)
#         plt.axis('off')
#         plt.show()