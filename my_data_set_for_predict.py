from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def default_loader(path):
    return Image.open(path).convert('L')

# 获取一个图片的地址，返回48+35个大小为64*64的裁剪子图，原图label在调用MyDataset前已经获取
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        coords = np.zeros(shape=(48+35, 2))
        side = int(618 / 6)
        for index in range(48+35):
            if (index + 1) < 49:
                y = int(index / 8) * side
                x = (index % 8) * side
                coords[index][0] = x
                coords[index][1] = y
            else:
                y = int((index - 48) / 7) * side + int(side / 2)
                x = ((index - 48) % 7) * side + int(side / 2)
                coords[index][0] = x
                coords[index][1] = y

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        img = self.loader(txt)
        self.img = img
        self.coords = coords
        self.side = side

    def __getitem__(self, index):
        x, y = self.coords[index][0], self.coords[index][1]
        tail = transforms.functional.crop(self.img, y, x, self.side, self.side)
        if self.transform is not None:
            tail = self.transform(tail)
        return tail

    def __len__(self):
        return self.coords.shape[0]


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