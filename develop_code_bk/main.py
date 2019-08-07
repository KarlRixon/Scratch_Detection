import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from resnet import ResNet18
from torch import nn, optim

def main():
    batchsz = 32
    cifar_train = datasets.CIFAR10('./cifar/', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.RandomRotation(5, 10, 15)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('./cifar/', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.RandomRotation(5, 10, 15)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)
    print(model)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1000):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # x: [b, 3, 32, 32]
            # label[b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits: [b, 10]
            # label: [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch: ', epoch, 'loss: ', loss.item())

        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # x: [b, 3, 32, 32]
                # label[b]
                x, label = x.to(device), label.to(device)
                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] -> scalar tensor
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print('acc: ', acc)

if __name__ == '__main__':
    main()