import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time

DEVICE = None

def parse_args():
    parser = argparse.ArgumentParser(description='Simple random search implementation')
    parser.add_argument('dataset', type=str,
                        help='name of dataset')
    parser.add_argument('net', type=str, choices=['mobile', 'inception', 'shuffle', 'rex', 'dense'],
                        help='network architecture')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size to use (default 8)')
    parser.add_argument('--q', action='store_false',
                        help='Disables input-gated warnings')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default 0.01)')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='SGD momentum (default 0.9)')
    return parser.parse_args()

def get_accuracy(model, loader, specific_class=None) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if specific_class is None:
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            else:
                for predict, label in zip(predicted, labels):
                    if label == specific_class:
                        total += 1
                        if label == predict:
                            correct += 1
    model.train()
    return correct / total


def init_datasets(dataset: str, args):
    transformations = transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(80),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join('datasets', dataset, 'training'), transform = transformations)
    test_set = datasets.ImageFolder(os.path.join('datasets', dataset, 'test'), transform = transformations)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    return train_loader, test_loader


def init_results_file(args) -> str:
    if not os.path.exists('results/clean'):
        os.mkdir('results/clean')


    if not os.path.exists(os.path.join('results/clean', args.net)):
        os.mkdir(os.path.join('results/clean', args.net))

    results_fp = os.path.join('results/clean', args.net, f'{args.dataset}.csv')
    if args.q and os.path.exists(results_fp):
        input('Warning. Results File Exists [return to continue]')

    with open(results_fp, 'w') as results_f:
        results_f.write('Batches, Training Accuracy, Test Accuracy \n')

    return results_fp


def train_backdoor(dataset, epochs, args):
    train_loader, test_loader = init_datasets(dataset, args)
    results_fp = init_results_file(args)

    if args.net == 'mobile':
        net = models.mobilenet_v2(pretrained=False, progress=False).to(DEVICE)
    elif args.net == 'shuffle':
        net = models.shufflenet_v2_x1_0(pretrained=False, progress=False).to(DEVICE)
    elif args.net == 'inception':
        net = models.inception_v3(pretrained=False, progress=False).to(DEVICE)
    elif args.net == 'rex':
        net = models.resnext50_32x4d(pretrained=False, progress=False).to(DEVICE)
    elif args.net == 'dense':
        net = models.densenet121(pretrained=False, progress=False).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.gamma)

    time_start = time.time()
    print('Training Started...')
    best_test_acc = 0.0
    loss = 0.0
    for epoch in range(epochs):
        running_accuracy = []
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_accuracy.append((labels == predicted).sum().item())

            #Report stats
            running_loss += loss.item()
            if i % 1000 == 999:
                with torch.no_grad():
                    test_accuracy = get_accuracy(net, test_loader)

                with open(results_fp, 'a') as results_f:
                    results_f.write(f'{(i + 1) * (epoch + 1)},{np.mean(running_accuracy) / args.batch_size},{test_accuracy}\n')

                print(f'Running: {results_fp}  Time: {time.time() - time_start}')
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                print(f'Train: {np.mean(running_accuracy) / args.batch_size} Test: {test_accuracy}')

                if test_accuracy > best_test_acc:
                    torch.save(net.state_dict(), os.path.join('results/clean', args.net, f'{dataset}_best.p'))
                    best_test_acc = test_accuracy

    torch.save(net.state_dict(), os.path.join('results/clean', args.net, f'{dataset}_final.p'))
    print('Finished Training')

    with open(results_fp, 'a') as results_f:
        results_f.write(f'\n\n# Time Passed: {time.time() - time_start}')


def main():
    global DEVICE
    args = parse_args()
    DEVICE = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    train_backdoor(args.dataset, args.epochs, args)

if __name__ == '__main__':
    main()