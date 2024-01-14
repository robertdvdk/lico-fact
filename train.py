
import torch
import torch.nn as nn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
from models.wideresnet_prompt import *
from models.modules.sinkhorn_distance import SinkhornDistance


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_manifold_loss(A, B):

    return torch.mean(torch.sum(A*torch.log(A/B), dim=1))


def test_accuracy(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _, _, _ = model(inputs, targets=labels, mode='test')

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    return accuracy

wrn_builder = build_WideResNet(1, 10, 2, 0.01, 0.1, 0.5)
wrn = wrn_builder.build(10)
wrn = wrn.to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# for testing purposes, just take a small fraction of the data
'''
subset_size = int(0.2 * len(trainset))
small_trainset, _ = torch.utils.data.random_split(trainset, [subset_size, len(trainset) - subset_size])
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training loop

num_epochs = 10
w_distance = SinkhornDistance(0.1, 1000)

CELoss = nn.CrossEntropyLoss()
optimizer = SGD(wrn.parameters(), lr=0.005, momentum=0.9)
alpha = 10
beta = 1

alpha = 0
beta = 0

total_loss = 0

for epoch in range(num_epochs):

    for batch in trainloader:

        x, y = batch
        x, y = x.to(device), y.to(device)
        
        # out: output logits, emb_matrix: similarity matrix of the feature maps
        # emb: unrolled feature maps, w_loss: OT loss
        # label_distribution: similarity matrix of the embedded prompts

        out, emb_matrix, emb, w_loss, label_distribution = wrn(x, targets=y, w_distance=w_distance)

        # cross-entropy loss
        ce_loss = CELoss(out, y)

        # manifold loss
        m_loss = calculate_manifold_loss(label_distribution, emb_matrix)

        # get the full loss

        loss = ce_loss + alpha*m_loss + beta*w_loss

        # training step
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {round(total_loss, 3)}")

    test_acc = test_accuracy(wrn, testloader, device)
    print(f"Epoch {epoch+1} test accuracy: {round(test_acc, 3)}")

    total_loss = 0
        
