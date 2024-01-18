import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math

def calculate_manifold_loss(A, B):
    return torch.mean(torch.sum(A*torch.log(A/B), dim=1))


def get_accuracy(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _, _ = model(inputs, targets=labels, mode='test')

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    return accuracy

def get_loss(model, dataloader, device, w_distance, alpha, beta):

    # get avg loss for given dataloader, use for e.g. validation loss

    model.eval()

    CELoss = nn.CrossEntropyLoss()

    running_loss = 0
    num_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # out: output logits, emb_matrix: similarity matrix of the feature maps
            # emb: unrolled feature maps, w_loss: OT loss
            # label_distribution: similarity matrix of the embedded prompts

            out, emb_matrix, w_loss, label_distribution = model(x, targets=y, w_distance=w_distance)

            # cross-entropy loss
            ce_loss = CELoss(out, y)

            # manifold loss
            m_loss = calculate_manifold_loss(label_distribution, emb_matrix)

            # get the full loss
            loss = ce_loss + alpha*m_loss + beta*w_loss

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss/num_batches
    return avg_loss

class CosineAnnealingStepLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                math.cos(7 * math.pi * self._step_count / (16 * self.T_max)) 
                for base_lr in self.base_lrs]
