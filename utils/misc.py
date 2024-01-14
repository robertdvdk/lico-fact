import torch
import torch.nn as nn
from models.modules.sinkhorn_distance import SinkhornDistance

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
            outputs, _, _, _ = model(inputs, targets=labels, mode='test')

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    return accuracy

def get_loss(model, dataloader, args, device):

    # get avg loss for given dataloader, use for e.g. validation loss

    model.eval()

    CELoss = nn.CrossEntropyLoss()
    w_distance = SinkhornDistance(args.sinkhorn_eps, args.sinkhorn_max_iters)

    running_loss = 0
    num_batches = 0

    with torch.no_grad():

        for batch in dataloader:

            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # out: output logits, emb_matrix: similarity matrix of the feature maps
            # emb: unrolled feature maps, w_loss: OT loss
            # label_distribution: similarity matrix of the embedded prompts

            out, emb_matrix, emb, w_loss, label_distribution = model(x, targets=y, w_distance=w_distance)

            # cross-entropy loss
            ce_loss = CELoss(out, y)

            # manifold loss
            m_loss = calculate_manifold_loss(label_distribution, emb_matrix)

            # get the full loss

            loss = ce_loss + args.alpha*m_loss + args.beta*w_loss

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss/num_batches
    return avg_loss

