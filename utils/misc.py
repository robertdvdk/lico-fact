import torch

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
