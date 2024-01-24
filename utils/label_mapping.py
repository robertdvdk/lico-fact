import torch

def euclidean_distance(A):
    """Calculates the Euclidean distance between samples in a tensor A.

    Args:
        A (torch.Tensor): A tensor of shape (batch_size, num_channels or num_context, feature_dim). This is either
            F_B or G_B, where F_B = [F_1, ..., F_n] (paper section 3.1). Here, n denotes the number of samples in the
            batch. Similarly, G_B = [G_1, ..., G_n].

    Returns:
        """
    # Convert the tensor to 2D, and then calculate the pairwise Euclidean distances between samples in the batch.
    # Calculating the Euclidean distance between row vectors in a 2D tensor in this way is equivalent to calculating the
    # Euclidean distance between the matrices in a 3D tensor.
    A = A.view(A.shape[0], -1)
    return torch.cdist(A, A, p=2.0)

