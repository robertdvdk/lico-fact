import torch
import torch.nn.functional as F


def save_features(module, input, output):
    global features
    features = output.detach()


def save_gradients(module, input, output):
    global gradients
    gradients = output[0].detach()


def apply_grad_cam(model, target_layer, input_tensor):
    feature_handle = target_layer.register_forward_hook(save_features)
    gradient_handle = target_layer.register_backward_hook(save_gradients)

    output = model(input_tensor)
    feature_handle.remove()

    # get feature map
    _, predicted_classes = torch.max(output, dim=-1)
    class_scores = output.gather(1, predicted_classes.view(-1, 1)).squeeze()

    # get gradients
    model.zero_grad()
    class_scores.backward(torch.ones_like(class_scores))
    gradient_handle.remove()

    # global average pooling for gradients per channel
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    for i in range(features.shape[1]):
        features[:, i, :, :] *= pooled_gradients[:, i].view(-1, 1, 1)

    # mean over channels
    heatmap = torch.mean(features, dim=1).squeeze()
    # apply activation
    heatmap = F.relu(heatmap)
    # normalize to 0-1 range
    heatmap /= torch.max(heatmap)

    # interpolate to image size
    saliency_maps = F.interpolate(heatmap.unsqueeze(1), size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False).squeeze()
    return saliency_maps.detach()


def __init__():
    return

