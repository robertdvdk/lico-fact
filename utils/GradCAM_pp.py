import torch
import torch.nn.functional as F


def save_features(module, input, output):
    global features
    features = output.detach()


def save_gradients(module, input, output):
    global gradients
    gradients = output[0].detach()


def apply_grad_cam_pp(model, target_layer, input_tensor):
    forward_handle = target_layer.register_forward_hook(save_features)
    backward_handle = target_layer.register_backward_hook(save_gradients)

    output = model.get_logits(input_tensor)
    forward_handle.remove()

    _, predicted_classes = torch.max(output, dim=1)
    class_scores = output.gather(1, predicted_classes.view(-1, 1)).squeeze()

    model.zero_grad()
    class_scores.backward(torch.ones_like(class_scores))
    backward_handle.remove()

    grad_pow_2 = gradients ** 2
    grad_pow_3 = grad_pow_2 * gradients

    global_sum = features.view(features.size(0), features.size(1), -1).sum(dim=2).view(features.size(0), features.size(1), 1, 1)

    eps = 0.000001

    alpha_num = grad_pow_2
    alpha_denom = grad_pow_2 * 2 + global_sum * grad_pow_3 + eps
    alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    alpha_normalization_constant = torch.sum(alphas, dim=(2, 3), keepdim=True)
    alphas /= alpha_normalization_constant
    weights = torch.sum(alphas * F.relu(gradients), dim=(2, 3), keepdim=True)
    #print(weights)

    #aij = grad_pow_2 / (grad_pow_2 * 2 + global_sum * grad_pow_3 + eps)
    #aij = torch.where(gradients != 0, aij, torch.zeros_like(aij))
    #weights = torch.clamp_min(gradients, 0) * aij
    #weights = torch.sum(weights, dim=(2, 3), keepdim=True)
    #print(weights)

    grad_cam_map = torch.sum(weights * features, dim=1)
    grad_cam_map = F.relu(grad_cam_map)
    grad_cam_map = F.interpolate(grad_cam_map.unsqueeze(1), input_tensor.shape[2:], mode='bilinear', align_corners=False)

    saliency_maps = grad_cam_map.squeeze()

    return saliency_maps


def __init__():
    return
