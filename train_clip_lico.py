import argparse
import torch
import clip
from datasets import ImageNetDataLoader
from models.text_encoder import CustomCLIP, TextEncoder, load_clip_to_cpu, get_ImageNet_ClassNames

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train CLIP model')
parser.add_argument('--model', type=str, default='ViT-B/32', help='Pre-trained CLIP model')
parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
parser.add_argument('--image_dim', type=int, default=512, help='Dimension of the image encoder')
parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the classification head')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--save_path', type=str, default='fine_tuned_clip.pt', help='Path to save the fine-tuned model')
args = parser.parse_args()


# Classnames
classnames = get_ImageNet_ClassNames()

# Load the pre-trained CLIP model
clip_model = load_clip_to_cpu()
clip_model = CustomCLIP(args, classnames, clip_model)

# Freeze the text encoder
for param in clip_model.text_encoder.parameters():
    param.requires_grad = False

# Move the model to the specified device
clip_model = clip_model.to(args.device)

# Define your loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clip_model.parameters(), lr=args.lr)

train_loader, val_loader, test_loader = ImageNetDataLoader(dataset_path='path/to/imagenet').load_data()

# Fine-tuning loop
for epoch in range(args.num_epochs):
    for images, labels in train_loader:
        # Preprocess the images and labels
        images = preprocess(images).to(args.device)
        labels = labels.to(args.device)

        # Forward pass
        features = model.encode_image(images)
        logits = model.image_encoder(features)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss for monitoring
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item()}")

# Save the fine-tuned model
torch.save(model.state_dict(), args.save_path)