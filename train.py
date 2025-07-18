import torch
from dataset import get_card_loaders
from model.model import CardCNN
from utils.training_utils import train_model
from utils.visualization_utils import save_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_card_loaders(batch_size=64)
    model = CardCNN(num_classes=53)
    model.to(device)

    train_model(model, train_loader, val_loader, 30, device=str(device))
    save_model(model, './model/best_saved_model.pth')