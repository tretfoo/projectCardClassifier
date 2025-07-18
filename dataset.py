import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class CardDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = ImageFolder(data, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


def get_card_loaders(batch_size=64):
    # Аугментации и нормализация для обучения
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Только нормализация для валидации и теста
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_folder = './data/train/'
    val_folder = './data/valid/'
    test_folder = './data/test/'

    # Инициализация датасетов
    train_dataset = CardDataset(train_folder, transform=train_transform)
    val_dataset = CardDataset(val_folder, transform=valid_transform)
    test_dataset = CardDataset(test_folder, transform=valid_transform)

    # Загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
