import torchvision.transforms as transforms
from dataset import CardDataset


def dataset_classes():
    """ Получает список классов из тестового датасета. """
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    test_folder = './data/test/'
    test_dataset = CardDataset(test_folder, transform=transform)
    return test_dataset.classes