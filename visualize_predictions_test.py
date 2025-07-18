import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model.model import CardCNN
from utils.test_utils import dataset_classes
from utils.visualization_utils import visualize_predictions


def preprocess_img(image_path, transform):
    """ Функция предобработки изображения"""
    img = Image.open(image_path).convert("RGB")
    return img, transform(img).unsqueeze(0)


def predict(model, image_tensor, device):
    """ Выполняет предсказание для одного изображения с использованием обученной модели."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Определение устройства, загрузка модели и её весов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardCNN().to(device)
model.load_state_dict(torch.load('model/saved_model.pth'))
model.eval()


image_paths = glob.glob('./data/test/*/*') # Поиск всех изображений в подкаталогах
test_examples = np.random.choice(image_paths, 5) # Выбор 5 случайных изображений из набора

for idx, example in enumerate(test_examples):
    original_image, image_tensor = preprocess_img(example, transform)
    probabilities = predict(model, image_tensor, device)

    class_names = dataset_classes() # Получение списка имён классов
    visualize_predictions(original_image, probabilities, class_names, idx=idx+1) # Визуализация результатов
