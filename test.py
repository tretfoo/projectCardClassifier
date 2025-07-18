import torch
from sklearn.metrics import confusion_matrix
from dataset import get_card_loaders
from model.model import CardCNN
from utils.test_utils import dataset_classes
from utils.visualization_utils import draw_confusion_matrix

def testing_model():
    """ Загружает обученную модель, вычисляет предсказания на тестовом наборе,
       строит матрицу ошибок и сохраняет её в файл. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_card_loaders(batch_size=64)

    model = CardCNN().to(device)
    model.load_state_dict(torch.load('model/saved_model.pth'))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for img, labels in test_loader:
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Построение матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    draw_confusion_matrix(cm, dataset_classes(), save_path='plots/confusion_matrix.png')


if __name__ == '__main__':
    testing_model()


