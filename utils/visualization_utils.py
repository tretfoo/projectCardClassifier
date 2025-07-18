import os
import seaborn as sns
import torch
import matplotlib.pyplot as plt

def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def training_history(train_losses, valid_losses, train_accs, valid_accs):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(valid_losses, label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(valid_accs, label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.show()


def visualize_predictions(original_image, probabilities, class_names, idx):
    """ Визуализирует предсказания модели для одного изображения. """
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)


    if idx is not None:
        plot_path = os.path.join('./plots', f'visualized_prediction{idx}.png')
    else:
        plot_path = os.path.join('./plots', 'visualized_prediction.png')

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()


def draw_confusion_matrix(cm, class_names, save_path=None):
    """ Рисует и сохраняет матрицу ошибок. """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

