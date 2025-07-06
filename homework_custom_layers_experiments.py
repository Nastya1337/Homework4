# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.autograd import Function
#
#
# # 1. Кастомный сверточный слой с дополнительной логикой (добавляет learnable шум)
# class NoisyConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, noise_scale=0.1):
#         super(NoisyConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         self.noise_scale = nn.Parameter(torch.tensor(noise_scale))
#         self.noise = None
#
#     def forward(self, x):
#         if self.training:
#             # Генерируем шум только во время обучения
#             noise = torch.randn_like(x) * self.noise_scale
#             self.noise = noise.detach()  # сохраняем для визуализации
#             x = x + noise
#         return self.conv(x)
#
#
# # 2. Attention механизм для CNN
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=8):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(in_channels // reduction_ratio, in_channels)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#
#         # Avg и Max Pooling
#         avg_out = self.fc(self.avg_pool(x).view(b, c))
#         max_out = self.fc(self.max_pool(x).view(b, c))
#
#         # Складываем и применяем sigmoid
#         out = avg_out + max_out
#         out = self.sigmoid(out).view(b, c, 1, 1)
#
#         return x * out
#
#
# # 3. Кастомная функция активации (Swish + learnable параметр)
# class SwishFunction(Function):
#     @staticmethod
#     def forward(ctx, x, beta):
#         ctx.save_for_backward(x, beta)
#         return x * torch.sigmoid(beta * x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, beta = ctx.saved_tensors
#         sigmoid = torch.sigmoid(beta * x)
#         return grad_output * (sigmoid + beta * x * sigmoid * (1 - sigmoid)), None
#
#
# class Swish(nn.Module):
#     def __init__(self):
#         super(Swish, self).__init__()
#         self.beta = nn.Parameter(torch.tensor(1.0))
#
#     def forward(self, x):
#         return SwishFunction.apply(x, self.beta)
#
#
# # 4. Кастомный pooling слой (средний + максимальный)
# class HybridPool2d(nn.Module):
#     def __init__(self, kernel_size, stride=None, padding=0):
#         super(HybridPool2d, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride or kernel_size
#         self.padding = padding
#         self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable весовой коэффициент
#
#     def forward(self, x):
#         avg_pool = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
#         max_pool = F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
#         return self.alpha * avg_pool + (1 - self.alpha) * max_pool
#
#
# # Тестирование кастомных слоев
# def test_custom_layers():
#     # Тестовые данные
#     x = torch.randn(1, 3, 32, 32)
#
#     print("\n=== Тестирование кастомных слоев ===")
#
#     # 1. Тестирование NoisyConv2d
#     print("\n1. NoisyConv2d:")
#     noisy_conv = NoisyConv2d(3, 16, 3, padding=1)
#     noisy_conv.train()
#     out = noisy_conv(x)
#     print("Входной размер:", x.shape)
#     print("Выходной размер:", out.shape)
#     print("Масштаб шума:", noisy_conv.noise_scale.item())
#
#     # Визуализация шума
#     if noisy_conv.noise is not None:
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.imshow(x[0, 0].detach().numpy(), cmap='gray')
#         plt.title("Оригинальное изображение")
#         plt.subplot(1, 2, 2)
#         plt.imshow((x + noisy_conv.noise)[0, 0].detach().numpy(), cmap='gray')
#         plt.title("С добавленным шумом")
#         plt.show()
#
#     # 2. Тестирование ChannelAttention
#     print("\n2. ChannelAttention:")
#     attn = ChannelAttention(3)
#     out = attn(x)
#     print("Входной размер:", x.shape)
#     print("Выходной размер:", out.shape)
#
#     # 3. Тестирование Swish
#     print("\n3. Swish активация:")
#     swish = Swish()
#     out = swish(x)
#     print("Входной размер:", x.shape)
#     print("Выходной размер:", out.shape)
#     print("Параметр beta:", swish.beta.item())
#
#     # Сравнение с обычными функциями активации
#     plt.figure(figsize=(10, 5))
#     x_vals = torch.linspace(-5, 5, 100)
#     plt.plot(x_vals.numpy(), swish(x_vals).detach().numpy(), label='Swish')
#     plt.plot(x_vals.numpy(), F.relu(x_vals).numpy(), label='ReLU')
#     plt.plot(x_vals.numpy(), torch.sigmoid(x_vals).numpy(), label='Sigmoid')
#     plt.legend()
#     plt.title("Сравнение функций активации")
#     plt.show()
#
#     # 4. Тестирование HybridPool2d
#     print("\n4. HybridPool2d:")
#     hybrid_pool = HybridPool2d(2)
#     out = hybrid_pool(x)
#     print("Входной размер:", x.shape)
#     print("Выходной размер:", out.shape)
#     print("Параметр alpha:", hybrid_pool.alpha.item())
#
#     # Сравнение с обычными пулингами
#     avg_out = F.avg_pool2d(x, 2)
#     max_out = F.max_pool2d(x, 2)
#     hybrid_out = hybrid_pool(x)
#
#     print("\nСравнение с обычными пулингами:")
#     print("Средний пулинг (первый элемент):", avg_out[0, 0, 0, 0].item())
#     print("Максимальный пулинг (первый элемент):", max_out[0, 0, 0, 0].item())
#     print("Гибридный пулинг (первый элемент):", hybrid_out[0, 0, 0, 0].item())
#
#
# # CNN с кастомными слоями для тестирования
# class CustomCNN(nn.Module):
#     def __init__(self):
#         super(CustomCNN, self).__init__()
#
#         self.conv1 = NoisyConv2d(3, 32, 3, padding=1)
#         self.attn1 = ChannelAttention(32)
#         self.swish1 = Swish()
#         self.pool1 = HybridPool2d(2)
#
#         self.conv2 = NoisyConv2d(32, 64, 3, padding=1)
#         self.attn2 = ChannelAttention(64)
#         self.swish2 = Swish()
#         self.pool2 = HybridPool2d(2)
#
#         self.fc = nn.Linear(64 * 8 * 8, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.attn1(x)
#         x = self.swish1(x)
#         x = self.pool1(x)
#
#         x = self.conv2(x)
#         x = self.attn2(x)
#         x = self.swish2(x)
#         x = self.pool2(x)
#
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#
# # Тестирование полной модели
# def test_full_model():
#     print("\n=== Тестирование полной модели с кастомными слоями ===")
#     model = CustomCNN()
#     x = torch.randn(2, 3, 32, 32)
#     out = model(x)
#     print("Входной размер:", x.shape)
#     print("Выходной размер:", out.shape)
#
#     # Проверка обратного прохода
#     criterion = nn.MSELoss()
#     target = torch.randn_like(out)
#     loss = criterion(out, target)
#     loss.backward()
#     print("Обратный проход выполнен успешно!")
#
#     # Вывод параметров
#     print("\nПараметры модели:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.shape}")
#
#
# if __name__ == '__main__':
#     test_custom_layers()
#     test_full_model()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# Установка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры
batch_size = 128
learning_rate = 0.1
num_epochs = 50
weight_decay = 5e-4
momentum = 0.9

# Загрузка данных CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# 1. Базовый Residual блок
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 2. Bottleneck Residual блок
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 3. Wide Residual блок
class WideBlock(nn.Module):
    expansion = 1  # Добавляем атрибут expansion

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(WideBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Общая архитектура ResNet
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, widen_factor=1):
        super(ResNet, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16 * widen_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * widen_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * widen_factor, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * widen_factor * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# Функция для обучения
def train_model(model, train_loader, test_loader, optimizer, scheduler, num_epochs, model_name):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accs = []
    test_accs = []
    grad_norms = []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Сбор градиентов для анализа
        grad_norm = []

        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Сбор норм градиентов
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm.append(total_norm ** 0.5)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        grad_norms.append(np.mean(grad_norm))

        # Тестирование
        test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)

        print(
            f"{model_name} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    return train_losses, train_accs, test_accs, grad_norms


# Функция для оценки
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Функция для построения графиков
def plot_results(model_name, train_losses, train_accs, test_accs, grad_norms):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(grad_norms)
    plt.title(f'{model_name} Gradient Norms')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')

    plt.tight_layout()
    plt.show()


# Создание и обучение моделей
def experiment():
    models = {
        'BasicResNet': ResNet(BasicBlock, [3, 3, 3]),
        'BottleneckResNet': ResNet(BottleneckBlock, [3, 4, 6]),
        'WideResNet': ResNet(WideBlock, [3, 3, 3], widen_factor=4)
    }

    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")

        # Оптимизатор и планировщик
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

        # Обучение
        train_losses, train_accs, test_accs, grad_norms = train_model(
            model, train_loader, test_loader, optimizer, scheduler, num_epochs, name
        )

        # Сохранение результатов
        results[name] = {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'test_acc': test_accs,
            'grad_norms': grad_norms,
            'params': sum(p.numel() for p in model.parameters()),
            'final_test_acc': test_accs[-1]
        }

        # Визуализация
        plot_results(name, train_losses, train_accs, test_accs, grad_norms)

    # Сравнение моделей
    print("\n=== Сравнение моделей ===")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"Количество параметров: {res['params']:,}")
        print(f"Финальная точность на тесте: {res['final_test_acc']:.2f}%")

    # Визуализация сравнения
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['test_acc'], label=name)
    plt.title('Сравнение Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(1, 2, 2)
    params = [res['params'] for res in results.values()]
    accs = [res['final_test_acc'] for res in results.values()]
    plt.bar(results.keys(), params)
    plt.title('Количество параметров')
    plt.ylabel('Параметры')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    experiment()