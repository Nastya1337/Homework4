# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# import time
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # Установка устройства
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Гиперпараметры
# batch_size = 64
# learning_rate = 0.001
# num_epochs = 10
#
# # Загрузка данных
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#
# # 1. Полносвязная сеть
# class FCNet(nn.Module):
#     def __init__(self):
#         super(FCNet, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # flatten
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x
#
#
# # 2. Простая CNN
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # flatten
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
#
#
# # 3. CNN с Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(residual)
#         out = self.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self):
#         super(ResNet, self).__init__()
#         self.in_channels = 32
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(32, 2, stride=1)
#         self.layer2 = self.make_layer(64, 2, stride=2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64, 10)
#
#     def make_layer(self, out_channels, blocks, stride=1):
#         layers = []
#         layers.append(ResidualBlock(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         for _ in range(1, blocks):
#             layers.append(ResidualBlock(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
#
#
# # Функция для обучения
# def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_name):
#     train_losses = []
#     train_accs = []
#     test_accs = []
#     times = []
#
#     model.to(device)
#
#     start_time = time.time()
#
#     for epoch in range(num_epochs):
#         epoch_start = time.time()
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}"):
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         epoch_loss = running_loss / len(train_loader)
#         epoch_acc = 100 * correct / total
#         train_losses.append(epoch_loss)
#         train_accs.append(epoch_acc)
#
#         # Тестирование
#         test_acc = evaluate(model, test_loader)
#         test_accs.append(test_acc)
#
#         epoch_time = time.time() - epoch_start
#         times.append(epoch_time)
#
#         print(
#             f"{model_name} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")
#
#     total_time = time.time() - start_time
#     print(f"{model_name} Training complete in {total_time:.2f}s")
#
#     return train_losses, train_accs, test_accs, times
#
#
# # Функция для оценки
# def evaluate(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return 100 * correct / total
#
#
# # Функция для измерения времени инференса
# def measure_inference_time(model, test_loader, num_runs=100):
#     model.eval()
#     model.to(device)
#
#     # Прогрев
#     with torch.no_grad():
#         for images, _ in test_loader:
#             images = images.to(device)
#             _ = model(images)
#             break
#
#     # Измерение времени
#     start_time = time.time()
#     with torch.no_grad():
#         for i, (images, _) in enumerate(test_loader):
#             if i >= num_runs:
#                 break
#             images = images.to(device)
#             _ = model(images)
#
#     total_time = time.time() - start_time
#     avg_time = total_time / num_runs
#     return avg_time
#
#
# # Функция для визуализации
# def plot_results(model_name, train_losses, train_accs, test_accs, times):
#     plt.figure(figsize=(15, 5))
#
#     plt.subplot(1, 3, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.title(f'{model_name} Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#
#     plt.subplot(1, 3, 2)
#     plt.plot(train_accs, label='Train Accuracy')
#     plt.plot(test_accs, label='Test Accuracy')
#     plt.title(f'{model_name} Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#
#     plt.subplot(1, 3, 3)
#     plt.plot(times, label='Time per epoch')
#     plt.title(f'{model_name} Training Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Time (s)')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Инициализация моделей
# fc_model = FCNet()
# cnn_model = SimpleCNN()
# resnet_model = ResNet()
#
# # Критерий и оптимизатор
# criterion = nn.CrossEntropyLoss()
#
# # Словарь для хранения результатов
# results = {}
#
# # Обучение и оценка FCNet
# print("\nTraining FCNet...")
# optimizer = optim.Adam(fc_model.parameters(), lr=learning_rate)
# fc_train_losses, fc_train_accs, fc_test_accs, fc_times = train_model(
#     fc_model, train_loader, test_loader, criterion, optimizer, num_epochs, "FCNet"
# )
# fc_inference_time = measure_inference_time(fc_model, test_loader)
# fc_params = sum(p.numel() for p in fc_model.parameters())
#
# results['FCNet'] = {
#     'train_acc': fc_train_accs[-1],
#     'test_acc': fc_test_accs[-1],
#     'train_time': sum(fc_times),
#     'inference_time': fc_inference_time,
#     'params': fc_params
# }
#
# plot_results("FCNet", fc_train_losses, fc_train_accs, fc_test_accs, fc_times)
#
# # Обучение и оценка SimpleCNN
# print("\nTraining SimpleCNN...")
# optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
# cnn_train_losses, cnn_train_accs, cnn_test_accs, cnn_times = train_model(
#     cnn_model, train_loader, test_loader, criterion, optimizer, num_epochs, "SimpleCNN"
# )
# cnn_inference_time = measure_inference_time(cnn_model, test_loader)
# cnn_params = sum(p.numel() for p in cnn_model.parameters())
#
# results['SimpleCNN'] = {
#     'train_acc': cnn_train_accs[-1],
#     'test_acc': cnn_test_accs[-1],
#     'train_time': sum(cnn_times),
#     'inference_time': cnn_inference_time,
#     'params': cnn_params
# }
#
# plot_results("SimpleCNN", cnn_train_losses, cnn_train_accs, cnn_test_accs, cnn_times)
#
# # Обучение и оценка ResNet
# print("\nTraining ResNet...")
# optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate)
# resnet_train_losses, resnet_train_accs, resnet_test_accs, resnet_times = train_model(
#     resnet_model, train_loader, test_loader, criterion, optimizer, num_epochs, "ResNet"
# )
# resnet_inference_time = measure_inference_time(resnet_model, test_loader)
# resnet_params = sum(p.numel() for p in resnet_model.parameters())
#
# results['ResNet'] = {
#     'train_acc': resnet_train_accs[-1],
#     'test_acc': resnet_test_accs[-1],
#     'train_time': sum(resnet_times),
#     'inference_time': resnet_inference_time,
#     'params': resnet_params
# }
#
# plot_results("ResNet", resnet_train_losses, resnet_train_accs, resnet_test_accs, resnet_times)
#
# # Вывод результатов
# print("\n=== Final Results ===")
# for model_name, metrics in results.items():
#     print(f"\n{model_name}:")
#     print(f"Train Accuracy: {metrics['train_acc']:.2f}%")
#     print(f"Test Accuracy: {metrics['test_acc']:.2f}%")
#     print(f"Total Training Time: {metrics['train_time']:.2f}s")
#     print(f"Inference Time per batch: {metrics['inference_time']:.4f}s")
#     print(f"Number of Parameters: {metrics['params']}")
#
#
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from multiprocessing import freeze_support


def main():
    # Установка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Гиперпараметры
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 50
    weight_decay = 1e-4  # для L2 регуляризации

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0)  # Изменил num_workers на 0 для Windows
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0)  # Изменил num_workers на 0 для Windows

    # 1. Глубокая полносвязная сеть
    class DeepFCN(nn.Module):
        def __init__(self):
            super(DeepFCN, self).__init__()
            self.fc1 = nn.Linear(32 * 32 * 3, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # flatten
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.dropout(self.relu(self.fc3(x)))
            x = self.dropout(self.relu(self.fc4(x)))
            x = self.fc5(x)
            return x

    # 2. CNN с Residual блоками
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += self.shortcut(residual)
            out = self.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.in_channels = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self.make_layer(16, 2, stride=1)
            self.layer2 = self.make_layer(32, 2, stride=2)
            self.layer3 = self.make_layer(64, 2, stride=2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)

        def make_layer(self, out_channels, blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    # 3. CNN с Residual блоками и регуляризацией
    class RegularizedResNet(nn.Module):
        def __init__(self):
            super(RegularizedResNet, self).__init__()
            self.in_channels = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self.make_layer(16, 2, stride=1)
            self.layer2 = self.make_layer(32, 2, stride=2)
            self.layer3 = self.make_layer(64, 2, stride=2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.2)

        def make_layer(self, out_channels, blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.fc(out)
            return out

    # Функция для обучения
    def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_name):
        train_losses = []
        train_accs = []
        test_accs = []
        times = []

        model.to(device)

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)

            # Тестирование
            test_acc = evaluate(model, test_loader)
            test_accs.append(test_acc)

            epoch_time = time.time() - epoch_start
            times.append(epoch_time)

            print(
                f"{model_name} Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print(f"{model_name} Training complete in {total_time:.2f}s")

        return train_losses, train_accs, test_accs, times

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

    # Функция для построения confusion matrix
    def plot_confusion_matrix(model, test_loader, class_names):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    # Функция для исследования градиентов
    def plot_gradient_flow(model):
        gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None and "weight" in name:
                gradients.append(param.grad.abs().mean().item())

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(gradients)), gradients)
        plt.xlabel("Layer")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.yscale("log")
        plt.show()

    # Функция для визуализации результатов
    def plot_results(model_name, train_losses, train_accs, test_accs):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title(f'{model_name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(test_accs, label='Test Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Инициализация моделей
    fcn_model = DeepFCN()
    resnet_model = ResNet()
    reg_resnet_model = RegularizedResNet()

    # Критерий
    criterion = nn.CrossEntropyLoss()

    # Словарь для хранения результатов
    results = {}

    # Классы CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Обучение и оценка DeepFCN
    print("\nTraining DeepFCN...")
    optimizer = optim.Adam(fcn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fcn_train_losses, fcn_train_accs, fcn_test_accs, fcn_times = train_model(
        fcn_model, train_loader, test_loader, criterion, optimizer, num_epochs, "DeepFCN"
    )

    results['DeepFCN'] = {
        'train_acc': fcn_train_accs[-1],
        'test_acc': fcn_test_accs[-1],
        'train_time': sum(fcn_times),
        'params': sum(p.numel() for p in fcn_model.parameters()),
        'gradient_flow': None
    }

    plot_results("DeepFCN", fcn_train_losses, fcn_train_accs, fcn_test_accs)
    plot_confusion_matrix(fcn_model, test_loader, class_names)

    # Исследование градиентов для DeepFCN
    plot_gradient_flow(fcn_model)

    # Обучение и оценка ResNet
    print("\nTraining ResNet...")
    optimizer = optim.Adam(resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    resnet_train_losses, resnet_train_accs, resnet_test_accs, resnet_times = train_model(
        resnet_model, train_loader, test_loader, criterion, optimizer, num_epochs, "ResNet"
    )

    results['ResNet'] = {
        'train_acc': resnet_train_accs[-1],
        'test_acc': resnet_test_accs[-1],
        'train_time': sum(resnet_times),
        'params': sum(p.numel() for p in resnet_model.parameters()),
        'gradient_flow': None
    }

    plot_results("ResNet", resnet_train_losses, resnet_train_accs, resnet_test_accs)
    plot_confusion_matrix(resnet_model, test_loader, class_names)

    # Исследование градиентов для ResNet
    plot_gradient_flow(resnet_model)

    # Обучение и оценка RegularizedResNet
    print("\nTraining RegularizedResNet...")
    optimizer = optim.Adam(reg_resnet_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    reg_resnet_train_losses, reg_resnet_train_accs, reg_resnet_test_accs, reg_resnet_times = train_model(
        reg_resnet_model, train_loader, test_loader, criterion, optimizer, num_epochs, "RegularizedResNet"
    )

    results['RegularizedResNet'] = {
        'train_acc': reg_resnet_train_accs[-1],
        'test_acc': reg_resnet_test_accs[-1],
        'train_time': sum(reg_resnet_times),
        'params': sum(p.numel() for p in reg_resnet_model.parameters()),
        'gradient_flow': None
    }

    plot_results("RegularizedResNet", reg_resnet_train_losses, reg_resnet_train_accs, reg_resnet_test_accs)
    plot_confusion_matrix(reg_resnet_model, test_loader, class_names)

    # Исследование градиентов для RegularizedResNet
    plot_gradient_flow(reg_resnet_model)

    # Вывод результатов
    print("\n=== Final Results ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Train Accuracy: {metrics['train_acc']:.2f}%")
        print(f"Test Accuracy: {metrics['test_acc']:.2f}%")
        print(f"Total Training Time: {metrics['train_time']:.2f}s")
        print(f"Number of Parameters: {metrics['params']}")


if __name__ == '__main__':
    freeze_support()
    main()