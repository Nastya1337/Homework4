# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score
#
#
# def main():
#     # Проверка доступности GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Загрузка и подготовка данных CIFAR-10
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     # Убираем многопроцессорность для Windows
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
#
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
#
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#     # Функция для вычисления количества параметров
#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#     # Базовый класс модели
#     class ConvNet(nn.Module):
#         def __init__(self, kernel_sizes):
#             super(ConvNet, self).__init__()
#             self.kernel_sizes = kernel_sizes
#
#             # Первый сверточный слой
#             self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
#             self.bn1 = nn.BatchNorm2d(32)
#             self.relu = nn.ReLU()
#             self.pool = nn.MaxPool2d(2, 2)
#
#             # Второй сверточный слой
#             self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
#             self.bn2 = nn.BatchNorm2d(64)
#
#             # Полносвязные слои
#             self.fc1 = nn.Linear(64 * 8 * 8, 512)
#             self.fc2 = nn.Linear(512, 10)
#
#         def forward(self, x):
#             x = self.pool(self.relu(self.bn1(self.conv1(x))))
#             x = self.pool(self.relu(self.bn2(self.conv2(x))))
#             x = x.view(-1, 64 * 8 * 8)
#             x = self.relu(self.fc1(x))
#             x = self.fc2(x)
#             return x
#
#     # Модель с комбинацией 1x1 и 3x3 ядер
#     class MixedConvNet(nn.Module):
#         def __init__(self):
#             super(MixedConvNet, self).__init__()
#
#             # Первый слой: комбинация 1x1 и 3x3 сверток
#             self.conv1_1x1 = nn.Conv2d(3, 16, kernel_size=1)
#             self.conv1_3x3 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#             self.bn1 = nn.BatchNorm2d(32)
#             self.relu = nn.ReLU()
#             self.pool = nn.MaxPool2d(2, 2)
#
#             # Второй слой
#             self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#             self.bn2 = nn.BatchNorm2d(64)
#
#             # Полносвязные слои
#             self.fc1 = nn.Linear(64 * 8 * 8, 512)
#             self.fc2 = nn.Linear(512, 10)
#
#         def forward(self, x):
#             # Комбинируем выходы 1x1 и 3x3 сверток
#             x1 = self.conv1_1x1(x)
#             x2 = self.conv1_3x3(x)
#             x = torch.cat((x1, x2), dim=1)
#             x = self.pool(self.relu(self.bn1(x)))
#
#             x = self.pool(self.relu(self.bn2(self.conv2(x))))
#             x = x.view(-1, 64 * 8 * 8)
#             x = self.relu(self.fc1(x))
#             x = self.fc2(x)
#             return x
#
#     # Функция для обучения модели
#     def train_model(model, trainloader, testloader, epochs=10):
#         model.to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#         train_accs, test_accs = [], []
#         start_time = time.time()
#
#         for epoch in range(epochs):
#             model.train()
#             running_loss = 0.0
#             all_preds, all_labels = [], []
#
#             for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
#                 inputs, labels = inputs.to(device), labels.to(device)
#
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 running_loss += loss.item()
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#
#             train_acc = accuracy_score(all_labels, all_preds)
#             train_accs.append(train_acc)
#
#             # Оценка на тестовом наборе
#             test_acc = evaluate_model(model, testloader)
#             test_accs.append(test_acc)
#
#             print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
#
#         training_time = time.time() - start_time
#         return train_accs, test_accs, training_time
#
#     # Функция для оценки модели
#     def evaluate_model(model, testloader):
#         model.eval()
#         all_preds, all_labels = [], []
#
#         with torch.no_grad():
#             for inputs, labels in testloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 all_preds.extend(preds.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#
#         return accuracy_score(all_labels, all_preds)
#
#     # Функция для визуализации активаций
#     def visualize_activations(model, testloader):
#         model.eval()
#         with torch.no_grad():
#             # Получаем один батч данных
#             dataiter = iter(testloader)
#             images, _ = next(dataiter)
#             images = images.to(device)
#
#             # Получаем активации первого слоя
#             first_conv = next(model.children())
#             if isinstance(first_conv, nn.Sequential):
#                 first_conv = first_conv[0]
#
#             activations = first_conv(images)
#
#             # Визуализируем первые 16 фильтров
#             plt.figure(figsize=(12, 8))
#             for i in range(16):
#                 plt.subplot(4, 4, i + 1)
#                 plt.imshow(activations[0, i].cpu().numpy(), cmap='viridis')
#                 plt.axis('off')
#                 plt.title(f'Filter {i + 1}')
#             plt.suptitle('Активации первого сверточного слоя', fontsize=16)
#             plt.show()
#
#     # Функция для анализа рецептивных полей
#     def analyze_receptive_fields(kernel_sizes):
#         rf = 1
#         stride_product = 1
#         for ks in kernel_sizes:
#             rf += (ks - 1) * stride_product
#             stride_product *= 2  # учитываем пулинг с stride=2
#         return rf
#
#     # Создаем и обучаем модели с разными ядрами
#     kernel_configs = [
#         {'name': '3x3 kernels', 'kernels': [3, 3]},
#         {'name': '5x5 kernels', 'kernels': [5, 5]},
#         {'name': '7x7 kernels', 'kernels': [7, 7]},
#         {'name': 'Mixed 1x1+3x3', 'model_class': MixedConvNet}
#     ]
#
#     results = []
#
#     for config in kernel_configs:
#         print(f"\nTraining model with {config['name']}")
#
#         if 'model_class' in config:
#             model = config['model_class']()
#         else:
#             model = ConvNet(config['kernels'])
#
#         # Подгоняем количество параметров (примерно)
#         print(f"Number of parameters: {count_parameters(model):,}")
#
#         # Обучаем модель
#         train_acc, test_acc, time_taken = train_model(model, trainloader, testloader, epochs=10)
#
#         # Анализ рецептивных полей
#         if 'kernels' in config:
#             rf = analyze_receptive_fields(config['kernels'])
#         else:
#             rf = analyze_receptive_fields([1, 3])  # для mixed модели
#
#         results.append({
#             'name': config['name'],
#             'train_acc': max(train_acc),
#             'test_acc': max(test_acc),
#             'time': time_taken,
#             'receptive_field': rf
#         })
#
#         # Визуализация активаций
#         visualize_activations(model, testloader)
#
#     # Выводим результаты сравнения
#     print("\nComparison Results:")
#     print("{:<15} {:<10} {:<10} {:<10} {:<15}".format(
#         'Model', 'Train Acc', 'Test Acc', 'Time (s)', 'Receptive Field'))
#     for res in results:
#         print("{:<15} {:<10.4f} {:<10.4f} {:<10.2f} {:<15}".format(
#             res['name'], res['train_acc'], res['test_acc'], res['time'], res['receptive_field']))
#
#     # Визуализация результатов
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.bar([res['name'] for res in results], [res['test_acc'] for res in results])
#     plt.title('Test Accuracy')
#     plt.xticks(rotation=45)
#     plt.ylim(0.7, 0.9)
#
#     plt.subplot(1, 2, 2)
#     plt.bar([res['name'] for res in results], [res['time'] for res in results])
#     plt.title('Training Time (s)')
#     plt.xticks(rotation=45)
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def main():
    # Проверка доступности GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Загрузка и подготовка данных CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Функция для вычисления количества параметров
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. Неглубокая CNN (2 сверточных слоя)
    class ShallowCNN(nn.Module):
        def __init__(self):
            super(ShallowCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)  # Исправлено: автоматический расчет размера
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 2. Средняя CNN (4 сверточных слоя)
    class MediumCNN(nn.Module):
        def __init__(self):
            super(MediumCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool = nn.MaxPool2d(2, 2)
            # После 4 пулингов размер будет 32/(2^4) = 2, но CIFAR10 имеет размер 32x32
            # Поэтому правильнее считать, что после 2 пулингов размер будет 8x8
            self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Исправленный размер входа
            self.fc2 = nn.Linear(512, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 3. Глубокая CNN (6 сверточных слоев)
    class DeepCNN(nn.Module):
        def __init__(self):
            super(DeepCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2, 2)
            # После 2 пулингов размер 8x8
            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(self.relu(self.bn4(self.conv4(x))))
            x = self.relu(self.bn5(self.conv5(x)))
            x = self.relu(self.bn6(self.conv6(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 4. CNN с Residual связями
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self):
            super(ResNet, self).__init__()
            self.relu = nn.ReLU()
            self.in_channels = 32

            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)

            self.layer1 = self.make_layer(32, 2, stride=1)
            self.layer2 = self.make_layer(64, 2, stride=2)
            self.layer3 = self.make_layer(128, 2, stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # Добавлено для стабилизации размера
            self.fc = nn.Linear(128 * 4 * 4, 10)

        def make_layer(self, out_channels, blocks, stride):
            strides = [stride] + [1] * (blocks - 1)
            layers = []
            for stride in strides:
                layers.append(ResidualBlock(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)  # Исправлено: автоматический расчет размера
            out = self.fc(out)
            return out

    # Функция для обучения модели с анализом градиентов
    def train_model(model, trainloader, testloader, epochs=10):
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_accs, test_accs = [], []
        grad_norms = []  # Для отслеживания норм градиентов
        start_time = time.time()

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            all_preds, all_labels = [], []

            for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # Проверка размерностей
                if outputs.shape[0] != labels.shape[0]:
                    print(f"Ошибка размерностей: outputs {outputs.shape}, labels {labels.shape}")
                    continue

                loss = criterion(outputs, labels)
                loss.backward()

                # Анализ градиентов
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                grad_norms.append(total_norm)

                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            train_acc = accuracy_score(all_labels, all_preds)
            train_accs.append(train_acc)

            test_acc = evaluate_model(model, testloader)
            test_accs.append(test_acc)

            print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        training_time = time.time() - start_time
        return train_accs, test_accs, training_time, grad_norms

    # Функция для оценки модели
    def evaluate_model(model, testloader):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return accuracy_score(all_labels, all_preds)

    # Функция для визуализации feature maps
    def visualize_feature_maps(model, testloader, layer_num=1):
        model.eval()
        with torch.no_grad():
            # Получаем один батч данных
            dataiter = iter(testloader)
            images, _ = next(dataiter)
            images = images.to(device)

            # Регистрируем хук для получения активаций
            activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach()

                return hook

            # Получаем указатель на нужный слой
            conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
            if layer_num <= len(conv_layers):
                conv_layers[layer_num - 1].register_forward_hook(get_activation(f'conv{layer_num}'))

                # Пропускаем данные через модель
                model(images)

                # Визуализируем feature maps
                if f'conv{layer_num}' in activations:
                    fmaps = activations[f'conv{layer_num}']
                    plt.figure(figsize=(12, 8))
                    for i in range(min(16, fmaps.shape[1])):  # Покажем первые 16 карт признаков
                        plt.subplot(4, 4, i + 1)
                        plt.imshow(fmaps[0, i].cpu().numpy(), cmap='viridis')
                        plt.axis('off')
                        plt.title(f'Map {i + 1}')
                    plt.suptitle(f'Feature maps слоя {layer_num}', fontsize=16)
                    plt.show()
                else:
                    print(f"Не удалось получить feature maps для слоя {layer_num}")
            else:
                print(f"Модель имеет только {len(conv_layers)} сверточных слоев")

    # Создаем и обучаем модели разной глубины
    models = [
        {'name': 'Shallow CNN (2 слоя)', 'model': ShallowCNN()},
        {'name': 'Medium CNN (4 слоя)', 'model': MediumCNN()},
        {'name': 'Deep CNN (6 слоев)', 'model': DeepCNN()},
        {'name': 'ResNet (с Residual)', 'model': ResNet()}
    ]

    results = []

    for m in models:
        print(f"\nTraining {m['name']}")
        print(f"Number of parameters: {count_parameters(m['model']):,}")

        train_acc, test_acc, time_taken, grad_norms = train_model(m['model'], trainloader, testloader, epochs=10)

        # Анализ vanishing/exploding gradients
        avg_grad_norm = np.mean(grad_norms)
        min_grad_norm = np.min(grad_norms)
        max_grad_norm = np.max(grad_norms)

        results.append({
            'name': m['name'],
            'train_acc': max(train_acc),
            'test_acc': max(test_acc),
            'time': time_taken,
            'avg_grad_norm': avg_grad_norm,
            'min_grad_norm': min_grad_norm,
            'max_grad_norm': max_grad_norm
        })

        # Визуализация feature maps
        visualize_feature_maps(m['model'], testloader, layer_num=1)
        if 'ResNet' in m['name']:
            visualize_feature_maps(m['model'], testloader, layer_num=3)

    # Выводим результаты сравнения
    print("\nComparison Results:")
    print("{:<20} {:<10} {:<10} {:<10} {:<15} {:<15} {:<15}".format(
        'Model', 'Train Acc', 'Test Acc', 'Time (s)', 'Avg Grad Norm', 'Min Grad Norm', 'Max Grad Norm'))
    for res in results:
        print("{:<20} {:<10.4f} {:<10.4f} {:<10.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            res['name'], res['train_acc'], res['test_acc'], res['time'],
            res['avg_grad_norm'], res['min_grad_norm'], res['max_grad_norm']))

    # Визуализация результатов
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.bar([res['name'] for res in results], [res['test_acc'] for res in results])
    plt.title('Test Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 0.9)

    plt.subplot(1, 3, 2)
    plt.bar([res['name'] for res in results], [res['time'] for res in results])
    plt.title('Training Time (s)')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    for i, res in enumerate(results):
        plt.plot([res['avg_grad_norm']] * 10, label=models[i]['name'])
    plt.title('Average Gradient Norms')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()