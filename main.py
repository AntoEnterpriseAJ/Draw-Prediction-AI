from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

import matplotlib.pyplot as plt
import numpy as np

import torch.optim as optim

from model import Model

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        fill=255
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


dataset = datasets.ImageFolder(root='./dataset/quick_draw_subset/', transform=transform)
print(f'dataset size: {len(dataset)}')
print(f'dataset classes count: {len(dataset.classes)}')

train_size: int = int(0.8 * len(dataset))
test_size: int = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
print(f'train_dataset size: {len(train_data)}')
print(f'test_dataset size: {len(test_data)}')

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

image_tensor, label = train_data[100]
image = image_tensor.permute(1, 2, 0).numpy()

plt.imshow(image, cmap='gray')
plt.title(f'{dataset.classes[label]}')
plt.show()


model = Model()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20

for epoch in range(epochs):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs.data, dim=1)
        batch_correct = torch.eq(predictions, labels).sum().item()
        batch_count = labels.size(0)

        if i % 100 == 0 :
            batch_accuracy = 100 * batch_correct / batch_count

            print(f"[{epoch + 1}, {i + 1:5d}] loss: {batch_loss.item():.3f}  accuracy: {batch_accuracy:.2f}%")

        total_loss += batch_loss.item()
        total_correct += batch_correct
        total_count += batch_count

    epoch_acc = 100 * total_correct / total_count
    print(f"Epoch {epoch + 1} finished. Loss: {total_loss:.3f}, Accuracy: {epoch_acc:.2f}%\n")


print("finished training")
torch.save(model.state_dict(), "model.pth")
print("saved the model")

model.eval()

correct_pred = {classname: 0 for classname in dataset.classes}
total_pred = {classname: 0 for classname in dataset.classes}

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)

        _, predictions = torch.max(outputs.data, dim=1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[dataset.classes[label]] += 1
            total_pred[dataset.classes[label]] += 1

total_accuracy = 0

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    total_accuracy += accuracy

print(f"Total accuracy is {total_accuracy / len(dataset.classes):.2f}%")
