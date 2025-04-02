from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from model import Model
from drawing_dataset import DrawingDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = DrawingDataset(data_dir='./npydataset/')

print(dataset[250])

print(f'Dataset size: {len(dataset)}')
print(f'Dataset classes count: {len(dataset.classes)}')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

print(f'Train dataset size: {len(train_data)}')
print(f'Test dataset size: {len(test_data)}')

train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

image_tensor, label = train_data[800]
image = image_tensor.permute(1, 2, 0).numpy()
plt.imshow(image, cmap='gray')
plt.title(f'{dataset.classes[label]}')
plt.axis('off')
plt.show()

model = Model(num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predictions = torch.max(outputs.data, 1)
        batch_correct = (predictions == labels).sum().item()
        batch_count = labels.size(0)

        if i % 100 == 0:
            batch_accuracy = 100 * batch_correct / batch_count
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}  accuracy: {batch_accuracy:.2f}%")

        total_loss += loss.item()
        total_correct += batch_correct
        total_count += batch_count

    epoch_accuracy = 100 * total_correct / total_count
    print(f"Epoch {epoch + 1} finished. Loss: {total_loss:.3f}, Accuracy: {epoch_accuracy:.2f}%\n")

# Save model
print("Finished training")
torch.save(model.state_dict(), "model.pth")
print("Saved the model")

model.eval()
correct_pred = {classname: 0 for classname in dataset.classes}
total_pred = {classname: 0 for classname in dataset.classes}

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[dataset.classes[label]] += 1
            total_pred[dataset.classes[label]] += 1

total_accuracy = 0
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class {classname:20s}: {accuracy:.1f} %')
    total_accuracy += accuracy

print(f"\nTotal accuracy: {total_accuracy / len(dataset.classes):.2f}%")
