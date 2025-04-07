import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import config

class MNISTDataModule:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_data = datasets.MNIST(
            root=config.DATA_DIR,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_data = datasets.MNIST(
            root=config.DATA_DIR,
            train=False,
            download=True,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


class Trainer:
    def __init__(self, model, datamodule, lr=0.001, epochs=5):
        self.model = model
        self.dm = datamodule
        self.epochs = epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(self):
        train_loader = self.dm.train_dataloader()

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                if i % 10 == 0:
                    batch_acc = (predicted == labels).sum().item() / labels.size(0)
                    print(f'[Epoch {epoch+1}, Batch {i+1:3d}] Loss: {running_loss / 10:.3f}, Batch Acc: {batch_acc*100:.2f}%')
                    running_loss = 0.0

            epoch_acc = 100.0 * correct / total
            print(f'--- Epoch {epoch + 1} finished --- Accuracy: {epoch_acc:.2f}%\n')

        print("Finished Training")

    def test(self):
        test_loader = self.dm.test_dataloader()
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total:.2f}%')

    def save(self, path="mnist_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')
