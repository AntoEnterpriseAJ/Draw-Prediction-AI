import torch
from torchvision import datasets, transforms
from model import Model
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((128)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root='./dataset/quick_draw_subset/', transform=transform)
model = Model()

model.load_state_dict(torch.load('model.pth'))

model.eval()

image = Image.open('ballMaybw.png')

input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probabilities, 10)

    for i in range(10):
        class_index = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        print(f'{dataset.classes[class_index]} with probability {prob:.2%}')

import matplotlib.pyplot as plt
plt.imshow(input_tensor.squeeze().numpy(), cmap='gray')
plt.title("Input to the model")
plt.show()
