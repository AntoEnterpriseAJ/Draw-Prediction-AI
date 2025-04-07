import gradio as gr
import torch
import numpy as np
from PIL import Image
from model.model import Model
import config.config as config

model = Model(num_output=10)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
model.eval()

def predict_from_canvas(drawing):
    if isinstance(drawing, dict):
        composite = drawing.get("composite")
        if composite is None:
            raise ValueError("Composite image not found. Available keys: " + str(drawing.keys()))
    else:
        composite = drawing

    alpha_channel = composite[:, :, 3]

    img = Image.fromarray(alpha_channel, mode='L')
    img = img.resize((28, 28), resample=Image.BILINEAR)

    threshold = 128
    img = img.point(lambda p: 255 if p > threshold else 0)

    img.save("my_drawing_mnist.png")

    img_array = np.array(img).astype('float32') / 255.0
    img_array = (img_array - 0.1307) / 0.3081

    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_digit = output.argmax(dim=1).item()

    return str(predicted_digit)

iface = gr.Interface(
    fn=predict_from_canvas,
    inputs=gr.Sketchpad(canvas_size=(280, 280)),
    outputs=gr.Label(label="Predicted Digit"),
    title="Draw a Digit (0–9)",
)

iface.launch()
