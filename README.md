---

# MNIST Handwritten Digit Classifier

A PyTorch-based deep learning project for classifying handwritten digits from the **MNIST dataset**. The project includes a **Gradio interface** that allows users to draw digits and get **real-time predictions**.

---

## Features

- Convolutional Neural Network (CNN) using PyTorch  
- Training and evaluation on the MNIST dataset  
- Model saving and loading  
- Gradio-powered drawing interface for digit prediction  

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the Gradio Interface

Before using the digit drawing interface, you **must run the main script**:

```bash
python src/main.py
```

This starts the Gradio app locally and opens a web UI:

```text
* Running on local URL: http://127.0.0.1:7860
```

---

### 2. Train the Model

To train the model on the MNIST dataset:

```bash
python src/model/train.py
```

- The model is trained over several epochs
- Final weights are saved to `src/model/mnist_model.pth`

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.

---
