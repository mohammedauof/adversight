import gradio as gr
from PIL import Image
import torch
from torchvision import models, transforms

# Load model
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load classes
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

def predict(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    _, predicted = torch.max(output, 1)
    return classes[predicted.item()]

# UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AdverSight Demo"
)

interface.launch()
