from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import base64
import io
import json
import os

app = Flask(__name__)

# Load model once at startup
device = torch.device("cpu")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Inverse normalize for converting tensor back to image
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


def tensor_to_base64(tensor):
    """Convert a normalized image tensor to base64 PNG string."""
    img_tensor = inv_normalize(tensor.squeeze(0)).clamp(0, 1)
    img_np = (img_tensor.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_top5(logits):
    """Return top 5 predictions as list of {label, confidence}."""
    probs = F.softmax(logits, dim=1)[0]
    top5_probs, top5_idx = torch.topk(probs, 5)
    return [
        {"label": classes[idx.item()], "confidence": round(prob.item() * 100, 2)}
        for prob, idx in zip(top5_probs, top5_idx)
    ]


def fgsm_attack(image_tensor, epsilon, data_grad):
    """Apply FGSM perturbation."""
    sign_data_grad = data_grad.sign()
    perturbed = image_tensor + epsilon * sign_data_grad
    # Re-normalize to valid range after perturbation
    perturbed = torch.clamp(perturbed, -3, 3)
    return perturbed


def pgd_attack(image_tensor, epsilon, alpha, num_iter):
    """Apply PGD (iterative) attack."""
    original = image_tensor.clone().detach()
    perturbed = image_tensor.clone().detach().requires_grad_(True)

    for _ in range(num_iter):
        if perturbed.grad is not None:
            perturbed.grad.zero_()
        perturbed = perturbed.detach().requires_grad_(True)
        output = model(perturbed)
        loss = F.cross_entropy(output, output.argmax(dim=1))
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            perturbed = perturbed + alpha * perturbed.grad.sign()
            delta = torch.clamp(perturbed - original, -epsilon, epsilon)
            perturbed = torch.clamp(original + delta, -3, 3)

    return perturbed.detach()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    epsilon = float(request.form.get("epsilon", 0.03))
    attack_type = request.form.get("attack", "fgsm")  # fgsm or pgd

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    # Preprocess
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    # Original prediction
    output = model(input_tensor)
    original_preds = get_top5(output)
    original_label = original_preds[0]["label"]

    # Compute gradient for FGSM
    loss = F.cross_entropy(output, output.argmax(dim=1))
    model.zero_grad()
    loss.backward()
    data_grad = input_tensor.grad.data

    # Apply attack
    if attack_type == "pgd":
        perturbed_tensor = pgd_attack(input_tensor.detach(), epsilon, alpha=epsilon/4, num_iter=10)
    else:
        perturbed_tensor = fgsm_attack(input_tensor, epsilon, data_grad)

    # Perturbed prediction
    with torch.no_grad():
        perturbed_output = model(perturbed_tensor)
    perturbed_preds = get_top5(perturbed_output)
    perturbed_label = perturbed_preds[0]["label"]

    # Compute amplified noise for visualization
    noise = (perturbed_tensor - input_tensor.detach())
    noise_amplified = (noise * 10 + 0.5).clamp(0, 1)
    noise_np = (noise_amplified.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    noise_pil = Image.fromarray(noise_np)
    noise_buffer = io.BytesIO()
    noise_pil.save(noise_buffer, format="PNG")
    noise_b64 = base64.b64encode(noise_buffer.getvalue()).decode("utf-8")

    return jsonify({
        "original_image": tensor_to_base64(input_tensor.detach()),
        "perturbed_image": tensor_to_base64(perturbed_tensor),
        "noise_image": noise_b64,
        "original_preds": original_preds,
        "perturbed_preds": perturbed_preds,
        "original_label": original_label,
        "perturbed_label": perturbed_label,
        "attack_succeeded": original_label != perturbed_label,
        "epsilon": epsilon,
        "attack_type": attack_type.upper()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

