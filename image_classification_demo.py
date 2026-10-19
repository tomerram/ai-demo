import torch
import requests
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import gradio as gr


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
# Download human-readable labels for ImageNet
response = requests.get("https://git.io/JJkYN")
labels = [l.strip() for l in response.text.split("\n") if l.strip()]

# Define image preprocessing (IMPORTANT for ResNet)
transform = transforms.Compose([ transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(inp):
    # preprocess image
    inp = transform(inp).unsqueeze(0)

    # ensure model runs in inference mode
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)

    # map predictions to labels
    confidences = {
        labels[i]: float(prediction[i])
        for i in range(len(labels))
    }

    return confidences

gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["lion.jpg", "cheetah.webp"]).launch()