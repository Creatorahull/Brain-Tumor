import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests

# --- CNN Architecture ---
class CancerCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(64, 4)  # 4 classes
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CancerCNN(num_features=1)
model.load_state_dict(torch.load("CNN_CANCER_GS_2.pth", map_location=device))
model.to(device)
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# --- Prediction Function ---
classes = ['glioma_tumor','meningioma_tumor','normal','pituitary_tumor']  # replace with your labels
def predict_image(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_class = torch.max(outputs, 1)
    return classes[predicted_class.item()]

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=1),
    title="Cancer Image Classification",
    description="Upload an image or provide a URL to predict its class."
)

iface.launch()

if __name__ == "__main__":
    interface.launch()