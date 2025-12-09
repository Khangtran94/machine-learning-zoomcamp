import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import torchvision.transforms as transforms

# ----------------------------
# Preprocessing
# ----------------------------
def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    img = Image.open(BytesIO(buffer))
    return img

def preprocess(img):
    img = prepare_image(img)
    img = train_transforms(img)           # tensor (3,H,W)
    img = img.numpy().astype(np.float32)  # convert to numpy
    img = np.expand_dims(img, axis=0)     # add batch dim -> (1,3,H,W)
    return img

# ----------------------------
# Load ONNX model
# ----------------------------
session = ort.InferenceSession("hair_classifier_v1.onnx")
# session = ort.InferenceSession("/app/hair_classifier_v1.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ----------------------------
# Prediction
# ----------------------------
def predict(url):
    img = download_image(url)
    x = preprocess(img)
    pred = session.run([output_name], {input_name: x})
    return float(pred[0][0][0])  # adjust if output shape differs

# ----------------------------
# Quick test
# ----------------------------
if __name__ == "__main__":
    test_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    result = predict(test_url)
    print("Prediction:", result)
