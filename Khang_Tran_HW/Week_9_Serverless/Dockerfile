FROM agrigorev/model-2025-hairstyle:v1

# Set working directory
WORKDIR /app

# Copy Python code and model into /app
COPY Hair_Classifier_Model.py /app/
COPY hair_classifier_v1.onnx /app/
COPY hair_classifier_v1.onnx.data /app/

# Install required packages (CPU-only PyTorch)
RUN pip install --no-cache-dir numpy pillow onnxruntime \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
