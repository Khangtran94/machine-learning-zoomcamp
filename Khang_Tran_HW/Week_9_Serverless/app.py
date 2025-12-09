from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from Hair_Classifier_Model import predict, prepare_image, preprocess, session, input_name, output_name
from PIL import Image
import io
import numpy as np
import base64

app = FastAPI(title="Hair Classifier API")

# -------------------------
# HTML UI for testing
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head>
        <title>Hair Classifier</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            .container { max-width: 600px; }
            .result { margin-top: 20px; padding: 20px; background: #f0f0f0; }
            img { max-width: 300px; border: 1px solid #ccc; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Hair Classifier API</h1>
        <div class="container">
            <h2>Upload Image</h2>
            <form id="uploadForm">
                <input type="file" id="fileInput" accept="image/*" required>
                <button type="submit">Predict</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const file = document.getElementById('fileInput').files[0];
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/predict/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `
                    <div class="result">
                        <h3>Prediction: ${(data.prediction * 100).toFixed(2)}%</h3>
                        <p>Score: ${data.prediction.toFixed(4)}</p>
                        <img src="data:image/jpeg;base64,${data.image_base64}" alt="Uploaded Image">
                    </div>
                `;
            });
        </script>
    </body>
    </html>
    """

# -------------------------
# Input via image URL
# -------------------------
class ImageURL(BaseModel):
    url: str

@app.post("/predict/url")
def predict_url(image: ImageURL):
    try:
        raw_score = predict(image.url)
        # Apply sigmoid transformation to convert logit to probability
        score = 1 / (1 + np.exp(-raw_score))
        return {"prediction": float(score)}  # Ensure it's JSON serializable
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# Input via file upload
# -------------------------
# @app.post("/predict/upload")
# def predict_upload(file: UploadFile = File(...)):
#     content = file.file.read()
#     img = Image.open(io.BytesIO(content))
#     x = preprocess(img)
#     pred = session.run([output_name], {input_name: x})
#     raw_score = float(pred[0][0][0])
    
#     # Apply sigmoid transformation to convert logit to probability
#     score = 1 / (1 + np.exp(-raw_score))
    
#     return {"prediction": score}

@app.post("/predict/upload")
def predict_upload(file: UploadFile = File(...)):
    try:
        # Read and open the uploaded image
        content = file.file.read()
        img = Image.open(io.BytesIO(content))

        # Preprocess for the model
        x = preprocess(img)
        pred = session.run([output_name], {input_name: x})
        raw_score = float(pred[0][0][0])

        # Apply sigmoid to get probability
        score = 1 / (1 + np.exp(-raw_score))

        # Convert image to Base64 for JSON
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return both prediction and image
        return {
            "prediction": score,
            "image_base64": img_base64
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
