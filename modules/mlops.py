import streamlit as st
import json

def show():
    st.title("🚀 MLOps & Model Deployment")
    
    st.markdown("""
    Building a model is only 10% of the work. Deploying and maintaining it is the other 90%!
    Learn how to take models from notebook to production.
    """)
    
    tabs = st.tabs([
        "📦 Model Serialization",
        "🌐 Flask/FastAPI",
        "🐳 Docker Basics",
        "☁️ Cloud Deployment",
        "📊 MLOps Pipeline"
    ])
    
    # TAB 1: Model Serialization
    with tabs[0]:
        st.header("📦 Saving & Loading Models")
        
        st.markdown("""
        Before deploying, you need to **serialize** (save) your trained model.
        """)
        
        st.subheader("1. Pickle (Python's Built-in)")
        
        st.code("""
import pickle
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use it
prediction = loaded_model.predict(X_new)
        """, language="python")
        
        st.warning("⚠️ **Security Note:** Never unpickle files from untrusted sources!")
        
        st.subheader("2. Joblib (Recommended for Sklearn)")
        
        st.code("""
import joblib

# Save (more efficient for large numpy arrays)
joblib.dump(model, 'model.joblib')

# Load
model = joblib.load('model.joblib')
        """, language="python")
        
        st.subheader("3. ONNX (Cross-Platform)")
        
        st.code("""
# Convert sklearn model to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
        """, language="python")
        
        st.info("💡 **ONNX** allows you to train in Python and deploy in C++, Java, JavaScript, etc.")
        
        st.subheader("Comparison Table")
        
        st.markdown("""
        | Format | Pros | Cons | Best For |
        |--------|------|------|----------|
        | **Pickle** | Built-in, simple | Python-only, security risks | Quick prototypes |
        | **Joblib** | Efficient for numpy | Python-only | Sklearn models |
        | **ONNX** | Cross-platform | Setup complexity | Production |
        | **TensorFlow SavedModel** | TF ecosystem | TF-only | Deep learning |
        | **PyTorch .pt** | PyTorch native | PyTorch-only | Deep learning |
        """)
    
    # TAB 2: Flask/FastAPI
    with tabs[1]:
        st.header("🌐 Building ML APIs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Flask (Simple)")
            st.code("""
# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    prediction = model.predict([features])
    return jsonify({
        'prediction': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
            """, language="python")
            
        with col2:
            st.subheader("FastAPI (Modern, Async)")
            st.code("""
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.joblib')

class PredictRequest(BaseModel):
    features: list[float]

@app.post('/predict')
def predict(req: PredictRequest):
    prediction = model.predict([req.features])
    return {
        'prediction': prediction.tolist()
    }

# Run: uvicorn main:app --reload
            """, language="python")
        
        st.subheader("Testing Your API")
        
        st.code("""
# Using requests library
import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={'features': [5.1, 3.5, 1.4, 0.2]}
)

print(response.json())
# {'prediction': [0]}  # Iris setosa
        """, language="python")
        
        st.subheader("Flask vs FastAPI")
        
        st.markdown("""
        | Feature | Flask | FastAPI |
        |---------|-------|---------|
        | Speed | Slower | 2-3x faster (async) |
        | Docs | Manual | Auto-generated Swagger |
        | Validation | Manual | Built-in Pydantic |
        | Learning Curve | Easy | Easy-Medium |
        | Best For | Simple APIs | Production APIs |
        """)
        
        st.success("💡 **Recommendation:** Use FastAPI for new projects!")
    
    # TAB 3: Docker
    with tabs[2]:
        st.header("🐳 Containerization with Docker")
        
        st.markdown("""
        **Docker** packages your app + dependencies into a container that runs anywhere.
        
        > "It works on my machine" → "It works everywhere"
        """)
        
        st.subheader("1. Dockerfile")
        
        st.code("""
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
        """, language="dockerfile")
        
        st.subheader("2. requirements.txt")
        
        st.code("""
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
joblib==1.3.2
pydantic==2.5.2
        """, language="text")
        
        st.subheader("3. Docker Commands")
        
        st.code("""
# Build the image
docker build -t ml-api:v1 .

# Run the container
docker run -d -p 8000:8000 ml-api:v1

# Check running containers
docker ps

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>
        """, language="bash")
        
        st.subheader("4. Docker Compose (Multi-Container)")
        
        st.code("""
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.joblib
      
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
        """, language="yaml")
        
        st.info("💡 Use Docker Compose when you have multiple services (API + Database + Cache)")
    
    # TAB 4: Cloud Deployment
    with tabs[3]:
        st.header("☁️ Cloud Deployment Options")
        
        st.subheader("Quick Comparison")
        
        st.markdown("""
        | Platform | Difficulty | Cost | Best For |
        |----------|------------|------|----------|
        | **Streamlit Cloud** | ⭐ Easy | Free | Demos, POCs |
        | **Hugging Face Spaces** | ⭐ Easy | Free | ML demos |
        | **Railway** | ⭐⭐ Easy | Free tier | Small APIs |
        | **Render** | ⭐⭐ Easy | Free tier | Web apps |
        | **AWS Lambda** | ⭐⭐⭐ Medium | Pay-per-use | Serverless |
        | **Google Cloud Run** | ⭐⭐⭐ Medium | Pay-per-use | Containers |
        | **AWS SageMaker** | ⭐⭐⭐⭐ Hard | Expensive | Enterprise ML |
        """)
        
        st.subheader("1. Streamlit Cloud (Easiest)")
        
        st.code("""
# 1. Push your app to GitHub
# 2. Go to share.streamlit.io
# 3. Connect your repo
# 4. Deploy!

# That's it! Free hosting for Streamlit apps.
        """, language="text")
        
        st.subheader("2. Google Cloud Run")
        
        st.code("""
# Install gcloud CLI first

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/ml-api

# Deploy to Cloud Run
gcloud run deploy ml-api \\
    --image gcr.io/PROJECT_ID/ml-api \\
    --platform managed \\
    --region us-central1 \\
    --allow-unauthenticated
        """, language="bash")
        
        st.subheader("3. AWS Lambda (Serverless)")
        
        st.code("""
# Using AWS SAM or Serverless Framework

# serverless.yml
service: ml-api

provider:
  name: aws
  runtime: python3.10

functions:
  predict:
    handler: handler.predict
    events:
      - http:
          path: predict
          method: post
        """, language="yaml")
    
    # TAB 5: MLOps Pipeline
    with tabs[4]:
        st.header("📊 MLOps Pipeline Overview")
        
        st.markdown("""
        **MLOps** = DevOps + ML. It's about automating the entire ML lifecycle.
        """)
        
        st.subheader("The MLOps Lifecycle")
        
        st.graphviz_chart("""
        digraph MLOps {
            rankdir=LR;
            node [shape=box, style=filled];
            
            Data [label="1. Data\\nCollection", fillcolor=lightyellow];
            Features [label="2. Feature\\nEngineering", fillcolor=lightblue];
            Train [label="3. Model\\nTraining", fillcolor=lightgreen];
            Evaluate [label="4. Model\\nEvaluation", fillcolor=orange];
            Deploy [label="5. Model\\nDeployment", fillcolor=lightpink];
            Monitor [label="6. Model\\nMonitoring", fillcolor=lavender];
            
            Data -> Features -> Train -> Evaluate -> Deploy -> Monitor;
            Monitor -> Data [label="Retrain", style=dashed];
        }
        """)
        
        st.subheader("Key MLOps Tools")
        
        st.markdown("""
        | Stage | Tools |
        |-------|-------|
        | **Data Versioning** | DVC, LakeFS, Delta Lake |
        | **Experiment Tracking** | MLflow, Weights & Biases, Neptune |
        | **Feature Store** | Feast, Tecton, Hopsworks |
        | **Model Registry** | MLflow, Vertex AI, SageMaker |
        | **Model Serving** | TensorFlow Serving, TorchServe, Seldon |
        | **Monitoring** | Evidently AI, WhyLabs, Arize |
        | **Orchestration** | Airflow, Kubeflow, Prefect |
        """)
        
        st.subheader("MLflow Example")
        
        st.code("""
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
        """, language="python")
        
        st.subheader("Model Monitoring")
        
        st.markdown("""
        **Why Monitor?** Models degrade over time due to:
        
        1. **Data Drift**: Input distribution changes
        2. **Concept Drift**: Relationship between X and Y changes
        3. **Feature Drift**: Feature values change
        
        **What to Monitor:**
        - Prediction latency
        - Error rates
        - Feature distributions
        - Prediction distributions
        - Business metrics (conversions, etc.)
        """)
        
        st.success("""
        💡 **Best Practice:** Set up alerts for when metrics exceed thresholds.
        Automate retraining when drift is detected!
        """)
