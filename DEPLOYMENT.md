# Deployment Guide for AI Tutorial System

## 🚀 Overview
This application is a **Streamlit** web app. It is lightweight, stateless, and easy to deploy to various cloud platforms.

## 📦 Option 1: Streamlit Cloud (Easiest)
Streamlit Cloud is free and connects directly to your GitHub repository.

1. **Push code to GitHub.**
2. **Go to** [share.streamlit.io](https://share.streamlit.io).
3. **Click** "New App".
4. **Select** your repository, branch, and main file path (`app.py`).
5. **Click** "Deploy"!

*Note: Ensure `requirements.txt` is in the root directory.*

---

## 🐳 Option 2: Docker (Recommended for Production)
Build a container to run anywhere (AWS, Google Cloud, Azure, Railway).

### 1. Build the Image
```bash
docker build -t ai-tutorial-system .
```

### 2. Run the Container
```bash
docker run -p 8501:8501 ai-tutorial-system
```

Access at `http://localhost:8501`.

---

## ☁️ Option 3: Cloud Platforms (Railway/Render)

### Railway
1. Connect GitHub repo.
2. Railway detects the `Dockerfile` automatically.
3. Deploy!

### Render
1. Create "Web Service".
2. Connect GitHub repo.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `streamlit run app.py`

---

## 🛠️ Configuration
- **Port:** The app runs on port `8501` by default.
- **Environment Variables:**
  - No secret keys required for the base version.
  - If using OpenAI/Gemini API keys later, set them in the cloud dashboard variables.

## 📚 Project Structure
```
/
├── app.py                  # Entry point
├── modules/                # Content pages
├── utils/                  # Helper functions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container config
└── README.md               # Documentation
```
