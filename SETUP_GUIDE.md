# 🚀 AI Nexus Academy - Setup Guide

## Quick Start (5 Minutes)

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/ai-nexus-academy.git
cd ai-nexus-academy
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

That's it! The platform works in **Demo Mode** without any API keys.

---

## 🤖 Enable AI Features (Optional)

### Nexus Tutor (AI Assistant)

**Get a Free Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key (starts with `AIza...`)

**Configure:**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add:
GEMINI_API_KEY=AIzaSy...YOUR_KEY_HERE
```

**Free Tier Limits:**
- 60 requests/minute
- 1M tokens/day
- Perfect for learning!

---

## 📊 Enable Analytics (Optional)

### Google Analytics 4

**Setup:**
1. Go to [Google Analytics](https://analytics.google.com/)
2. Create a new GA4 property
3. Copy your Measurement ID (format: `G-XXXXXXXXXX`)

**Configure:**
```bash
# Add to .env
GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t ai-nexus-academy .

# Run
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  -e GA_MEASUREMENT_ID=your_id \
  ai-nexus-academy
```

---

## ☁️ Cloud Deployment

### Streamlit Community Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add secrets in "Advanced settings":
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   GA_MEASUREMENT_ID = "G-XXXXXXXXXX"
   ```
5. Deploy!

### Railway.app

1. Connect GitHub repo
2. Add environment variables
3. Deploy automatically

---

## 🔧 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### Nexus Tutor not responding
- Check `.env` file exists
- Verify `GEMINI_API_KEY` is set correctly
- Ensure `google-generativeai` is installed

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

---

## 📚 Project Structure

```
ai-nexus-academy/
├── app.py                 # Main application
├── modules/               # All learning modules
│   ├── introduction.py
│   ├── computer_vision.py
│   ├── ab_testing.py
│   └── ...
├── utils/                 # Helper functions
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
└── docs/                 # Documentation
```

---

## 🎯 Next Steps

1. **Explore the Curriculum** - Browse all 49 modules
2. **Try the Lab** - Model Arena, Prompt Lab, Code Playground
3. **Ask Nexus Tutor** - Get AI-powered help (if API key configured)
4. **Contribute** - Submit PRs to improve content

---

## 💡 Tips

- **No API Key?** Platform works fully in Demo Mode
- **Rate Limits?** Gemini free tier is generous for learning
- **Customization?** Fork the repo and add your own modules!

---

**Need Help?** Open an issue on GitHub or check the [STRATEGIC_REVIEW_2025.md](./STRATEGIC_REVIEW_2025.md) for detailed insights.
