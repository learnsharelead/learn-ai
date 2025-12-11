import streamlit as st

def show():
    st.title("👁️ Multi-Modal AI")
    
    st.markdown("""
    ### Beyond Text: Vision & Audio
    
    Modern models can see, hear, and speak. This opens up entirely new classes of applications.
    """)
    
    tabs = st.tabs([
        "🖼️ Computer Vision (LLMs)",
        "🗣️ Audio & Voice",
        "👨‍💻 Code Examples"
    ])
    
    # TAB 1: Vision
    with tabs[0]:
        st.header("🖼️ Vision-Language Models (VLMs)")
        
        st.markdown("""
        ### Models that "See"
        
        Leading models like **GPT-4o**, **Claude 3.5 Sonnet**, and **Gemini 1.5 Pro** are natively multi-modal.
        They can analyze images, diagrams, and screenshots.
        """)
        
        st.subheader("Use Cases")
        st.markdown("""
        - **Screen to Code:** Screenshot a website -> Get HTML/CSS.
        - **Medical Imaging:** Analyze X-rays (assistive).
        - **Data Extraction:** Photo of a receipt -> JSON data.
        - **Visual Q&A:** "Why is my plant dying?" (Upload photo of leaf).
        """)
    
    # TAB 2: Audio
    with tabs[1]:
        st.header("🗣️ Voice AI")
        
        st.subheader("1. ASR (Automatic Speech Recognition)")
        st.markdown("**Speech-to-Text.** Turning audio into transcripts.")
        st.info("Best Models: **OpenAI Whisper**, **Deepgram**, **Google Chirp**")
        
        st.subheader("2. TTS (Text-to-Speech)")
        st.markdown("**Voice Generation.** Turning text into lifelike audio.")
        st.info("Best Models: **ElevenLabs** (Industry standard), **OpenAI TTS**, **Play.ht**")
        
        st.subheader("3. Speech-to-Speech")
        st.markdown("Real-time conversational AI (like GPT-4o Voice Mode). Minimal latency.")

    # TAB 3: Code
    with tabs[2]:
        st.header("👨‍💻 Code Examples")
        
        st.subheader("Analyzing Images with GPT-4o")
        st.code('''
import base64
from openai import OpenAI

# 1. Encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("chart.jpg")

# 2. Call API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this chart. What is the trend?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("Transcribing Audio with Whisper")
        st.code('''
from openai import OpenAI

client = OpenAI()

audio_file = open("speech.mp3", "rb")

transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
)

print(transcript.text)
        ''', language="python")
