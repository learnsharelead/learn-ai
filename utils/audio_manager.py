import streamlit as st
import os
from gtts import gTTS
import hashlib

# Ensure audio directory exists
AUDIO_DIR = "static/audio_cache"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR, exist_ok=True)

def text_to_speech(text, lang='en', slow=False):
    """
    Converts text to speech using Google TTS API.
    Caches the audio file based on the text hash to avoid redundant API calls.
    Returns the path to the audio file.
    """
    if not text or len(text.strip()) == 0:
        return None

    # Create a unique filename hash
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    filename = f"{text_hash}_{lang}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # Return cached path if exists
    if os.path.exists(filepath):
        return filepath

    try:
        # Generate new audio
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(filepath)
        return filepath
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

def listen_section(label, text_content):
    """
    Renders a button to generate/play audio for a specific text section.
    """
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        if st.button("🔊 Listen", key=f"btn_listen_{hash(label)}", help="Read this section out loud"):
            st.session_state[f"playing_{hash(label)}"] = True
    
    # Check if this specific section is set to play
    if st.session_state.get(f"playing_{hash(label)}"):
        with st.spinner(f"Generating audio for '{label}'..."):
            audio_file = text_to_speech(text_content)
            if audio_file:
                st.audio(audio_file, format='audio/mp3')
