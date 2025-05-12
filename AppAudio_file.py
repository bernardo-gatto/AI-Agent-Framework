import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pyttsx3
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
import torch
import librosa
import speech_recognition as sr
from groq import Groq

# ------------------------
# 1) READ THE VIDEO EMOTION
# ------------------------
def read_video_emotion(filename="resultVideo.txt"):
    """
    Reads the 'resultVideo.txt' file, which should have a line:
        video_emotion: xxxxx
    Returns the emotion (e.g., "neutral") if found, otherwise returns None.
    """
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')
            if len(lines) < 1:
                return None
            # Expect a line like "video_emotion: neutral"
            line = lines[0].strip()
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                if key == "video_emotion":
                    return val
    except:
        pass
    
    return None

# ------------------------
# Initialize Components
# ------------------------

# Load the pre-trained model and tokenizer for speech recognition
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load the pre-trained model for emotion recognition from Hugging Face
audio_emotion_model = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
text_emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Initialize the recognizer
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = True
recognizer.energy_threshold = 300
recognizer.pause_threshold = 2.0  # stops after 2 seconds of silence

# Get the default microphone
microphone = sr.Microphone()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Example: set the desired voice (e.g., Microsoft Zira Desktop on Windows)
voices = engine.getProperty('voices')
for voice in voices:
    if 'zira' in voice.name.lower():  # You can customize this condition
        engine.setProperty('voice', voice.id)
        break

# Initialize Groq client with the API key
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ------------------------
# Helper Functions
# ------------------------

def recognize_speech():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Say something (stops if you're silent for ~2 seconds):")
        audio = recognizer.listen(source, timeout=None)
        print("Finished listening, processing audio...")

        # Save raw audio to WAV for further processing
        audio_data = audio.get_wav_data()
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)

    # Load and preprocess the audio
    speech, rate = librosa.load("temp_audio.wav", sr=16000)

    # Tokenize the audio
    input_values = tokenizer(speech, return_tensors="pt").input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    print("You said: " + transcription)
    return transcription, "temp_audio.wav"


def get_lmm_response(prompt):
    context = "Please provide advice related to mental or physical health. Make it shorter, no more than 2 phrases."
    full_prompt = f"{context}\n{prompt}"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": full_prompt}],
        model="llama3-70b-8192",
    )
    response = chat_completion.choices[0].message.content
    print("LMM response: " + response)
    return response


def extract_audio_emotion(audio_path):
    speech, rate = librosa.load(audio_path, sr=16000)
    results = audio_emotion_model(speech)
    emotion = results[0]['label']
    print("Detected audio emotion: " + emotion)
    return emotion


def extract_text_emotion(text):
    results = text_emotion_model(text)
    # Find the emotion with the highest score
    emotion = max(results[0], key=lambda x: x['score'])['label']
    print("Detected text emotion: " + emotion)
    return emotion

# ------------------------
# MAIN LOOP
# ------------------------

# We can read the video emotion just once outside the loop, or inside the loop
# if we expect it to change frequently. Usually it's set once, so we do it here:
video_emotion = read_video_emotion()
if video_emotion:
    print("Detected video emotion (from resultVideo.txt): " + video_emotion)
else:
    print("No valid video_emotion found in resultVideo.txt.")

while True:
    # 1) Recognize speech
    text, audio_path = recognize_speech()
    
    # 2) Check if the user wants to finish
    if "finish recording" in text.lower():
        print("Finishing recording.")
        break
    
    # 3) Extract audio and text emotions
    audio_emotion = extract_audio_emotion(audio_path)
    text_emotion = extract_text_emotion(text)
    
    # 4) Write the resultAudio file
    #    Format: 
    #       audio_emotion: neutral
    #       text_emotion: neutral
    with open("resultAudio.txt", "w") as f:
        f.write(f"audio_emotion: {audio_emotion}\n")
        f.write(f"text_emotion: {text_emotion}\n")
    print("Wrote audio emotions to resultAudio.txt.")

    # 5) Get a response from the language model
    response = get_lmm_response(text)
    
    # 6) Include all three (video, audio, text) emotions in the final spoken response
    #    (if video_emotion is not found, it might be None)
    response_with_emotion = (
        f"[Detected Video Emotion: {video_emotion if video_emotion else 'N/A'}]\n"
        f"[Detected Audio Emotion: {audio_emotion}]\n"
        f"[Detected Text Emotion: {text_emotion}]\n"
        f"{response}"
    )
    
    # 7) Synthesize and speak the response
    engine.say(response_with_emotion)
    engine.runAndWait()
