# AI Agent Framework

This repository provides a general framework for an AI agent that combines **audio and video emotion detection**, **speech recognition**, **text-based emotion analysis**, and **language model interaction**. It serves as a starting point for applications in real-time emotion-aware systems, conversational AI, or personalized user experiences.


Our framework for a multimodal AI agent performs:

    Real-time video emotion detection (using DeepFace).
    Audio speech-to-text processing and audio emotion recognition.
    Text emotion detection for user-provided speech.
    LLM-based responses via Groq.
    Text-to-speech output with pyttsx3.

This system monitors webcam video for facial emotions, simultaneously listens for user speech and classifies its emotion, and also processes the user’s spoken text for emotional content. Finally, it provides a short LLM-based response incorporating the recognized emotional states, then speaks that response out loud.

**FEATURES**

    VIDEO EMOTION RECOGNITION
        Uses DeepFace to analyze frames from a live video feed (or webcam) and detects the dominant emotion (e.g., happy, sad, neutral).
        Keeps track of a window of recent emotions for more stable results.

    AUDIO SPEECH & EMOTION DETECTION
        Listens in real time using the microphone via SpeechRecognition.
        Transcribes speech with a Wav2Vec2 model (facebook/wav2vec2-base-960h).
        Classifies the audio emotion (e.g., neutral, happy, angry) using a pretrained emotion recognition model (ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition).

    TEXT EMOTION RECOGNITION
        Processes the transcribed text using a huggingface pipeline to classify the text’s emotional tone (e.g., sadness, joy, fear).

    LLM INTEGRATION VIA GROQ
        Sends user’s transcribed text to a large language model for a short, context-appropriate response.
        Currently configured for llama3-70b-8192 (adjust as needed).

    TEXT-TO-SPEECH OUTPUT
        Speaks back the LLM’s short advice or message to the user using pyttsx3.

**REQUIREMENTS**

deepface==0.0.92

groq==0.18.0

librosa==0.10.2

opencv_contrib_python==4.10.0.84

opencv_python==4.11.0.86

pyttsx3==2.90

SpeechRecognition==3.12.0

torch==2.4.1

transformers==4.44.0

Note: You may need additional system-level dependencies (for example, ffmpeg for librosa, or a Windows/Linux TTS engine for pyttsx3).

**INSTALLATION**

    Clone the repository: git clone https://github.com/yourusername/your-repo-name.git cd your-repo-name

    Create a virtual environment (recommended): python -m venv venv source venv/bin/activate (Linux/macOS) OR venv\Scripts\activate (Windows)

    Install Python dependencies: pip install -r requirements.txt

    Or install them individually with: pip install deepface==0.0.92 groq==0.18.0 librosa==0.10.2.post1 ...

    Set up Groq API Key (if you plan to use the LLM): export GROQ_API_KEY="your_groq_api_key_here" (Adjust for your OS.)

**USAGE**

    Run the main script: python AppProcess.py This will:
        Launch AppVideo_file.py for webcam-based emotion detection.
        Launch AppAudio_file.py for speech recognition and audio emotion detection.
        Keep reading resultVideo.txt and resultAudio.txt to gather the dominant emotions.

    Interaction:
        The camera will analyze your facial expressions in real time.
        Speak into your microphone. The system transcribes your speech, classifies your audio/text emotions, and stores the results in resultAudio.txt.
        After each utterance, it prompts a short response from the LLM, then speaks it via TTS.

    Exiting:
        Press Ctrl + C in the terminal to stop.
        Or say "finish recording" to signal a graceful finish (depending on code implementation).

**CONFIGURATION**

    Groq Model: In main.py, look for the get_lmm_response function. You can change the model name in client.chat.completions.create(...).

    Text-to-Speech Voice: Near the pyttsx3.init() call, specify which voice you want by adjusting the loop that checks for voice.id.

    Thresholds and Timings: In AppAudio_file.py (or wherever your main code is), adjust recognizer.energy_threshold or recognizer.pause_threshold to suit your environment and speaking style.
