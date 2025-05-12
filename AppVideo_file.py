from deepface import DeepFace
import cv2
from collections import Counter
import os

# A list to keep track of the last 30 emotions from video
video_emotions = []

def get_dominant_emotion(emotions_list):
    """
    Given a list of emotion labels, return the most common (dominant) emotion.
    If the list is empty, return None.
    """
    if not emotions_list:
        return None
    emotion_counter = Counter(emotions_list)
    # most_common(1) returns a list like [(emotion, count)]
    return emotion_counter.most_common(1)[0][0]


def read_audio_emotions(filename="resultAudio.txt"):
    """
    Reads the resultAudio.txt file which should have:
      line 1: audio_emotion: xxxxx
      line 2: text_emotion: xxxxx

    Returns a tuple (audio_emotion, text_emotion) if successful, or (None, None) if failed.
    """
    if not os.path.exists(filename):
        return None, None  # File doesn't exist

    try:
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')
            if len(lines) < 2:
                return None, None  # Not enough lines

            # Expect lines in the format "audio_emotion: xxxxx"
            audio_line = lines[0].strip()
            text_line = lines[1].strip()

            # Parse out the emotion part after the colon
            audio_emotion = audio_line.split(':', 1)[1].strip() if ':' in audio_line else None
            text_emotion = text_line.split(':', 1)[1].strip() if ':' in text_line else None

            return audio_emotion, text_emotion
    except:
        return None, None

# --- MAIN CODE ---

cap = cv2.VideoCapture(0)  # or replace 0 with a video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame to speed up analysis
    height, width = frame.shape[:2]
    small_frame = cv2.resize(frame, (width // 2, height // 2))

    try:
        # DeepFace analyze
        result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
        if result:
            # Current frame's recognized emotion
            current_emotion = result[0]["dominant_emotion"]
            
            # Keep track of last 30 emotions
            video_emotions.append(current_emotion)
            if len(video_emotions) > 30:
                video_emotions.pop(0)

            # Find the dominant emotion of the last 30 frames
            dominant_emotion_30 = get_dominant_emotion(video_emotions)

            if dominant_emotion_30:
                # Write the dominant emotion to resultVideo.txt
                with open("resultVideo.txt", "w") as f:
                    f.write(f"video_emotion: {dominant_emotion_30}\n")

                # Display the text on the video frame
                # Making it look more "professional"
                font = cv2.FONT_HERSHEY_COMPLEX
                text = f"Video Emotion: {dominant_emotion_30}"
                cv2.putText(small_frame, text, (50, 50), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Attempt to read audio/text emotions from file
        audio_emotion, text_emotion = read_audio_emotions("resultAudio.txt")
        if audio_emotion and text_emotion:
            # If successfully read, display them
            font = cv2.FONT_HERSHEY_COMPLEX
            audio_text = f"Audio Emotion: {audio_emotion}"
            text_text = f"Text Emotion: {text_emotion}"

            # Display below the video emotion line
            cv2.putText(small_frame, audio_text, (50, 100), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(small_frame, text_text, (50, 150), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    except ValueError as e:
        # Handle the case when no face is detected
        print(f"No face detected: {e}")
    
    # Show the frame
    cv2.imshow('Emotion Recognition', small_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
