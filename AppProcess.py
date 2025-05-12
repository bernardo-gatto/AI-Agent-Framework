import subprocess
import time

# Launch AppVideo.py
video_process = subprocess.Popen(["python", "AppVideo_file.py"])

# Launch AppAudio.py
audio_process = subprocess.Popen(["python", "AppAudio_file.py"])

try:
    # Keep the main script running while processes are alive
    while True:
        # Optionally check if either process has terminated
        # If needed, handle restarts or cleanups
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    # Gracefully terminate if needed
    video_process.terminate()
    audio_process.terminate()
