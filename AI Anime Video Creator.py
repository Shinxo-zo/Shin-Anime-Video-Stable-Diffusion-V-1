
# Anime Video Creator (Working CPU Version)
# Step 1: Install Verified Dependencies
!pip install -q opencv-python moviepy gtts requests imageio

# Step 2: Download Working Anime Model
!wget https://github.com/bryandlee/animegan2-pytorch/raw/main/weights/face_paint_512_v2.pt
!wget https://raw.githubusercontent.com/bryandlee/animegan2-pytorch/main/model.py

# Step 3: Import Libraries
import torch
import cv2
import numpy as np
from moviepy.editor import *
from gtts import gTTS
from PIL import Image
from model import Generator

# Step 4: Initialize AnimeGAN Model
device = torch.device("cpu")
model = Generator().to(device)
model.load_state_dict(torch.load("face_paint_512_v2.pt", map_location=device))
model.eval()

# Step 5: Anime Conversion Function
def convert_to_anime(frame):
    img = Image.fromarray(frame).convert("RGB")
    with torch.no_grad():
        out = model(torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float()/255.)
    return (out.squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)

# Step 6: Video Processing Pipeline
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        anime_frame = convert_to_anime(frame_rgb)
        writer.write(cv2.cvtColor(anime_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    writer.release()

# Step 7: Create Sample Video
!ffmpeg -f lavfi -i testsrc=duration=5:size=640x480:rate=30 input.mp4 -y

# Process Video (Takes 2-3 minutes for 5s clip)
process_video("input.mp4", "anime_output.mp4")

# Step 8: Add Audio
def add_voiceover(video_path, text):
    tts = gTTS(text=text, lang='en')
    tts.save("voiceover.mp3")

    video = VideoFileClip(video_path)
    audio = AudioFileClip("voiceover.mp3")
    return video.set_audio(audio)

final = add_voiceover("anime_output.mp4", "This is my anime creation!")
final.write_videofile("final_anime.mp4", codec="libx264", audio_codec="aac")

print("✅ Success! Final video saved as final_anime.mp4")