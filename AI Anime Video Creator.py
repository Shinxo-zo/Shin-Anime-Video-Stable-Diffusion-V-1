# Colab‑Ready: Anime Video + Voice + Music + SFX

# 1) Install dependencies
!pip install -q diffusers accelerate transformers torch moviepy bark[all] safetensors

# 2) Imports & setup
import torch
from diffusers import StableVideoDiffusionPipeline
from bark import generate_audio
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip,
    concatenate_videoclips, ImageSequenceClip
)
import numpy as np
from PIL import Image

# 3) Load the Stable Video Diffusion pipeline (img2vid-xt)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    variant="fp16", torch_dtype=torch.float16
).to("cuda")
pipe.enable_model_cpu_offload()

# 4) Helper to generate and save a short anime‑style clip
def generate_scene_video(prompt, out_path, num_frames=14, height=576, width=1024, seed=42):
    cond_img = Image.new("RGB", (width, height), color=(15,15,30))
    generator = torch.manual_seed(seed)
    result = pipe(
        image=cond_img,
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    )
    pil_frames = result.frames[0]  # list of PIL images
    frame_arrays = [np.array(fr) for fr in pil_frames]
    clip = ImageSequenceClip(frame_arrays, fps=7)
    clip.write_videofile(out_path, codec="libx264", audio=False)
    return out_path

# 5) Generate Scene 1 & Scene 2
scene1 = generate_scene_video(
    "Cinematic anime: Kurenai Shu from Beyblade in a stormy forest, dramatic lighting, Ufotable/A1 Pictures style, poised to launch Spriggan Requiem",
    "scene1.mp4"
)
scene2 = generate_scene_video(
    "Haunting anime: Tokisaki Kurumi in gothic lolita dress hidden in rain‑soaked forest, emotional expression, cinematic Ufotable style",
    "scene2.mp4"
)

# 6) Generate voice lines with Bark TTS
generate_audio("I am Kurenai Shu, witness my power!", history_prompt="priax_male", output_path="shu.wav")
generate_audio("Please... don't go!", history_prompt="paras_xp", output_path="kurumi.wav")

# 7) Generate background music & SFX
generate_audio("Dramatic orchestral music with thunder and wind for 30 seconds", history_prompt="melody_music", output_path="music1.wav")
generate_audio("Haunting piano melody with rain ambiance for 30 seconds", history_prompt="melody_music", output_path="music2.wav")
generate_audio("Heavy rain with distant thunder for 30 seconds", history_prompt="sound_effect", output_path="rain.wav")
generate_audio("Mechanical click and whoosh of a Beyblade launcher", history_prompt="sound_effect", output_path="bey.wav")

# 8) Assemble each scene with its audio layers
def assemble(video_in, audio_clips, out_file):
    vid = VideoFileClip(video_in)
    audio = CompositeAudioClip(audio_clips)
    vid = vid.set_audio(audio)
    vid.write_videofile(out_file, codec="libx264", audio_codec="aac")
    return out_file

scene1_final = assemble("scene1.mp4", [
    AudioFileClip("music1.wav").volumex(0.6),
    AudioFileClip("rain.wav").volumex(0.4),
    AudioFileClip("bey.wav").volumex(0.8),
    AudioFileClip("shu.wav").set_start(2)
], "scene1_final.mp4")

scene2_final = assemble("scene2.mp4", [
    AudioFileClip("music2.wav").volumex(0.6),
    AudioFileClip("rain.wav").volumex(0.4),
    AudioFileClip("kurumi.wav").set_start(1)
], "scene2_final.mp4")

# 9) Concatenate both scenes into the final video
final = concatenate_videoclips([
    VideoFileClip(scene1_final),
    VideoFileClip(scene2_final)
])
final.write_videofile("final_anime_video.mp4", codec="libx264", audio_codec="aac")

print("✨ Done! Download 'final_anime_video.mp4' from the Files panel.")