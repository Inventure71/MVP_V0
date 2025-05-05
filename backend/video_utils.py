from moviepy.video.VideoClip import TextClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
import os
import uuid

def create_video_from_scenes(scenes, audio_path, output_dir):
    audio = AudioFileClip(audio_path)
    duration_per_scene = audio.duration / max(1, len(scenes))
    clips = []
    for scene in scenes:
        txt = scene['text']
        clip = TextClip(txt, fontsize=48, color='white', bg_color='black', size=(1280, 720), method='caption')
        clip = clip.set_duration(duration_per_scene)
        clips.append(clip)
    video = concatenate_videoclips(clips)
    video = video.set_audio(audio)
    video_filename = f"lesson_{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    video.write_videofile(video_path, fps=24)
    return video_path 