import os
import numpy as np  # For image to array conversion
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import List, Dict, Any, Optional
import math

# --- MoviePy Imports ---
# Moved imports to top-level for clarity and earlier error detection
try:
    from moviepy import (
        VideoClip, # Base class, might be needed depending on usage pattern, safe to include
        ImageClip, 
        TextClip, # Although TextClip isn't directly used now, keep for potential future use or remove if certain
        concatenate_videoclips, 
        CompositeVideoClip,
        ColorClip # Used implicitly perhaps? Keep for safety or remove if certain
    )
    from moviepy.audio.AudioClip import AudioClip
    # Import specific functions needed
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.audio.tools.cuts import find_audio_period # Example if needed
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip # If creating from image sequences
    # Import concatenate_audioclips directly if available at top level
    from moviepy.audio.AudioClip import concatenate_audioclips 

    # Optional: If moviepy struggles to find ImageMagick, uncomment and set the path
    # from moviepy.config import change_settings
    # change_settings({"IMAGEMAGICK_BINARY": r"/path/to/magick"}) 
except ModuleNotFoundError:
    print("ERROR: moviepy library not found. Please install it: pip install moviepy")
    # Depending on desired behavior, you could raise the error, exit, 
    # or set flags to disable moviepy functionality gracefully elsewhere.
    # For now, we print the error. Functions using moviepy will likely fail later.
    pass # Allow the rest of the module to load, functions will fail at runtime if moviepy was needed.

# --- Constants --- (Adjust as needed)
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 24
TEXT_FONT = 'Arial' # Ensure this font is available on the system
TEXT_FONTSIZE = 40
TEXT_COLOR = 'white'
BACKGROUND_COLOR = 'black'
TEXT_PADDING = 40 # Increased padding for better margins
MAX_TEXT_LINES = 10 # Limit lines to prevent text overflowing
INTER_SCENE_DELAY = 0.5 # Seconds of silence/pause between scenes (after last audio of a scene)

def make_silence(duration, fps=44100):
    # Ensure AudioClip was imported at the top level
    if 'AudioClip' not in globals():
        print("Error: AudioClip not available (moviepy import failed?). Cannot make silence.")
        raise ImportError("moviepy.audio.AudioClip.AudioClip not loaded.")
    
    # Define a function that returns stereo zero samples for a given time chunk t
    # The input t can be a float or a numpy array of times.
    # The function should return a numpy array of shape (len(t_indices), 2)
    def make_frame_stereo_silence(t):
        # Determine the number of samples requested. If t is a single float, it's 1 sample.
        # If t is an array, it's the length of the array.
        if isinstance(t, np.ndarray):
            num_samples = len(t)
        else: # Assuming t is a scalar (float/int)
            num_samples = 1 
        return np.zeros((num_samples, 2), dtype=np.float32) # Explicit float32 dtype

    return AudioClip(make_frame_stereo_silence, duration=duration, fps=fps)

    # --- Previous attempts ---
    # Explicitly create a 2-channel (stereo) silent array (Might cause issues with chunking)
    # n_samples = int(fps * duration)
    # silence_array = np.zeros((n_samples, 2), dtype=np.float32)
    # return AudioClip(lambda t: silence_array, duration=duration, fps=fps)
    # Original potentially problematic version:
    # return AudioClip(lambda t: np.zeros_like(t), duration=duration, fps=fps)

def _create_text_image(text: str, width: int, height: int) -> np.ndarray:
    """Renders text onto a background image using PIL and returns as numpy array."""
    img = Image.new("RGB", (width, height), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        pil_font = ImageFont.truetype(TEXT_FONT, TEXT_FONTSIZE)
    except IOError:
        print(f"Warning: Font '{TEXT_FONT}' not found. Using default font.")
        pil_font = ImageFont.load_default() # Default bitmap font, size might differ
        # Recalculate max_chars for default font if possible, or use a safe default
        # For simplicity, we'll keep the original max_chars calculation but it might be less accurate
    except Exception as e:
        print(f"Warning: Error loading font '{TEXT_FONT}': {e}. Using default font.")
        pil_font = ImageFont.load_default()

    # Estimate character width (crude)
    avg_char_width = TEXT_FONTSIZE * 0.6
    max_chars = int((width - 2 * TEXT_PADDING) / avg_char_width) if avg_char_width > 0 else 50
    
    lines = textwrap.wrap(text, width=max_chars, max_lines=MAX_TEXT_LINES, placeholder=' [...]')
    
    # Calculate vertical starting position to center the text block roughly
    total_text_height = len(lines) * (TEXT_FONTSIZE + 5) # +5 for line spacing
    y_text = max(TEXT_PADDING, (height - total_text_height) // 2)
    
    for line in lines:
        try:
             # Use getbbox for potentially better positioning/width calculation if needed
             # left, top, right, bottom = draw.textbbox((TEXT_PADDING, y_text), line, font=pil_font)
             # draw.text(((width - (right-left)) // 2 , y_text), line, font=pil_font, fill=TEXT_COLOR) # Centered line
             draw.text((TEXT_PADDING, y_text), line, font=pil_font, fill=TEXT_COLOR) # Left-aligned
        except Exception as e:
            print(f"Error drawing text line '{line[:30]}...': {e}")
        y_text += TEXT_FONTSIZE + 5 # Line spacing
        if y_text > height - TEXT_PADDING: # Stop if text overflows vertically
            break
            
    return np.array(img)

def create_video_from_blocks(
    scenes: List[Dict[str, Any]], 
    processed_dialogue_blocks: List[Dict[str, Any]], 
    output_dir: str, 
    constant_delay: float = INTER_SCENE_DELAY,
    filename: str = "lesson_video.mp4"
) -> Optional[str]:
    """Creates a video from scenes and synchronized audio blocks.

    Args:
        scenes: List of scene dictionaries [{ "scene_number": int, "content": str }, ...]
        processed_dialogue_blocks: List of dialogue blocks with audio info
                                   [{ "speaker": str, "scene_number": int, "dialogue": str,
                                      "audio_path": Optional[str], "duration": Optional[float] }, ...]
        output_dir: Directory to save the final video.
        constant_delay: Extra time (in seconds) to hold the visual scene after the last audio block for that scene.
        filename: Name for the output video file.

    Returns:
        The full path to the generated MP4 video file, or None if failed.
    """
    output_path = os.path.join(output_dir, filename)

    # Dynamically import required MoviePy functions from the top-level package
    # try:
    #     from moviepy import TextClip, concatenate_videoclips, concatenate_audioclips, AudioFileClip, ImageClip, ColorClip, CompositeVideoClip
    #     from moviepy.audio.AudioClip import AudioClip
    # except ModuleNotFoundError as e:
    #     print(f"Error: moviepy import failed: {e}. Please install moviepy.")
    #     return None
    
    # Check if necessary moviepy components were loaded at the top level
    required_moviepy_funcs = ['ImageClip', 'concatenate_videoclips', 'concatenate_audioclips', 'AudioFileClip', 'AudioClip']
    missing_funcs = [func for func in required_moviepy_funcs if func not in globals()]
    if missing_funcs:
        print(f"Error: Missing required moviepy components: {missing_funcs}. Moviepy might not be installed correctly.")
        return None

    if not scenes:
        print("Error: Cannot create video - No scenes provided.")
        return None
    if not processed_dialogue_blocks:
        print("Warning: No dialogue blocks provided. Creating a silent video with scene content.")
        # Allow silent video creation if desired, otherwise return None
        # return None 

    all_visual_clips = []
    all_audio_clips = []
    processed_audio_files = [] # Keep track of files to close later

    try:
        print(f"Processing {len(scenes)} scenes for video generation...")
        # Group dialogue blocks by scene number for easier lookup
        dialogue_by_scene = {}
        for block in processed_dialogue_blocks:
            scene_num = block.get("scene_number")
            if scene_num is not None:
                if scene_num not in dialogue_by_scene:
                    dialogue_by_scene[scene_num] = []
                dialogue_by_scene[scene_num].append(block)

        # Process scenes in their intended order
        scenes.sort(key=lambda s: s["scene_number"]) # Ensure scenes are sorted

        for scene_info in scenes:
            scene_num = scene_info["scene_number"]
            scene_content = scene_info.get("content", f"Scene {scene_num}") # Fallback content
            
            print(f"-- Processing Scene {scene_num} --")

            # Find associated audio blocks for this scene
            scene_audio_blocks = dialogue_by_scene.get(scene_num, [])
            current_scene_audio_clips = []
            total_scene_audio_duration = 0.0

            if not scene_audio_blocks:
                 print(f"   Warning: No dialogue blocks found for scene {scene_num}.")
                 # Decide duration for scenes without audio (e.g., fixed short duration?)
                 # Let's use the constant_delay as the minimum duration for now.
                 total_scene_audio_duration = 0.0 
            else:
                print(f"   Found {len(scene_audio_blocks)} dialogue blocks for scene {scene_num}.")
                for block in scene_audio_blocks:
                    audio_path = block.get("audio_path")
                    duration = block.get("duration")
                    if audio_path and duration is not None and duration > 0 and os.path.exists(audio_path):
                        try:
                             audio_clip = AudioFileClip(audio_path)
                             current_scene_audio_clips.append(audio_clip)
                             processed_audio_files.append(audio_clip) # Track for closing
                             total_scene_audio_duration += audio_clip.duration # Use actual clip duration
                             print(f"      Loaded audio: {os.path.basename(audio_path)} ({audio_clip.duration:.2f}s)")
                        except Exception as e:
                             print(f"      Warning: Failed to load audio file {audio_path}: {e}")
                    else:
                        print(f"      Warning: Skipping dialogue block due to missing audio or invalid duration: {block}")
            
            # Calculate visual duration for this scene
            scene_visual_duration = total_scene_audio_duration + constant_delay
            # Ensure a minimum duration, e.g., the delay itself, or slightly more
            scene_visual_duration = max(constant_delay, scene_visual_duration) 
            print(f"   Scene audio duration: {total_scene_audio_duration:.2f}s")
            print(f"   Scene visual duration (audio + delay): {scene_visual_duration:.2f}s")

            # Create visual clip (Text on background)
            scene_frame = _create_text_image(scene_content, VIDEO_WIDTH, VIDEO_HEIGHT)
            # Ensure the numpy array is in a format moviepy expects (e.g., uint8)
            scene_frame_np = np.array(scene_frame, dtype=np.uint8)
            # --- DEBUG --- 
            print(f"   DEBUG: scene_frame_np shape: {scene_frame_np.shape}, dtype: {scene_frame_np.dtype}")
            # --- END DEBUG ---
            visual_clip = ImageClip(scene_frame_np, duration=scene_visual_duration)
            all_visual_clips.append(visual_clip)

            # --- SINGLE CLIP WRITE TEST (Temporary) --- REMOVED
            # if scene_num == scenes[0]["scene_number"]: # Only test for the first scene
            #     try:
            #         test_output_path = os.path.join(output_dir, f"_test_scene_{scene_num}.mp4")
            #         print(f"   DEBUG: Attempting to write single clip: {test_output_path}")
            #         visual_clip.write_videofile(
            #             test_output_path,
            #             codec='libx264',
            #             fps=VIDEO_FPS,
            #             logger='bar'
            #         )
            #         print(f"   DEBUG: Single clip write successful.")
            #     except Exception as e_test:
            #         print(f"   DEBUG: Single clip write FAILED: {e_test}")
            # --- END SINGLE CLIP WRITE TEST ---

            # Concatenate audio for this scene + silence
            if current_scene_audio_clips:
                print(f"   DEBUG: Processing audio for Scene {scene_num}...") # DEBUG
                if 'concatenate_audioclips' not in globals():
                     print("Error: concatenate_audioclips not available (moviepy import failed?).")
                     raise ImportError("moviepy.audio.AudioClip.concatenate_audioclips not loaded.")
                
                print(f"      DEBUG: Concatenating {len(current_scene_audio_clips)} scene audio clips...") # DEBUG
                scene_audio_concat = concatenate_audioclips(current_scene_audio_clips)
                print(f"      DEBUG: Scene audio concatenated. Duration: {scene_audio_concat.duration:.2f}s") # DEBUG
                
                # Add silence if needed to match visual duration
                silence_needed = scene_visual_duration - scene_audio_concat.duration
                print(f"      DEBUG: Silence needed: {silence_needed:.2f}s") # DEBUG
                if silence_needed > 0.01: # Add tolerance
                    print(f"      DEBUG: Making silence clip ({silence_needed:.2f}s)...") # DEBUG
                    silence = make_silence(silence_needed)
                    print(f"      DEBUG: Silence clip created. Duration: {silence.duration:.2f}s") # DEBUG
                    print(f"      DEBUG: Concatenating scene audio with silence...") # DEBUG
                    final_scene_audio = concatenate_audioclips([scene_audio_concat, silence])
                    print(f"      DEBUG: Scene audio + silence concatenated. Duration: {final_scene_audio.duration:.2f}s") # DEBUG
                    print(f"      Added {silence_needed:.2f}s silence to scene audio.")
                else:
                     final_scene_audio = scene_audio_concat
                     print(f"      DEBUG: No silence concatenation needed.") # DEBUG
                
                # Ensure audio doesn't exceed visual duration due to rounding? Trim if necessary.
                # Use subclip instead of set_duration, as CompositeAudioClip might not have set_duration
                if abs(final_scene_audio.duration - scene_visual_duration) > 0.01: # Only trim if needed
                    print(f"      DEBUG: Trimming final scene audio from {final_scene_audio.duration:.3f}s to {scene_visual_duration:.3f}s using subclip...") # DEBUG
                    final_scene_audio = final_scene_audio.subclip(0, scene_visual_duration)
                    print(f"      DEBUG: Final scene audio trimmed. New duration: {final_scene_audio.duration:.3f}s") # DEBUG
                else:
                     print(f"      DEBUG: Final scene audio duration ({final_scene_audio.duration:.3f}s) is close enough to target ({scene_visual_duration:.3f}s). No trim needed.") # DEBUG
            else: # No audio for this scene, create silence for the full visual duration
                print(f"   DEBUG: No audio clips for Scene {scene_num}. Making silence for full duration {scene_visual_duration:.2f}s...") # DEBUG
                final_scene_audio = make_silence(scene_visual_duration)
                print(f"   DEBUG: Full silence clip created. Duration: {final_scene_audio.duration:.2f}s") # DEBUG
                print("      Using silence for scene audio.")
            
            all_audio_clips.append(final_scene_audio)
            print(f"   DEBUG: Appended final audio for Scene {scene_num}. Total audio clips: {len(all_audio_clips)}") # DEBUG
            print(f"-- Finished Scene {scene_num} --")

        # Concatenate all visual and audio clips
        if not all_visual_clips:
            print("Error: No visual clips were generated.")
            return None
            
        print("DEBUG: About to concatenate video clips...") # DEBUG
        final_video_clip = concatenate_videoclips(all_visual_clips, method="compose")
        print(f"DEBUG: Video concatenation done. Duration: {final_video_clip.duration:.2f}s") # DEBUG
        
        print("Concatenating final audio clips...")
        if all_audio_clips:
             # Ensure concatenate_audioclips was imported
             if 'concatenate_audioclips' not in globals():
                  print("Error: concatenate_audioclips not available (moviepy import failed?).")
                  raise ImportError("moviepy.audio.AudioClip.concatenate_audioclips not loaded.")
             
             print("DEBUG: About to concatenate audio clips...") # DEBUG
             final_audio_clip = concatenate_audioclips(all_audio_clips)
             print(f"DEBUG: Audio concatenation done. Duration: {final_audio_clip.duration:.2f}s") # DEBUG

             # Set the concatenated audio to the final video
             print(f"Setting final audio (Duration: {final_audio_clip.duration:.2f}s)")
             print("DEBUG: About to set audio...") # DEBUG
             final_video_clip.audio = final_audio_clip
             print("DEBUG: Set audio done.") # DEBUG
        else:
            print("Warning: No audio clips were generated for the final video.")
            # Video will be silent

        print(f"Final video duration: {final_video_clip.duration:.2f}s")

        # Write the final video file
        print(f"Writing final video to: {output_path}")
        final_video_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            preset='fast',
            fps=VIDEO_FPS,
            threads=4,
            logger='bar'
        )

        print("Video creation successful.")
        return output_path

    except Exception as e:
        print(f"Error creating video: {e}")
        return None
    finally:
        # Clean up: Close all opened audio file clips
        print(f"Closing {len(processed_audio_files)} opened audio files...")
        for af_clip in processed_audio_files:
            try:
                af_clip.close()
            except Exception as e_close:
                print(f"Warning: Error closing audio clip: {e_close}")
        # Moviepy recommends final_clip.close() but sometimes it causes issues after write_videofile
        # if 'final_video_clip' in locals() and hasattr(final_video_clip, 'close'):
        #      try:
        #          final_video_clip.close()
        #      except Exception as e_close:
        #          print(f"Warning: Error closing final video clip: {e_close}")
        pass # Avoid closing potentially already closed clips

# --- Old Function (Remove or Comment Out) ---
# def create_video(scenes: List[Dict[str, str]], audio_path: str, output_dir: str, filename: str = "lesson_video.mp4") -> str:
#     ...

# def create_video(scenes: List[Dict[str, str]], audio_path: str, output_dir: str, filename: str = "lesson_video.mp4") -> str:
#     """Creates a video from scenes (text overlays) synchronized with audio.
#
#     Args:
#         scenes: A list of scene dictionaries with 'text_overlay'.
#         audio_path: Path to the dialogue audio file (MP3).
#         output_dir: Directory to save the final video.
#         filename: Name for the output video file.
#
#     Returns:
#         The full path to the generated MP4 video file.
#         Returns None if video creation fails.
#     """
#     output_path = os.path.join(output_dir, filename)
#
#     if not scenes:
#         raise ValueError("Cannot create video: No scenes provided.")
#     if not audio_path or not os.path.exists(audio_path):
#         # If no audio, we could potentially create a silent video, 
#         # but for now, let's assume audio is required.
#         raise ValueError("Cannot create video: Audio file not found or not provided.")
#
#     try:
#         # 1. Load Audio
#         print(f"Loading audio: {audio_path}")
#         audio_clip = AudioFileClip(audio_path)
#         total_audio_duration = audio_clip.duration
#         print(f"Audio duration: {total_audio_duration:.2f} seconds")
#
#         if total_audio_duration <= 0:
#             raise ValueError("Audio clip has zero or negative duration.")
#
#         # 2. Calculate Scene Durations (simple equal division)
#         num_scenes = len(scenes)
#         if num_scenes == 0:
#             raise ValueError("Cannot create video with zero scenes.")
#         
#         scene_duration = total_audio_duration / num_scenes
#         print(f"Calculated duration per scene: {scene_duration:.2f} seconds")
#
#         # 3. Create Text Clips for Each Scene
#         video_clips = []
#         current_start_time = 0
#         print(f"Generating {num_scenes} scene clips...")
#         for i, scene in enumerate(scenes):
#             text = scene.get('text_overlay', '')
#             # Use visual description slightly (e.g., add it below overlay?)
#             # visual_desc = scene.get('visual_description', '') 
#             # For now, just using text_overlay
#
#             # Ensure text clip duration doesn't exceed total audio duration cumulative error
#             clip_end_time = min(current_start_time + scene_duration, total_audio_duration)
#             actual_clip_duration = max(0.01, clip_end_time - current_start_time) # Ensure non-zero duration
#             
#             # Handle the last clip precisely ending at the audio duration
#             if i == num_scenes - 1:
#                 actual_clip_duration = max(0.01, total_audio_duration - current_start_time)
#
#             print(f"  Scene {i}: Duration={actual_clip_duration:.2f}s, Text=\"{text[:50]}...\"")
#
#             # Render text to an image using PIL
#             # Create background image
#             img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), BACKGROUND_COLOR)
#             draw = ImageDraw.Draw(img)
#             # Load a truetype font, fallback to default if unavailable
#             try:
#                 pil_font = ImageFont.truetype(TEXT_FONT, TEXT_FONTSIZE)
#             except Exception:
#                 pil_font = ImageFont.load_default()
#             # Wrap text based on approximate character width
#             max_chars = int((VIDEO_WIDTH - 2 * TEXT_PADDING) / (TEXT_FONTSIZE * 0.6))
#             lines = textwrap.wrap(text, width=max_chars)
#             y_text = TEXT_PADDING
#             for line in lines:
#                 draw.text((TEXT_PADDING, y_text), line, font=pil_font, fill=TEXT_COLOR)
#                 y_text += TEXT_FONTSIZE + 5  # Line spacing
#             # Convert to numpy array and create an ImageClip
#             import numpy as _np
#             from moviepy import ImageClip
#             frame = _np.array(img)
#             clip = ImageClip(frame, duration=actual_clip_duration)
#             video_clips.append(clip)
#             current_start_time += actual_clip_duration
#             
#             # Safety break if cumulative time exceeds audio, though logic should prevent this
#             if current_start_time > total_audio_duration * 1.01: # Add small tolerance
#                  print(f"Warning: Cumulative clip time ({current_start_time}) exceeded audio duration ({total_audio_duration}). Stopping clip generation.")
#                  break 
#
#         # 4. Concatenate Clips
#         print("Concatenating video clips...")
#         final_clip = concatenate_videoclips(video_clips, method="compose")
#
#         # 5. Add Audio
#         print("Setting audio...")
#         final_clip.audio = audio_clip # Correct way: Assign to the .audio attribute
#
#         # 6. Write Video File
#         print(f"Writing video file to: {output_path}")
#         # Use codecs suitable for web playback. Default libx264 for video and aac for audio are usually good.
#         # Preset 'fast' provides a balance between speed and file size/quality.
#         # threads=4 can speed up encoding if multiple cores are available.
#         final_clip.write_videofile(
#             output_path,
#             codec='libx264',
#             audio_codec='aac',
#             temp_audiofile='temp-audio.m4a', # Recommended by moviepy docs
#             remove_temp=True,
#             preset='fast',
#             fps=VIDEO_FPS,
#             threads=4, # Adjust based on system cores
#             logger='bar' # Show progress bar
#         )
#
#         # Close clips to release resources
#         audio_clip.close()
#         for clip in video_clips:
#              clip.close()
#         final_clip.close()
#
#         print("Video creation successful.")
#         return output_path
#
#     except ImportError as e:
#          print(f"Moviepy Import Error: {e}. Ensure moviepy and its dependencies (like Pillow, numpy) are installed.")
#          raise
#     except OSError as e:
#          print(f"OS Error during video processing: {e}. Check file paths and permissions.")
#          print("This might also be related to missing dependencies like ImageMagick or ffmpeg.")
#          raise
#     except Exception as e:
#         print(f"Error creating video: {e}")
#         # Clean up potentially open clips in case of error
#         if 'audio_clip' in locals() and hasattr(audio_clip, 'close'): audio_clip.close()
#         if 'video_clips' in locals():
#             for clip in video_clips:
#                  if hasattr(clip, 'close'): clip.close()
#         if 'final_clip' in locals() and hasattr(final_clip, 'close'): final_clip.close()
#         raise Exception(f"Failed to create video: {e}") 