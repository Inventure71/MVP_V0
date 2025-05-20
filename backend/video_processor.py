import os
import numpy as np  # For image to array conversion
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import List, Dict, Any, Optional
import math
import traceback # For more detailed error logging

# --- MoviePy Imports ---
# Moved imports to top-level for clarity and earlier error detection
try:
    # Updated imports for MoviePy version 2.1.1
    import moviepy
    from moviepy import VideoClip, ImageClip, TextClip
    from moviepy import concatenate_videoclips, CompositeVideoClip, ColorClip
    from moviepy import AudioClip, AudioFileClip, concatenate_audioclips
    # Optional: If moviepy struggles to find ImageMagick, uncomment and set the path
    # from moviepy.config import change_settings
    # change_settings({"IMAGEMAGICK_BINARY": r"/path/to/magick"}) 
except ImportError:
    print("ERROR: moviepy library not found or import failed. Please install it: pip install moviepy")
    # Set required components to None or raise error to prevent execution
    ImageClip = None
    TextClip = None
    ColorClip = None
    CompositeVideoClip = None
    concatenate_videoclips = None
    AudioFileClip = None
    concatenate_audioclips = None
    AudioClip = None
    # Raise error to stop processing if moviepy is essential
    raise ImportError("MoviePy is required for video processing.")

# --- Constants --- (Adjust as needed)
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 24
TEXT_FONT_PATH = 'Arial.ttf' # Specify path if not in standard locations, or use default
DIALOGUE_FONT_SIZE = 36
DIALOGUE_TEXT_COLOR = 'white'
DIALOGUE_BG_COLOR = (0, 0, 0, 0.5) # Black with 50% opacity
DIALOGUE_POSITION = ('center', 'bottom')
DIALOGUE_PADDING = 20 # Padding around dialogue text box
MAX_DIALOGUE_WIDTH_FACTOR = 0.8 # Max width of dialogue box as fraction of video width
BACKGROUND_COLOR = (0, 0, 0) # Black background (RGB tuple)
IMAGE_PADDING = 30 # Padding around images
INTER_SCENE_DELAY = 0.5 # Seconds between scenes

def make_silence(duration, fps=44100):
    if AudioClip is None: raise ImportError("MoviePy AudioClip not loaded.")
    def make_frame_stereo_silence(t):
        num_samples = len(t) if isinstance(t, np.ndarray) else 1
        return np.zeros((num_samples, 2), dtype=np.float32)
    return AudioClip(make_frame_stereo_silence, duration=duration, fps=fps)

def _create_dialogue_clip(text: str, duration: float) -> Optional[TextClip]:
    """Creates a styled TextClip for dialogue, handling word wrap."""
    if TextClip is None: raise ImportError("MoviePy TextClip not loaded.")

    max_text_width = int(VIDEO_WIDTH * MAX_DIALOGUE_WIDTH_FACTOR - 2 * DIALOGUE_PADDING)

    # Use PIL to determine font and wrap text to calculate necessary size
    try:
        pil_font = ImageFont.truetype(TEXT_FONT_PATH, DIALOGUE_FONT_SIZE)
    except IOError:
        print(f"Warning: Font '{TEXT_FONT_PATH}' not found. Using default font for size calculation.")
        pil_font = ImageFont.load_default()

    # Estimate chars per line (less accurate for default font)
    avg_char_width = DIALOGUE_FONT_SIZE * 0.6
    chars_per_line = int(max_text_width / avg_char_width) if avg_char_width > 0 else 40

    wrapped_text = textwrap.fill(text, width=chars_per_line)

    # Create the TextClip using moviepy's text wrapping (method='caption')
    try:
        txt_clip = TextClip(
            font=TEXT_FONT_PATH,  # Font parameter is required
            text=wrapped_text,    # Text parameter
            color=DIALOGUE_TEXT_COLOR,
            bg_color=None,  # Use None instead of 'transparent'
            # Use size to constrain width, method='caption' handles height
            size=(max_text_width, None),
            method='label',   # Using 'label' method instead of 'caption'
            font_size=DIALOGUE_FONT_SIZE  # Use font_size parameter
        )

        # Add a semi-transparent background rectangle manually if needed
        # Or, if bg_color in TextClip supports rgba:
        # txt_clip = TextClip(...) # as above but with:
        # bg_color=f'rgba(0, 0, 0, {DIALOGUE_BG_COLOR[3]})' # Check moviepy docs for format

        # For manual background:
        txt_size = txt_clip.size
        bg_clip = ColorClip(
            size=(txt_size[0] + 2 * DIALOGUE_PADDING, txt_size[1] + 2 * DIALOGUE_PADDING),
            color=(0, 0, 0), # Black background
            is_mask=False,  # Changed from ismask to is_mask
            duration=duration
        ).with_opacity(DIALOGUE_BG_COLOR[3]) # Set opacity

        # Composite text on background
        dialogue_clip = CompositeVideoClip(
            [bg_clip.with_position('center'), txt_clip.with_position('center')],
            size=bg_clip.size
        )

        return dialogue_clip.with_duration(duration).with_position(DIALOGUE_POSITION)

    except Exception as e:
        print(f"Error creating TextClip for dialogue: '{text[:50]}...': {e}")
        traceback.print_exc()
        # Fallback: Create a blank clip of the correct duration?
        return ColorClip(size=(1,1), color=(0,0,0), duration=duration).with_opacity(0)

def _create_scene_visual(image_paths: List[str], duration: float) -> Optional[CompositeVideoClip]:
    """Creates the base visual clip for a scene (background + images)."""
    if ColorClip is None or ImageClip is None or CompositeVideoClip is None:
         raise ImportError("MoviePy components not loaded.")

    # Base background       
    background = ColorClip(size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=BACKGROUND_COLOR, duration=duration)
    clips_to_composite = [background]

    valid_images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                img_clip = ImageClip(img_path)
                valid_images.append(img_clip)
            except Exception as e:
                print(f"Warning: Failed to load image '{img_path}': {e}")
        else:
            print(f"Warning: Image file not found: '{img_path}'")

    num_images = len(valid_images)

    if num_images == 1:
        img = valid_images[0]
        # Resize to fit, maintaining aspect ratio
        img_resized = img.resized(height=VIDEO_HEIGHT - 2 * IMAGE_PADDING) # Fit height
        if img_resized.w > VIDEO_WIDTH - 2 * IMAGE_PADDING:
            img_resized = img.resized(width=VIDEO_WIDTH - 2 * IMAGE_PADDING) # Refit width if needed

        img_final = img_resized.with_duration(duration).with_position('center')
        clips_to_composite.append(img_final)
        print(f"   Layout: 1 image centered.")

    elif num_images >= 2:
        if num_images > 2:
            print(f"   Warning: More than 2 images requested for scene, using first 2.")
        img1 = valid_images[0]
        img2 = valid_images[1]

        # Target width for each image (half screen minus padding)
        target_w = (VIDEO_WIDTH - 3 * IMAGE_PADDING) // 2
        target_h = VIDEO_HEIGHT - 2 * IMAGE_PADDING

        # Resize both images
        img1_resized = img1.resized(width=target_w)
        if img1_resized.h > target_h:
            img1_resized = img1_resized.resized(height=target_h)

        img2_resized = img2.resized(width=target_w)
        if img2_resized.h > target_h:
            img2_resized = img2_resized.resized(height=target_h)

        # Position side-by-side
        pos1_x = IMAGE_PADDING
        pos1_y = 'center'
        pos2_x = IMAGE_PADDING + target_w + IMAGE_PADDING
        pos2_y = 'center'

        img1_final = img1_resized.with_duration(duration).with_position((pos1_x, pos1_y))
        img2_final = img2_resized.with_duration(duration).with_position((pos2_x, pos2_y))

        clips_to_composite.extend([img1_final, img2_final])
        print(f"   Layout: 2 images side-by-side.")
    else:
         print("   Layout: No valid images found for this scene.")


    # Create the composite clip for the scene's base visual
    scene_visual = CompositeVideoClip(clips_to_composite, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
    return scene_visual

def create_video_from_blocks(
    scenes: List[Dict[str, Any]], 
    processed_dialogue_blocks: List[Dict[str, Any]], 
    output_dir: str, # Expected to be results/{job_id}
    constant_delay: float = INTER_SCENE_DELAY,
    filename: str = "lesson_video.mp4"
) -> Optional[str]:
    """Creates a video from scenes, dialogue audio, and associated images.

    Args:
        scenes: List of scene dictionaries including "images_in_scene"
               [{ "scene_number": int, "content": str, "images_in_scene": List[str] }, ...]
        processed_dialogue_blocks: List of dialogue blocks with audio info
                                   [{ "speaker": str, "scene_number": int, "dialogue": str,
                                      "audio_path": Optional[str], "duration": Optional[float],
                                      "actions_or_visuals": str }, ...] # actions_or_visuals might contain IMG refs
        output_dir: Directory containing results for this job (e.g., "results/job_id").
                    Expected structure: output_dir/../annotated_images/
        constant_delay: Extra time (in seconds) to hold the visual scene after the last audio block for that scene.
        filename: Name for the output video file.

    Returns:
        The full path to the generated MP4 video file, or None if failed.
    """
    if ImageClip is None: # Check if Moviepy loaded
        print("Error: MoviePy components not loaded. Cannot create video.")
        return None

    output_path = os.path.join(output_dir, filename)
    # Derive image directory path relative to output_dir
    image_base_dir = os.path.join(output_dir, "annotated_images")
    # Correct path if output_dir is one level deeper (e.g. results/{job_id}/video)
    # image_base_dir = os.path.join(output_dir, "..", "annotated_images") # Alternative if structure differs
    if not os.path.isdir(image_base_dir):
         print(f"Warning: Annotated images directory not found at {image_base_dir}. Images will not be loaded.")
         # Decide whether to continue without images or fail
         # image_base_dir = None # Option: Set to None to skip image loading attempts


    if not scenes:
        print("Error: Cannot create video - No scenes provided.")
        return None
    if not processed_dialogue_blocks:
        print("Warning: No dialogue blocks provided. Video might be silent or very short.")

    all_final_scene_clips = []
    processed_audio_files = [] # Keep track of files to close later

    try:
        print(f"Processing {len(scenes)} scenes for video generation...")
        dialogue_by_scene = {}
        for block in processed_dialogue_blocks:
            scene_num = block.get("scene_number")
            if scene_num is not None:
                dialogue_by_scene.setdefault(scene_num, []).append(block)

        scenes.sort(key=lambda s: s["scene_number"])

        current_global_time = 0.0 # Keep track of absolute time for clip positioning

        for scene_info in scenes:
            scene_num = scene_info["scene_number"]
            # scene_content = scene_info.get("content", f"Scene {scene_num}") # Text content might be less relevant now
            image_labels = scene_info.get("images_in_scene", [])
            
            print(f"-- Processing Scene {scene_num} (Images: {image_labels}) --")

            scene_audio_blocks = dialogue_by_scene.get(scene_num, [])
            scene_timed_audio_clips = []
            scene_timed_dialogue_clips = []
            current_scene_time = 0.0

            # 1. Process audio blocks to get total duration and create timed clips
            total_scene_audio_duration = 0.0
            if not scene_audio_blocks:
                 print(f"   Warning: No dialogue blocks found for scene {scene_num}.")
            else:
                for block in scene_audio_blocks:
                    audio_path = block.get("audio_path")
                    duration = block.get("duration")
                    dialogue_text = block.get("dialogue", "").strip()

                    if audio_path and duration is not None and duration > 0 and os.path.exists(audio_path):
                        try:
                             audio_clip = AudioFileClip(audio_path)
                             # Ensure duration is not zero which can cause issues
                             actual_duration = max(0.01, audio_clip.duration)
                             # Corrected method name
                             audio_clip = audio_clip.with_start(current_scene_time)
                             scene_timed_audio_clips.append(audio_clip)
                             processed_audio_files.append(audio_clip) # Track for closing

                             # Create corresponding dialogue text clip
                             if dialogue_text:
                                 dialogue_visual_clip = _create_dialogue_clip(dialogue_text, actual_duration)
                                 if dialogue_visual_clip:
                                      # Use with_start instead of set_start
                                      dialogue_visual_clip = dialogue_visual_clip.with_start(current_scene_time)
                                      scene_timed_dialogue_clips.append(dialogue_visual_clip)

                             total_scene_audio_duration += actual_duration
                             current_scene_time += actual_duration
                             print(f"      Loaded audio: {os.path.basename(audio_path)} ({actual_duration:.2f}s), Time: {current_scene_time:.2f}s")
                        except Exception as e:
                             print(f"      Warning: Failed to load audio or create text clip for block: {e}")
                             traceback.print_exc()
                    else:
                        print(f"      Warning: Skipping dialogue block due to missing audio/duration: {block.get('dialogue', '')[:30]}...")

            # DEBUG: Check collected clips for the scene
            print(f"   DEBUG: Scene {scene_num} - Collected {len(scene_timed_audio_clips)} audio clips and {len(scene_timed_dialogue_clips)} dialogue visual clips.")
            
            # 2. Calculate scene visual duration
            scene_visual_duration = total_scene_audio_duration + constant_delay
            scene_visual_duration = max(constant_delay, scene_visual_duration) # Ensure minimum duration
            print(f"   Scene audio duration: {total_scene_audio_duration:.2f}s")
            print(f"   Scene visual duration (audio + delay): {scene_visual_duration:.2f}s")

            # 3. Create base scene visual (background + images)
            image_full_paths = []
            if image_base_dir: # Only proceed if directory exists
                for label in image_labels:
                    # Construct filename from label like "[IMG_1]" -> "IMG_1.png" (guess extension or check pdf_utils)
                    # Assuming png for now, might need adjustment if extensions vary
                    potential_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"] # Add more if needed
                    found_img = False
                    img_base_name = label[1:-1] # Extract "IMG_1"
                    for ext in potential_extensions:
                         img_file = f"{img_base_name}{ext}"
                         full_path = os.path.join(image_base_dir, img_file)
                         if os.path.exists(full_path):
                              image_full_paths.append(full_path)
                              found_img = True
                              break # Found one extension
                    if not found_img:
                         print(f"Warning: Could not find image file for label {label} in {image_base_dir}")


            scene_base_visual = _create_scene_visual(image_full_paths, scene_visual_duration)
            if not scene_base_visual:
                print(f"   Error: Failed to create base visual for scene {scene_num}. Skipping scene.")
                continue # Skip this scene if base visual fails

            # 4. Composite dialogue text clips onto the base visual
            clips_for_final_scene = [scene_base_visual] + scene_timed_dialogue_clips
            final_scene_visual = CompositeVideoClip(clips_for_final_scene, size=(VIDEO_WIDTH, VIDEO_HEIGHT))

            # 5. Combine audio for the scene (dialogue + potential silence at end)
            scene_final_audio = None
            if scene_timed_audio_clips:
                try:
                    print(f"   DEBUG: Scene {scene_num} - Attempting to concatenate {len(scene_timed_audio_clips)} audio clips.") # DEBUG
                    scene_final_audio = concatenate_audioclips(scene_timed_audio_clips)
                    print(f"   DEBUG: Scene {scene_num} - Audio concatenated. Duration: {scene_final_audio.duration:.2f}s") # DEBUG
                except Exception as e:
                    print(f"   Warning: Failed to concatenate audio clips for scene {scene_num}: {e}")
                    # Create silence for the expected duration as fallback?
                    scene_final_audio = make_silence(total_scene_audio_duration)
                    print(f"   DEBUG: Scene {scene_num} - Created silence clip. Duration: {scene_final_audio.duration:.2f}s") # DEBUG
            elif not scene_final_audio: # If no audio blocks, create silence for full duration
                scene_final_audio = make_silence(scene_visual_duration)
                print(f"   DEBUG: Scene {scene_num} - Created silence clip. Duration: {scene_final_audio.duration:.2f}s") # DEBUG

            # DEBUG: Check final audio duration before setting
            final_audio_duration_to_set = scene_final_audio.duration if scene_final_audio else 0.0
            print(f"   DEBUG: Scene {scene_num} - Final audio duration before setting: {final_audio_duration_to_set:.2f}s")

            # 6. Set audio for the final scene visual clip
            final_scene_clip = None  # Initialize outside conditionals
            if scene_final_audio:
                try:
                    # Ensure audio duration matches visual duration precisely before setting
                    if abs(scene_final_audio.duration - scene_visual_duration) > 0.01:
                        print(f"   Warning: Scene {scene_num} audio duration ({scene_final_audio.duration:.2f}s) differs from visual duration ({scene_visual_duration:.2f}s). Adjusting audio duration.")
                        # Use individual audio clip methods instead of composite method
                        if hasattr(scene_final_audio, 'subclip'):
                            scene_final_audio = scene_final_audio.subclip(0, scene_visual_duration)
                        else:
                            # For CompositeAudioClip that doesn't have subclip method
                            # Re-extract the audio clips and trim them if needed
                            print(f"   Warning: Audio clip type {type(scene_final_audio)} doesn't have subclip method.")
                            if isinstance(scene_final_audio, moviepy.audio.AudioClip.CompositeAudioClip) and scene_timed_audio_clips:
                                # Create a new CompositeAudioClip with the appropriate duration
                                from_clips = scene_final_audio.clips
                                # Keep only clips that start before the target duration
                                filtered_clips = [clip for clip in from_clips if clip.start < scene_visual_duration]
                                
                                # Create new list for updated clips
                                updated_clips = []
                                for clip in filtered_clips:
                                    if clip.start + clip.duration > scene_visual_duration:
                                        # Trim clip to end at scene_visual_duration
                                        if hasattr(clip, 'subclip'):
                                            end_time = scene_visual_duration - clip.start
                                            updated_clip = clip.subclip(0, end_time)
                                            updated_clips.append(updated_clip)
                                    else:
                                        # Keep clip as is if it already fits
                                        updated_clips.append(clip)
                                
                                # If we have clips to combine after filtering
                                if updated_clips:
                                    scene_final_audio = concatenate_audioclips(updated_clips)
                                    print(f"      Created new composite audio with duration: {scene_final_audio.duration:.2f}s")
                                else:
                                    # Fallback if no clips remain after filtering
                                    scene_final_audio = make_silence(scene_visual_duration)
                                    print(f"      Created silence clip with duration: {scene_visual_duration:.2f}s")
                            else:
                                # Generic fallback - create silent clip with appropriate duration
                                scene_final_audio = make_silence(scene_visual_duration)
                                print(f"      Created silence clip with duration: {scene_visual_duration:.2f}s")

                    # Set the audio on the final clip - this will include actual audio
                    final_scene_clip = final_scene_visual.with_audio(scene_final_audio)
                    print(f"   Scene {scene_num} audio set successfully.")
                except Exception as e:
                    print(f"   Error setting audio for scene {scene_num}: {e}. Scene will be silent.")
                    traceback.print_exc()
                    final_scene_clip = final_scene_visual.with_audio(None) # Fallback to silent
            else:
                print(f"   Scene {scene_num} will be silent (no audio to set).")
                final_scene_clip = final_scene_visual.with_audio(None)

            # Add the scene clip to our collection
            all_final_scene_clips.append(final_scene_clip)
            current_global_time += scene_visual_duration # Update global time marker

        # --- Final Video Assembly ---
        if not all_final_scene_clips:
            print("Error: No valid scene clips were generated. Cannot create video.")
            return None
            
        print("\nConcatenating final scene clips...")
        final_video = concatenate_videoclips(all_final_scene_clips)

        print(f"Writing final video to: {output_path}")
        # Use standard codecs, consider adding audio_codec='aac'
        final_video.write_videofile(
            output_path,
            fps=VIDEO_FPS,
            codec='libx264',
            audio_codec='aac',
            threads=8,          # Good setting for a MacBook Pro
            preset='superfast',
            temp_audiofile=os.path.join(output_dir, 'temp-audio.m4a'), # Explicit temp file
            remove_temp=True,
            logger='bar' # Progress bar
        )
        print("Video writing complete.")
        return output_path

    except Exception as e:
        print(f"\n---! Error during video creation !---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {e}")
        traceback.print_exc() # Print full traceback
        return None

    finally:
        # --- Cleanup ---
        print("Closing processed audio files...")
        for af in processed_audio_files:
            try:
                # Check if close method exists and call it
                if hasattr(af, 'close') and callable(af.close):
                     af.close()
            except Exception as e_close:
                # Log error but don't stop execution
                print(f"   Warning: Error closing audio file: {e_close}")
        print("Cleanup finished.") 