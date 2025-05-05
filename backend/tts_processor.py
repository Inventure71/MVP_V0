from gtts import gTTS
import os
from typing import List, Dict, Any, Optional
import uuid # To generate unique filenames

# Define a subdirectory within the job results for audio blocks
AUDIO_BLOCK_SUBDIR = "audio_blocks"


def _generate_single_tts_file(text: str, output_path: str) -> Optional[float]:
    """Helper function to generate a single TTS file and return its duration."""
    if not text or not text.strip():
        print("Warning: TTS received empty text. Skipping audio generation for this block.")
        return None
    # Generate file and then determine duration
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Generating TTS audio block... saving to {output_path}")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
        print(f"TTS audio block generated: {output_path}")

        # Attempt to get accurate duration via MoviePy
        words = text.split()
        avg_word_duration = 0.4
        try:
            from moviepy import AudioFileClip
            try:
                clip = AudioFileClip(output_path)
                duration = clip.duration
                clip.close()
                return duration
            except Exception as e:
                print(f"Warning: Could not load audio file for duration via MoviePy: {e}")
        except ModuleNotFoundError:
            print("Warning: moviepy package not found, falling back to duration estimation.")
        # Fallback estimation if MoviePy is not available or failed
        estimated = len(words) * avg_word_duration
        print(f"Estimated duration for {output_path}: {estimated:.2f}s ({len(words)} words)")
        return estimated

    except Exception as e:
        print(f"Error generating TTS or duration for {output_path}: {e}")
        return None

def generate_audio_blocks(dialogue_blocks: List[Dict[str, Any]], job_result_dir: str) -> List[Dict[str, Any]]:
    """
    Generates separate audio files for each dialogue block and gathers metadata.

    Args:
        dialogue_blocks: List of dialogue block dictionaries from Gemini.
                         Expected keys: "speaker", "scene_number", "dialogue".
        job_result_dir: The main result directory for the current job.

    Returns:
        A list of dictionaries, each corresponding to a dialogue block,
        with added keys "audio_path" (str or None) and "duration" (float or None).
        Example: [{
            "speaker": "Teacher", 
            "scene_number": 1, 
            "dialogue": "...", 
            "audio_path": "results/job_id/audio_blocks/block_0_teacher.mp3", 
            "duration": 3.45
        }, ...]
    """
    
    audio_output_dir = os.path.join(job_result_dir, AUDIO_BLOCK_SUBDIR)
    os.makedirs(audio_output_dir, exist_ok=True)

    processed_blocks = []
    
    if not dialogue_blocks:
         print("Warning: No dialogue blocks provided for TTS generation.")
         return []

    print(f"Generating {len(dialogue_blocks)} audio blocks in {audio_output_dir}...")
    for i, block in enumerate(dialogue_blocks):
        dialogue_text = block.get("dialogue", "")
        speaker = block.get("speaker", "unknown").lower()
        scene_num = block.get("scene_number", "noscn")

        # Create a somewhat descriptive filename
        # Use index to ensure uniqueness even if text/speaker/scene is identical
        filename = f"block_{i}_scn{scene_num}_{speaker}_{uuid.uuid4().hex[:6]}.mp3"
        output_path = os.path.join(audio_output_dir, filename)

        duration = _generate_single_tts_file(dialogue_text, output_path)

        # Create a new dict with the original block data plus audio info
        # Even if generation fails, include the block to keep lists aligned
        processed_info = block.copy()
        processed_info["audio_path"] = output_path if duration is not None else None
        processed_info["duration"] = duration
        processed_blocks.append(processed_info)

        if duration is None:
            print(f"Warning: Failed to generate or get duration for dialogue block {i} (Scene {scene_num}, Speaker {speaker}).")
            # Decide if this should halt the process or just continue without audio for this block
            # For now, continue but the video generator will need to handle None paths/durations

    print(f"Finished generating audio blocks. {sum(1 for b in processed_blocks if b['audio_path'])} successful.")
    return processed_blocks


# --- Keep old function for potential single use or comment out/remove ---
# def text_to_speech(text: str, output_dir: str, filename: str = "dialogue_audio.mp3") -> str:
#     """Converts text to speech using gTTS and saves it as an MP3 file.
#     ... (rest of old function)
#     """ 