from gtts import gTTS
import os
from typing import List, Dict, Any, Optional
import uuid # To generate unique filenames
import time
from mutagen.mp3 import MP3 # To get audio duration
import boto3 # For AWS Polly
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from dotenv import load_dotenv

# Load environment variables (consider moving if needed elsewhere)
load_dotenv()

# Define a subdirectory within the job results for audio blocks
AUDIO_BLOCK_SUBDIR = "audio_blocks"

# AWS Polly Configuration (Defaults)
POLLY_VOICE_ID = 'Danielle' # Use Danielle voice
POLLY_ENGINE = 'generative' # Generative voices require this engine
POLLY_OUTPUT_FORMAT = 'mp3'

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

def _get_audio_duration(file_path: str) -> Optional[float]:
    """Calculates the duration of an audio file (MP3) using mutagen."""
    try:
        audio = MP3(file_path)
        return audio.info.length
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return None

def generate_audio_blocks(
    dialogue_blocks: List[Dict[str, Any]], 
    job_result_dir: str, 
    tts_engine: str = "gtts" # Default to gtts
) -> List[Dict[str, Any]]:
    """
    Generates audio files for each dialogue block using the specified TTS engine.

    Args:
        dialogue_blocks: List of dialogue blocks from Gemini.
        job_result_dir: The main result directory for the current job.
        tts_engine: The TTS engine to use ("gtts" or "polly").

    Returns:
        The list of dialogue blocks, updated with "audio_path" and "duration".
    """
    audio_dir = os.path.join(job_result_dir, AUDIO_BLOCK_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)
    
    polly_client = None
    if tts_engine == "polly":
        try:
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION")
            
            if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
                print("Warning: AWS credentials or region not fully configured in .env for Polly. Falling back to gTTS.")
                tts_engine = "gtts" # Fallback
            else:
                polly_client = boto3.client(
                    'polly',
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key
                )
                # Test credentials with a simple call
                polly_client.describe_voices(LanguageCode='en-US') 
                print("Successfully initialized Amazon Polly client.")
        except (NoCredentialsError, PartialCredentialsError):
            print("Error: AWS credentials not found or incomplete. Check your environment/config. Falling back to gTTS.")
            tts_engine = "gtts"
        except ClientError as e:
            # Handle potential errors like invalid credentials or region
            print(f"Error initializing Polly client: {e}. Falling back to gTTS.")
            tts_engine = "gtts"
        except Exception as e: # Catch other potential issues
            print(f"Unexpected error initializing Polly: {e}. Falling back to gTTS.")
            tts_engine = "gtts"

    updated_blocks = []
    for i, block in enumerate(dialogue_blocks):
        text = block.get("dialogue", "").strip()
        scene_num = block.get("scene_number", "unknown")
        speaker = block.get("speaker", "narrator").lower()
        
        if not text:
            print(f"Warning: Skipping block {i} (Scene {scene_num}) due to empty dialogue.")
            block["audio_path"] = None
            block["duration"] = 0.0
            updated_blocks.append(block)
            continue
        
        # Generate a unique filename
        block_hash = uuid.uuid4().hex[:6]
        filename = f"block_{i}_scn{scene_num}_{speaker}_{block_hash}.mp3"
        output_path = os.path.join(audio_dir, filename)
        
        audio_path = None
        duration = None
        success = False
        
        print(f"  Processing block {i+1}/{len(dialogue_blocks)} (Scene {scene_num}, Engine: {tts_engine})...", end='')

        # --- TTS Generation --- 
        try:
            if tts_engine == "polly" and polly_client:
                # Use Amazon Polly
                response = polly_client.synthesize_speech(
                    Text=text,
                    OutputFormat=POLLY_OUTPUT_FORMAT,
                    VoiceId=POLLY_VOICE_ID,
                    Engine=POLLY_ENGINE
                )
                
                if "AudioStream" in response:
                    with open(output_path, 'wb') as file:
                        file.write(response['AudioStream'].read())
                    audio_path = output_path
                    success = True
                else:
                     print(f"\n   Error: Polly did not return AudioStream for block {i}.")
                    
            else:
                # Use gTTS (Original or fallback)
                if tts_engine != "gtts": # Notify if falling back
                     print(f" (using fallback gTTS) ", end='')
                tts = gTTS(text=text, lang='en')
                tts.save(output_path)
                audio_path = output_path
                success = True

            # --- Get Duration --- 
            if success and audio_path:
                duration = _get_audio_duration(audio_path)
                if duration is None:
                    print(f"\n   Warning: Could not get duration for generated audio: {filename}")
                    # Fallback or mark as error?
                    duration = 0.0 # Assign 0 duration if failed
                    success = False # Treat as failure if duration is crucial
                else:
                     print(f" -> Saved: {filename} ({duration:.2f}s)")
            else:
                # Ensure duration is 0 if TTS failed
                duration = 0.0
                if not success:
                    print(f" -> FAILED")
        
        except Exception as e:
            print(f"\n   Error during TTS generation for block {i}: {e}")
            audio_path = None
            duration = 0.0
            success = False
        
        # Update block info
        block["audio_path"] = audio_path
        block["duration"] = duration
        updated_blocks.append(block)
        
        # Optional: Add a small delay to avoid overwhelming TTS APIs (especially cloud ones)
        # time.sleep(0.1)

    return updated_blocks


# --- Keep old function for potential single use or comment out/remove ---
# def text_to_speech(text: str, output_dir: str, filename: str = "dialogue_audio.mp3") -> str:
#     """Converts text to speech using gTTS and saves it as an MP3 file.
#     ... (rest of old function)
#     """ 