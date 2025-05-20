import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uuid
from typing import Optional
from dotenv import load_dotenv

# Import helper modules
from . import pdf_processor
from . import gemini_processor
from . import tts_processor
from . import video_processor # Ensure this imports the new function correctly

# Import the specific function name we'll be calling
from .pdf_processor import process_uploaded_pdfs

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories (ensure they exist)
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")

@app.get("/")
async def read_root():
    """Serve the frontend HTML page"""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/process-pdfs/")
async def process_pdfs_endpoint(
    background_tasks: BackgroundTasks, 
    files: list[UploadFile] = File(...),
    custom_instructions: Optional[str] = Form(None),
    tts_engine: str = Form("gtts")
):
    """
    Endpoint to upload PDFs, process them using the new scene/dialogue flow, 
    and generate a video lesson.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    job_id = str(uuid.uuid4())
    job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
    job_result_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_result_dir, exist_ok=True) # This directory will now contain annotated_images subdir

    pdf_paths = []
    combined_text = "" # Initialize
    all_images_info = [] # Initialize list for image info
    scenes = []
    dialogue_blocks = []
    processed_dialogue_blocks = []
    video_path = None

    try:
        # --- 0. Upload Files --- 
        print(f"Job {job_id}: Handling PDF uploads...")
        for file in files:
            if file.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only PDFs are allowed.")

            file_path = os.path.join(job_upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            pdf_paths.append(file_path)
        print(f"Job {job_id}: Uploaded {len(pdf_paths)} PDF(s).")

        # --- 1. Extract Text & Images ---
        print(f"Job {job_id}: Extracting text and images from PDFs...")
        # Call the new processor function
        combined_text, all_images_info = process_uploaded_pdfs(pdf_paths, job_id)
        print(f"Job {job_id}: Extracted {len(combined_text)} characters and found {len(all_images_info)} images.")
        if not combined_text and not all_images_info: # Check if anything was extracted
             raise HTTPException(status_code=400, detail="Could not extract text or images from the provided PDF(s).")
        # Note: We might proceed even if only images OR text were found, depending on desired behavior.

        # --- 2. Generate Scenes (Gemini) ---
        try:
            print(f"Job {job_id}: Generating scenes from text with Gemini...")
            # Pass both text, image info, and custom instructions to Gemini processor
            scenes = gemini_processor.generate_scenes_from_text(
                combined_text, 
                all_images_info,
                custom_instructions
            )
            if not scenes:
                 raise HTTPException(status_code=500, detail="Gemini failed to generate scene content.")
            print(f"Job {job_id}: Successfully generated {len(scenes)} scenes.")
        except Exception as e:
            print(f"Job {job_id}: Gemini call failed during scene generation: {e}")
            raise HTTPException(status_code=500, detail=f"Error communicating with AI model for scene generation: {str(e)}")

        # --- 3. Generate Dialogue Blocks (Gemini) ---
        try:
            print(f"Job {job_id}: Generating structured dialogue with Gemini...")
            # Pass original text context, generated scenes, image info, and custom instructions
            dialogue_blocks = gemini_processor.generate_structured_dialogue(
                combined_text, 
                scenes, 
                all_images_info,
                custom_instructions
            )
            if not dialogue_blocks:
                 raise HTTPException(status_code=500, detail="Gemini failed to generate dialogue content.")
            print(f"Job {job_id}: Successfully generated {len(dialogue_blocks)} dialogue blocks.")
        except Exception as e:
            print(f"Job {job_id}: Gemini call failed during dialogue generation: {e}")
            raise HTTPException(status_code=500, detail=f"Error communicating with AI model for dialogue generation: {str(e)}")

        # --- 4. Generate Audio Blocks (TTS) ---
        try:
            print(f"Job {job_id}: Generating audio for {len(dialogue_blocks)} dialogue blocks...")
            # The dialogue blocks might *contain* image references from Gemini now
            processed_dialogue_blocks = tts_processor.generate_audio_blocks(
                dialogue_blocks, 
                job_result_dir,
                tts_engine=tts_engine
            )
            # Check if ANY audio was successfully generated
            successful_audio_count = sum(1 for b in processed_dialogue_blocks if b.get('audio_path') and b.get('duration') is not None)
            if successful_audio_count == 0 and len(processed_dialogue_blocks) > 0:
                 # No audio succeeded at all, even though blocks existed. Treat as failure?
                 print(f"Job {job_id}: Warning: TTS failed to generate any audio files.")
                 # Option: Allow silent video? For now, raise error.
                 raise HTTPException(status_code=500, detail="Failed to generate any audio for the dialogue blocks.")
            elif successful_audio_count < len(processed_dialogue_blocks):
                 print(f"Job {job_id}: Warning: TTS failed for {len(processed_dialogue_blocks) - successful_audio_count} dialogue blocks. Proceeding with available audio.")
            else:
                 print(f"Job {job_id}: Successfully generated audio for all dialogue blocks.")
        except Exception as e:
            print(f"Job {job_id}: TTS generation failed critically: {e}")
            raise HTTPException(status_code=500, detail=f"Failed during audio generation process: {str(e)}")

        # --- 5. Generate Video --- 
        try:
            print(f"Job {job_id}: Creating video using {len(scenes)} scenes and {len(processed_dialogue_blocks)} processed dialogue blocks...")
            # Pass the original scenes and the dialogue blocks that now include audio info
            video_path = video_processor.create_video_from_blocks(
                scenes=scenes, 
                processed_dialogue_blocks=processed_dialogue_blocks, 
                output_dir=job_result_dir
                # filename=f"{job_id}_lesson.mp4" # Can specify filename here if needed
            )
            if not video_path:
                 raise HTTPException(status_code=500, detail="Video processor failed to generate the video file.")
            print(f"Job {job_id}: Video created successfully: {video_path}")
        except Exception as e:
             print(f"Job {job_id}: Video creation failed: {e}")
             # Add specific moviepy/ffmpeg hints if possible?
             raise HTTPException(status_code=500, detail=f"Failed to create video: {str(e)}. Ensure ffmpeg and necessary fonts are installed and accessible.")

        # --- 6. Clean up & Return --- 
        # Clean up uploaded files *after* successful processing
        print(f"Job {job_id}: Scheduling cleanup of upload directory: {job_upload_dir}")
        background_tasks.add_task(shutil.rmtree, job_upload_dir, ignore_errors=True)
        
        # Optionally clean up intermediate audio blocks if desired (keep for debugging?)
        # audio_block_dir = os.path.join(job_result_dir, tts_processor.AUDIO_BLOCK_SUBDIR)
        # print(f"Job {job_id}: Scheduling cleanup of audio blocks directory: {audio_block_dir}")
        # background_tasks.add_task(shutil.rmtree, audio_block_dir, ignore_errors=True)

        # Return the final video file
        final_filename = os.path.basename(video_path)
        print(f"Job {job_id}: Returning video file: {final_filename}")
        return FileResponse(path=video_path, media_type='video/mp4', filename=final_filename)

    except HTTPException as e:
        # Clean up directories if specific HTTP error occurs
        print(f"Job {job_id}: HTTP Exception caught: {e.detail}")
        background_tasks.add_task(shutil.rmtree, job_upload_dir, ignore_errors=True)
        background_tasks.add_task(shutil.rmtree, job_result_dir, ignore_errors=True)
        raise e # Re-raise the HTTPException
    except Exception as e:
        # Clean up directories on any unexpected error
        print(f"Job {job_id}: Unexpected error during processing: {e}")
        import traceback
        traceback.print_exc()
        background_tasks.add_task(shutil.rmtree, job_upload_dir, ignore_errors=True)
        background_tasks.add_task(shutil.rmtree, job_result_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        # Ensure uploaded file handles are closed (FastAPI might do this automatically, but belt-and-suspenders)
        for file in files:
             if hasattr(file, 'file') and file.file and hasattr(file.file, 'close') and not file.file.closed:
                 try:
                      file.file.close()
                 except Exception as e_close:
                      print(f"Job {job_id}: Error closing uploaded file handle for {file.filename}: {e_close}")

# Add logic to run the app with uvicorn if the script is executed directly
if __name__ == "__main__":
    import uvicorn
    # Run from module name `backend.main` if running `python -m backend.main`
    # Or adjust based on how you run the application.
    # The string "main:app" assumes you run `uvicorn main:app` from within the `backend` directory.
    # If running from the root (`MVP_V0`), it should likely be "backend.main:app"
    
    # Detect if running via `python -m backend.main` vs `python backend/main.py`
    if __package__ == "backend":
         uvicorn_app_str = "backend.main:app" # Running as module
    else:
         uvicorn_app_str = "main:app" # Running as script

    print(f"Starting Uvicorn with app string: '{uvicorn_app_str}'")
    uvicorn.run(uvicorn_app_str, host="0.0.0.0", port=8000, reload=True, workers=1) 