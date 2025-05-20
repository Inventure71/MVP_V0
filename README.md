# esson Video MVP

## Features
- Upload PDFs
- Extracts text from PDFs
- Uses Gemini API to create lessons and dialogues
- Converts dialogue to speech (TTS)
- Generates scenes and creates a video (text frames + audio)

## Setup

1. Install dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```
2. Set up Gemini API credentials in a `.env` file in `backend/`:
    ```env
    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_API_URL=https://api.gemini.com/v1/generate
    ```
3. Run the backend:
    ```bash
    uvicorn main:app --reload
    ```

## Usage
- POST `/upload_pdf/` with a PDF file to upload.
- POST `/process_pdf/` with `{ "filename": "your_uploaded_file.pdf" }` to process and generate the lesson video.

## Notes
- Gemini API calls are placeholders; implement with your actual API details.
- Video frames are text-only for now.
- TTS uses Google gTTS (free). 