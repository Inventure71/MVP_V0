import requests
import os
import json

GEMINI_API_URL = os.getenv('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
MAX_CONTEXT_LENGTH = 4096  # adjust as needed

# Helper to chunk text
def chunk_text(text, max_length=MAX_CONTEXT_LENGTH):
    words = text.split()
    chunks = []
    chunk = []
    length = 0
    for word in words:
        if length + len(word) + 1 > max_length:
            chunks.append(' '.join(chunk))
            chunk = []
            length = 0
        chunk.append(word)
        length += len(word) + 1
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

def call_gemini_api(prompt, text):
    # Gemini Pro expects the prompt and text as a single string in the 'parts' list
    full_prompt = f"{prompt}\n{text}"
    payload = {
        "contents": [
            {"parts": [
                {"text": full_prompt}
            ]}
        ]
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Gemini Pro returns the response in 'candidates' -> 'content' -> 'parts' -> 'text'
        if "candidates" in data and data["candidates"]:
            text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            # Try to parse as JSON
            try:
                return json.loads(text_response)
            except Exception:
                return text_response
        return None
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def process_text_with_gemini(text):
    prompt = (
        "You are an expert educator. Based on the following text, create:\n"
        "1. A concise lesson summary.\n"
        "2. A dialogue between up to two characters that helps explain or explore the lesson.\n"
        "Respond in JSON with the following structure:\n"
        "{\n  \"lesson\": \"<lesson summary>\",\n  \"dialogue\": \"<dialogue text>\"\n}\n"
        "Text:\n\"\"\"\n"
        "{input}\n\"\"\"\n"
    )
    chunks = chunk_text(text)
    lessons = []
    dialogues = []
    for chunk in chunks:
        # Format the prompt with the chunk as input
        formatted_prompt = prompt.replace('{input}', chunk)
        result = call_gemini_api(formatted_prompt, "")
        if result and isinstance(result, dict) and 'lesson' in result and 'dialogue' in result:
            lessons.append(result["lesson"])
            dialogues.append(result["dialogue"])
        else:
            lessons.append("")
            dialogues.append("")
    return '\n'.join(lessons), '\n'.join(dialogues)

def generate_scenes_with_gemini(lesson, dialogue):
    prompt = (
        "You are a creative scriptwriter. Based on the following lesson and dialogue, create a list of scenes for a video. Each scene should have:\n"
        "- a scene number,\n"
        "- the text to display,\n"
        "- and a short description of a figure (if any, otherwise null).\n\n"
        "Respond in JSON as a list of objects, each with keys 'scene', 'text', and 'figure'. Example:\n"
        "[\n  {\"scene\": 0, \"text\": \"Scene text here\", \"figure\": \"Figure description or null\"},\n  ...\n]\n\n"
        "Lesson:\n\"\"\"\n{lesson}\n\"\"\"\n\nDialogue:\n\"\"\"\n{dialogue}\n\"\"\"\n"
    )
    formatted_prompt = prompt.replace('{lesson}', lesson).replace('{dialogue}', dialogue)
    result = call_gemini_api(formatted_prompt, "")
    if result and isinstance(result, list):
        return result
    # Try to parse if result is a string
    if result and isinstance(result, str):
        try:
            scenes = json.loads(result)
            if isinstance(scenes, list):
                return scenes
        except Exception:
            pass
    # Fallback
    return [
        {"scene": 0, "text": "2+2=4", "figure": None},
        {"scene": 1, "text": "A: What is 2+2? B: Four!", "figure": None}
    ] 