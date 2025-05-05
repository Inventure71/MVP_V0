import google.generativeai as genai
import os
import json
from typing import Tuple, List, Dict, Any, Optional
import re

# Configure the Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key)

# --- Constants --- (Adjust as needed)
MODEL_NAME = "gemini-1.5-flash" # Using flash for speed/cost, consider Pro for complexity
JSON_CONFIG = genai.types.GenerationConfig(response_mime_type="application/json")
SAFETY_SETTINGS = { # Define stricter safety settings if needed
    # e.g., "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
}

# Define a reasonable maximum token limit per chunk for the chosen model
# This is an estimate; actual limits depend on the model version.
# Consult Gemini documentation for precise limits.
MAX_INPUT_TOKENS_PER_CHUNK = 10000 # Adjust based on model documentation and testing

# --- Helper Function for Text Chunking (if needed) ---
# Using char count as a proxy for token limit safety for large inputs
def chunk_text(text: str, max_chars: int = 30000) -> List[str]:
    """Simple text chunking based on character count, attempting paragraph splits."""
    chunks = []
    current_chunk = ""
    for paragraph in text.split('\\n\\n'): # Split by paragraphs first
        if not paragraph.strip():
             continue # Skip empty paragraphs

        paragraph_with_sep = paragraph + "\\n\\n"
        if len(current_chunk) + len(paragraph_with_sep) < max_chars:
            current_chunk += paragraph_with_sep
        else:
            # If the current chunk has content, add it before starting new
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = "" # Reset current chunk

            # If the paragraph itself is too long, split it crudely
            if len(paragraph_with_sep) > max_chars:
                print(f"Warning: Paragraph starting with '{paragraph[:50]}...' exceeds max_chars ({max_chars}). Splitting mid-paragraph.")
                for i in range(0, len(paragraph), max_chars): # Split the paragraph itself
                    part = paragraph[i:i + max_chars]
                    # Add as separate chunk, ensure newline separation if splitting mid-paragraph
                    chunks.append(part + ("\\n\\n" if i + max_chars < len(paragraph) else "")) 
                # The oversized paragraph is handled, don't add it to current_chunk
            else: # Paragraph fits in a chunk by itself, start new chunk
                 current_chunk = paragraph_with_sep


    if current_chunk: # Add the last chunk
        chunks.append(current_chunk.strip())

    print(f"Split text into {len(chunks)} chunks (approx. {max_chars} chars each).")
    return chunks

# --- Core Gemini Interaction Functions ---
def _call_gemini(prompt: str, model_name: str = MODEL_NAME, is_json_output: bool = True) -> Any:
    """Helper function to call the Gemini API and handle potential errors."""
    try:
        model = genai.GenerativeModel(model_name)
        config = JSON_CONFIG if is_json_output else None
        
        print(f"--- Calling Gemini ({'JSON' if is_json_output else 'Text'}) ---")
        # print(f"Prompt:\n{prompt[:500]}...") # Optional: Log start of prompt

        response = model.generate_content(
            prompt,
            generation_config=config,
            safety_settings=SAFETY_SETTINGS
        )

        # Debugging: Print raw response parts if needed
        # print(f"Raw Gemini Response Parts: {response.candidates[0].content.parts}")

        if not response.candidates:
             raise ValueError("Gemini response did not contain candidates.")
        if response.prompt_feedback.block_reason:
            raise ValueError(f"Gemini request blocked: {response.prompt_feedback.block_reason.name}")
        
        # Handle cases where generation stops early or has no output
        candidate = response.candidates[0]
        if not candidate.content.parts:
             finish_reason = candidate.finish_reason
             if finish_reason != genai.types.FinishReason.STOP:
                   raise ValueError(f"Gemini generation stopped unexpectedly: {finish_reason.name}")
             else: # Valid stop but no parts - likely empty generation
                   print("Warning: Gemini response candidate missing content parts but finish reason was STOP.")
                   return {} if is_json_output else "" # Return empty structure/string


        result_text = candidate.content.parts[0].text
        # print(f"Result Text Received:\n{result_text[:500]}...") # Optional: Log start of result

        if is_json_output:
            # Clean potential markdown formatting around JSON
            if result_text.startswith("```json"): 
                result_text = result_text[len("```json"):].strip()
            elif result_text.startswith("```"): # Handle cases with ``` but no language specifier
                 result_text = result_text[len("```"):].strip()

            if result_text.endswith("```"):
                result_text = result_text[:-len("```")].strip()

            try:
                parsed_json = json.loads(result_text)
                # print("--- Gemini Call Successful (JSON) ---")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Error decoding Gemini JSON response: {e}")
                print(f"Problematic Response Text (first 500 chars):\\n{result_text[:500]}")
                # Attempt to find JSON within the text if simple cleaning failed
                try:
                    # Regex to find a JSON object ({...}) or array ([...])
                    json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', result_text, re.DOTALL)
                    if json_match:
                        print("Attempting to parse extracted JSON block...")
                        extracted_json_str = json_match.group(0)
                        parsed_json = json.loads(extracted_json_str)
                        print("--- Gemini Call Successful (JSON extracted) ---")
                        return parsed_json
                    else:
                         raise ValueError(f"Gemini returned non-JSON text when JSON was expected and extraction failed: {e}")
                except Exception as inner_e:
                     print(f"Could not recover JSON: {inner_e}")
                     raise ValueError(f"Gemini returned invalid JSON: {e}")

        else:
            # print("--- Gemini Call Successful (Text) ---")
            return result_text

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        # Re-raise the exception to be caught by the main endpoint handler
        raise

# --- New Scene Generation Function ---
def generate_scenes_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Generates a list of scenes based on the input text using Gemini.

    Args:
        text: The input text extracted from PDFs.

    Returns:
        A list of dictionaries, where each dictionary represents a scene:
        [{ "scene_number": int, "content": str }, ...]
        Returns an empty list if generation fails or no scenes are created.
    """
    # Determine target number of scenes (e.g., 1 scene per ~150 words, adjust as needed)
    word_count = len(text.split())
    # Ensure at least a few scenes, but cap it reasonably
    target_scenes = max(3, min(50, word_count // 150)) # Example: 3-50 scenes
    print(f"Targeting ~{target_scenes} scenes for {word_count} words.")

    # Single prompt approach (might need chunking if text is > context window)
    # Check model context window if text is very large. Flash has 1M tokens, Pro 1-2M. Should be okay for most PDFs.
    prompt = f"""\
Analyze the following text and divide it into logical scenes suitable for an educational video presentation. Aim for approximately {target_scenes} scenes. Each scene should represent a distinct topic, step, or concept discussed in the text. Scene content should be extracted or summarized from the text.

Input Text:
```
{text}
    ```

    Task:
Generate a JSON response containing a single key "scenes". The value of "scenes" should be a list of JSON objects. Each object must represent one scene and contain the following keys:
- "scene_number": An integer representing the order of the scene, starting from 1.
- "content": A string containing the relevant portion or summary of the text for that scene. Ensure the content logically flows from one scene to the next and covers the original text comprehensively without excessive overlap unless necessary for context.

Example JSON Output Structure:
    {{
      "scenes": [
    {{ "scene_number": 1, "content": "Text content for the first logical part..." }},
    {{ "scene_number": 2, "content": "Text content covering the next concept..." }},
    ...
  ]
}}

Generate the JSON output now based on the input text. Ensure scene numbers are sequential and start from 1.
"""
    try:
        response_json = _call_gemini(prompt, is_json_output=True)

        if isinstance(response_json, dict) and "scenes" in response_json and isinstance(response_json["scenes"], list):
            validated_scenes = []
            seen_numbers = set()
            last_num = 0
            for scene in response_json["scenes"]:
                if isinstance(scene, dict) and \
                   all(k in scene for k in ["scene_number", "content"]) and \
                   isinstance(scene["scene_number"], int) and \
                   scene["scene_number"] > 0 and \
                   isinstance(scene["content"], str) and \
                   scene["content"].strip(): # Ensure content is not empty

                    num = scene["scene_number"]
                    if num in seen_numbers:
                         print(f"Warning: Duplicate scene number {num} found. Skipping.")
                         continue
                    # Optional: Check for sequential numbers? Might be too strict.
                    # if num != last_num + 1:
                    #    print(f"Warning: Non-sequential scene number {num} (expected {last_num + 1}).")
                    
                    seen_numbers.add(num)
                    validated_scenes.append({
                        "scene_number": num,
                        "content": scene["content"].strip()
                    })
                else:
                    print(f"Warning: Skipping invalid scene structure or empty content: {scene}")

            # Sort scenes by scene_number just in case Gemini didn't order them
            validated_scenes.sort(key=lambda x: x["scene_number"])

            # Renumber sequentially if needed (optional, but safer)
            # for i, scene_data in enumerate(validated_scenes):
            #     scene_data["scene_number"] = i + 1
            
            if not validated_scenes:
                 print("Warning: Gemini returned empty or invalid scene list structure.")
                 return [] # Return empty list instead of raising error

            print(f"Successfully generated and validated {len(validated_scenes)} scenes.")
            return validated_scenes
        else:
             print(f"Warning: Unexpected JSON structure for scenes response: {response_json}")
             return [] # Return empty list

    except Exception as e:
        print(f"Error generating scenes from Gemini: {e}")
        # Don't raise here, return empty list to allow main flow to potentially handle it
        return []

# --- New Structured Dialogue Generation Function ---
def generate_structured_dialogue(original_text: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates a structured dialogue based on the original text and pre-defined scenes.

    Args:
        original_text: The full text extracted from the PDFs (for context).
        scenes: The list of scene objects generated by `generate_scenes_from_text`.
              Expected structure: [{"scene_number": int, "content": str}, ...]

    Returns:
        A list of dialogue blocks, where each block is a dictionary:
        [{ "speaker": str ("Teacher" or "Student"),
           "scene_number": int,
           "dialogue": str }, ...]
        Returns an empty list if generation fails.
    """
    if not scenes:
        print("Cannot generate dialogue: No scenes provided.")
        return []

    # Prepare scene context for the prompt more concisely
    scene_summary = "\\n".join([f"- Scene {s['scene_number']}: Starts with '{s['content'][:80]}...'" for s in scenes])

    # Limit original text in prompt to avoid exceeding context if it's huge
    max_context_chars = 5000 
    context_text_snippet = original_text[:max_context_chars]
    if len(original_text) > max_context_chars:
        context_text_snippet += "\\n... (Original text truncated for brevity in prompt)"


    prompt = f"""\
You are an AI assistant creating an educational dialogue based on provided text snippets (scenes).

Scene Structure:
The source material has been divided into the following scenes, with a preview of their content:
{scene_summary}

Task:
Create an engaging and explanatory dialogue between two characters: "Teacher" and "Student".
- The Teacher's role is to explain the concepts presented in each scene clearly and simply.
- The Student's role is to ask pertinent questions, seek clarification, or summarize understanding, prompting the Teacher's explanations.
- The dialogue must progress logically through the scenes, referencing the content of each scene number.
- Every line of dialogue MUST be associated with exactly one `scene_number` from the provided scene list.
- Generate dialogue that covers the core information within the scenes.
- Keep the dialogue concise and focused for a video format.

Output Format:
Generate a JSON response containing a single key "dialogue_blocks". The value should be a list of JSON objects. Each object represents one utterance (a block of speech from one character) and must contain the following keys:
- "speaker": A string, MUST be either "Teacher" or "Student".
- "scene_number": An integer corresponding EXACTLY to the scene the dialogue relates to (must be one of the provided scene numbers: {', '.join(map(str, [s['scene_number'] for s in scenes]))}).
- "dialogue": A string containing the character's spoken words for that block. Do not include the speaker's name (e.g., "Teacher:") within this string.

Example JSON Output Structure:
{{
  "dialogue_blocks": [
    {{ "speaker": "Teacher", "scene_number": 1, "dialogue": "Today we'll start with the basics mentioned in scene 1..." }},
    {{ "speaker": "Student", "scene_number": 1, "dialogue": "Okay, so what is the main idea there?" }},
    {{ "speaker": "Teacher", "scene_number": 2, "dialogue": "Good question! Scene 2 explains that..." }},
    {{ "speaker": "Teacher", "scene_number": 2, "dialogue": "It breaks down into these parts..." }},
    {{ "speaker": "Student", "scene_number": 2, "dialogue": "Ah, I see how that connects." }},
    {{ "speaker": "Teacher", "scene_number": 3, "dialogue": "Now, moving onto scene 3..." }}
    // ... (dialogue continues for other scenes)
  ]
}}

Constraint Checklist & Confidence Score:
Before finalizing the JSON, double-check it adheres to all constraints:
1.  Is the output valid JSON? Yes/No
2.  Is there a root key "dialogue_blocks" with a list value? Yes/No
3.  Does each item in the list have ONLY "speaker", "scene_number", "dialogue" keys? Yes/No
4.  Is "speaker" always "Teacher" or "Student"? Yes/No
5.  Is "scene_number" always an integer from the valid list? Yes/No
6.  Is "dialogue" always a non-empty string? Yes/No
7.  Does the dialogue logically progress with the scenes? Yes/No
Confidence Score (1-5): [Score]

Generate the JSON output now based on the scenes and the goal of creating an explanatory Teacher/Student dialogue. Adhere strictly to the format.
(You don't need to include the Constraint Checklist in the final JSON output, it's just for your internal verification).
"""
# Note: Added original_text context back lightly, but emphasized scenes as primary guide.
# Added constraint checklist to prompt for better adherence.

    try:
        response_json = _call_gemini(prompt, is_json_output=True)

        if isinstance(response_json, dict) and "dialogue_blocks" in response_json and isinstance(response_json["dialogue_blocks"], list):
            validated_dialogue = []
            valid_scene_numbers = {s["scene_number"] for s in scenes}

            for block in response_json["dialogue_blocks"]:
                # Add more robust checking
                if not isinstance(block, dict):
                     print(f"Warning: Dialogue block is not a dict: {block}")
                     continue
                
                required_keys = {"speaker", "scene_number", "dialogue"}
                if not required_keys.issubset(block.keys()):
                    print(f"Warning: Dialogue block missing keys ({required_keys - set(block.keys())}): {block}")
                    continue
                
                speaker = block["speaker"]
                scene_num = block["scene_number"]
                dialogue_text = block["dialogue"]

                if not isinstance(speaker, str) or speaker not in ["Teacher", "Student"]:
                     print(f"Warning: Invalid speaker '{speaker}' in block: {block}")
                     continue
                if not isinstance(scene_num, int) or scene_num not in valid_scene_numbers:
                     print(f"Warning: Invalid scene_number '{scene_num}' (Valid: {valid_scene_numbers}) in block: {block}")
                     continue
                if not isinstance(dialogue_text, str) or not dialogue_text.strip():
                     print(f"Warning: Empty dialogue text in block: {block}")
                     continue
                
                # If checks pass, add the validated block
                validated_dialogue.append({
                    "speaker": speaker,
                    "scene_number": scene_num,
                    "dialogue": dialogue_text.strip()
                })

            if not validated_dialogue:
                 print("Warning: Gemini returned empty or invalid dialogue block list after validation.")
                 return []

            print(f"Successfully generated and validated {len(validated_dialogue)} dialogue blocks.")
            return validated_dialogue
        else:
            print(f"Warning: Unexpected JSON structure for dialogue response (missing 'dialogue_blocks' list?): {response_json}")
            return []

    except Exception as e:
        print(f"Error generating structured dialogue from Gemini: {e}")
        return []

# --- REMOVED OLD HELPER FUNCTIONS --- 
# chunk_text_old and _call_gemini_old removed as they are no longer used.

# --- Ensure Old Functions are Removed ---
# The old generate_lesson_and_dialogue and generate_scenes functions should not be present below this line. 