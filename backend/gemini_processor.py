import google.generativeai as genai
import os
import json
from typing import Tuple, List, Dict, Any, Optional
import re
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set!")

genai.configure(api_key=API_KEY)

# Constants
MODEL_NAME = "gemini-1.5-pro" # Higher capability model
MAX_CHARS_PER_CHUNK = 25000    # Avoid exceeding token limits
MAX_IMAGES_PER_CHUNK = 20      # Limit images per API call to avoid limits

# --- Helper Function for Text Chunking (if needed) ---
# Using char count as a proxy for token limit safety for large inputs
def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """Simple text chunking based on character count, attempting paragraph splits."""
    chunks = []
    current_chunk = ""
    for paragraph in text.split('\n\n'): # Split by paragraphs first
        if not paragraph.strip():
             continue # Skip empty paragraphs

        paragraph_with_sep = paragraph + "\n\n"
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
                    chunks.append(part + ("\n\n" if i + max_chars < len(paragraph) else ""))
                # The oversized paragraph is handled, don't add it to current_chunk
            else: # Paragraph fits in a chunk by itself, start new chunk
                 current_chunk = paragraph_with_sep


    if current_chunk: # Add the last chunk
        chunks.append(current_chunk.strip())

    print(f"Split text into {len(chunks)} chunks (approx. {max_chars} chars each).")
    return chunks

# --- Core Gemini Interaction Functions ---
def _call_gemini_text_only(prompt: str, model_name: str = MODEL_NAME, is_json_output: bool = True) -> Any:
    """Helper function to call the Gemini API with text-only input and handle potential errors."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, 'text'):
            print("Warning: Gemini returned empty response or unexpected format.")
            return {} if is_json_output else ""
            
        result_text = response.text.strip()
        
        # Handle JSON output parsing
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
                print(f"Problematic Response Text (first 500 chars):\n{result_text[:500]}")
                # Attempt to find JSON within the text if simple cleaning failed
                try:
                    # Regex to find a JSON object ({...}) or array ([...])
                    json_match = re.search(r'(\{.*?\}|\[.*?\])', result_text, re.DOTALL)
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
        return {} if is_json_output else ""

def _call_gemini_with_images(prompt: str, image_paths: List[str], model_name: str = MODEL_NAME, is_json_output: bool = True) -> Any:
    """Helper function to call the Gemini API with multimodal content (text + images)."""
    try:
        model = genai.GenerativeModel(model_name)
        
        # Prepare multimodal content parts
        content_parts = [{"text": prompt}]
        
        # Add images to the content
        for img_path in image_paths:
            try:
                # Open and load the image
                img = Image.open(img_path)
                
                # Convert image to a content part
                content_parts.append({"inline_data": {
                    "mime_type": f"image/{img.format.lower() if img.format else 'png'}", 
                    "data": _encode_image_to_base64(img_path)
                }})
            except Exception as img_err:
                print(f"Error processing image {img_path}: {img_err}")
                # Continue with other images if one fails
        
        # Generate response from multimodal content
        response = model.generate_content(content_parts)
        
        if not response or not hasattr(response, 'text'):
            print("Warning: Gemini returned empty response or unexpected format.")
            return {} if is_json_output else ""
            
        result_text = response.text.strip()
        
        # Handle JSON output parsing
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
                print("--- Gemini Call Successful (JSON with images) ---")
                return parsed_json
            except json.JSONDecodeError as e:
                print(f"Error decoding Gemini JSON response: {e}")
                print(f"Problematic Response Text (first 500 chars):\n{result_text[:500]}")
                # Attempt to find JSON within the text
                try:
                    json_match = re.search(r'(\{.*?\}|\[.*?\])', result_text, re.DOTALL)
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
            print("--- Gemini Call Successful (Text with images) ---")
            return result_text
            
    except Exception as e:
        print(f"Error calling Gemini API with images: {e}")
        return {} if is_json_output else ""

def _encode_image_to_base64(image_path):
    """Convert an image to base64 encoding for API submission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _split_content_by_images(text: str, images_info: List[Dict[str, str]], max_images_per_chunk: int = MAX_IMAGES_PER_CHUNK) -> List[Dict[str, Any]]:
    """
    Split content into chunks based on image limits per API call.

    Args:
        text: The input text to split.
        images_info: List of dictionaries with image info.
        max_images_per_chunk: Maximum number of images to include in a single API call.

    Returns:
        List of dictionaries, each with "text" and "images" keys.
    """
    # Extract all image labels and their positions in text
    image_positions = []
    for img in images_info:
        if 'label' in img:
            label = img['label']
            # Find all occurrences of this label in the text
            for match in re.finditer(re.escape(label), text):
                start_pos = match.start()
                image_positions.append({
                    'position': start_pos,
                    'label': label,
                    'info': img
                })
    
    # Sort images by their position in text
    image_positions.sort(key=lambda x: x['position'])
    
    # Split content if we have too many images
    if not image_positions or len(image_positions) <= max_images_per_chunk:
        # If no images or few enough, just return a single chunk with all content
        return [{
            'text': text,
            'images': images_info
        }]
    
    # Otherwise, create multiple chunks
    chunks = []
    current_images = []
    last_split = 0
    
    for i, img_pos in enumerate(image_positions):
        current_images.append(img_pos['info'])
        
        # Check if we need to start a new chunk
        if len(current_images) >= max_images_per_chunk or i == len(image_positions) - 1:
            # Find a good split point (paragraph break)
            # For the final image, use the rest of the text
            if i == len(image_positions) - 1:
                split_point = len(text)
            else:
                # Try to find a paragraph break after the current image
                next_pos = image_positions[i + 1]['position']
                paragraph_break = text.find('\n\n', img_pos['position'], next_pos)
                
                if paragraph_break != -1:
                    split_point = paragraph_break + 2  # Include the newlines
                else:
                    # If no paragraph break, try a sentence end
                    sentence_end = max(
                        text.rfind('. ', img_pos['position'], next_pos),
                        text.rfind('! ', img_pos['position'], next_pos),
                        text.rfind('? ', img_pos['position'], next_pos)
                    )
                    
                    if sentence_end != -1:
                        split_point = sentence_end + 2  # Include the period and space
                    else:
                        # If all else fails, split at the midpoint
                        split_point = (img_pos['position'] + next_pos) // 2
            
            # Create a chunk
            chunk_text = text[last_split:split_point].strip()
            chunks.append({
                'text': chunk_text,
                'images': current_images
            })
            
            # Reset for next chunk
            last_split = split_point
            current_images = []
    
    # Handle any remainder text if it exists
    if last_split < len(text):
        chunks.append({
            'text': text[last_split:].strip(),
            'images': []
        })
    
    print(f"Split content into {len(chunks)} chunks based on image distribution (max {max_images_per_chunk} images per chunk).")
    return chunks

# --- Scene Generation Function (Updated) ---
def generate_scenes_from_text(text: str, images_info: List[Dict[str, str]], custom_instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generates a list of scenes based on the input text using Gemini, 
    instructing it to consider images directly.

    Args:
        text: The input text extracted from PDFs, potentially containing [IMG_N] placeholders.
        images_info: List of dictionaries with info about extracted images.
        custom_instructions: Optional custom instructions for the Gemini model.

    Returns:
        A list of dictionaries, where each dictionary represents a scene:
        [{ "scene_number": int, "content": str, "images_in_scene": List[str] }, ...]
        Returns an empty list if generation fails or no scenes are created.
    """
    word_count = len(text.split())
    target_scenes = max(3, min(50, word_count // 150)) # Example: 3-50 scenes
    image_labels = [img['label'] for img in images_info if 'label' in img]
    has_images = bool(image_labels)

    print(f"Targeting ~{target_scenes} scenes for {word_count} words. Found {len(image_labels)} image labels.")

    # Split content into chunks if there are many images
    content_chunks = _split_content_by_images(text, images_info) if has_images else [{"text": text, "images": []}]
    
    # Prepare custom instructions if provided
    custom_instructions_text = ""
    if custom_instructions:
        custom_instructions_text = f"""
CUSTOM INSTRUCTIONS FROM USER:
{custom_instructions}

Please follow these custom instructions while generating the content.
"""

    # Track all scenes across chunks
    all_scenes = []
    total_scenes = 0
    
    for chunk_index, chunk in enumerate(content_chunks):
        chunk_text = chunk["text"]
        chunk_images = chunk["images"]
        chunk_image_paths = [img["path"] for img in chunk_images]
        chunk_image_labels = [img["label"] for img in chunk_images]
        
        # Calculate scenes for this chunk
        chunk_word_count = len(chunk_text.split())
        chunk_target_scenes = max(1, min(50, chunk_word_count // 150))
        if chunk_index == len(content_chunks) - 1:
            # For last chunk, ensure we reach the total target scenes
            chunk_target_scenes = max(1, target_scenes - total_scenes)
            
        print(f"Processing chunk {chunk_index+1}/{len(content_chunks)} with {len(chunk_images)} images. Target scenes: {chunk_target_scenes}")
            
        # Build prompt for this chunk
        image_instructions = ""
        if chunk_image_labels:
            image_instructions = f"""\
IMPORTANT: I am providing you with {len(chunk_image_labels)} images extracted from the source document. Each image appears at a specific point in the text, marked with placeholders like [IMG_1], [IMG_2], etc.

When summarizing content for a scene, look at these images and consider them as part of your understanding. Your JSON output for each scene MUST include an "images_in_scene" key. This key's value should be a list containing the string labels (e.g., "[IMG_5]") of the images most relevant to that specific scene's content. Include a maximum of 2 image labels per scene. If no images are relevant to a scene, provide an empty list: "images_in_scene": [].
""" 
        else:
            image_instructions = """\
IMPORTANT: The input text does not appear to contain image references. For each scene, include the key "images_in_scene" with an empty list as its value: "images_in_scene": [].
"""

        prompt = f"""\
Analyze the following text and divide it into logical scenes suitable for an educational video presentation. Aim for approximately {chunk_target_scenes} scenes. Each scene should represent a distinct topic, step, or concept discussed in the text. Scene content should be extracted or summarized from the text.

{image_instructions}

{custom_instructions_text}

Input Text:
```
{chunk_text}
    ```

    Task:
Generate a JSON response containing a single key "scenes". The value of "scenes" should be a list of JSON objects. Each object must represent one scene and contain the following keys:
- "scene_number": An integer representing the order of the scene, starting from {total_scenes + 1}.
- "content": A string containing the relevant portion or summary of the text for that scene. Ensure the content logically flows and covers the original text comprehensively.
- "images_in_scene": A list of strings containing the labels (e.g., "[IMG_5]") of up to 2 images most relevant to this scene's content. Use an empty list [] if no images are relevant.

Example JSON Output Structure (if images exist):
{{
    "scenes": [
    {{ 
      "scene_number": {total_scenes + 1}, 
      "content": "Text content for the first logical part... mentions [IMG_1].",
      "images_in_scene": ["[IMG_1]"] 
    }},
    {{ 
      "scene_number": {total_scenes + 2}, 
      "content": "Text content covering the next concept... related to [IMG_2] and maybe [IMG_3].",
      "images_in_scene": ["[IMG_2]", "[IMG_3]"] 
    }},
    {{ 
      "scene_number": {total_scenes + 3}, 
      "content": "Text content with no relevant images.",
      "images_in_scene": [] 
    }},
    ...
  ]
}}

Generate the JSON output now based on the input text. Ensure scene numbers are sequential and start from {total_scenes + 1}. Ensure the "images_in_scene" key is present in every scene object.
"""
        try:
            # Call Gemini with or without images based on what's available in this chunk
            if chunk_image_paths:
                print(f"--- Calling Gemini (JSON) with {len(chunk_image_paths)} images ---")
                response_json = _call_gemini_with_images(prompt, chunk_image_paths, is_json_output=True)
            else:
                print(f"--- Calling Gemini (JSON) ---")
                response_json = _call_gemini_text_only(prompt, is_json_output=True)

            if isinstance(response_json, dict) and "scenes" in response_json and isinstance(response_json["scenes"], list):
                validated_scenes = []
                seen_numbers = set()
                for scene in response_json["scenes"]:
                    # Validate base structure + new images_in_scene field
                    if isinstance(scene, dict) and \
                       all(k in scene for k in ["scene_number", "content", "images_in_scene"]) and \
                       isinstance(scene["scene_number"], int) and scene["scene_number"] > 0 and \
                       isinstance(scene["content"], str) and scene["content"].strip() and \
                       isinstance(scene["images_in_scene"], list):

                        num = scene["scene_number"]
                        if num in seen_numbers:
                            print(f"Warning: Duplicate scene number {num} found. Renumbering.")
                            num = total_scenes + len(validated_scenes) + 1

                        # Validate image labels format (optional but good)
                        valid_image_labels = []
                        for label in scene["images_in_scene"]:
                            if isinstance(label, str) and re.match(r'^\[IMG_\d+\]$', label):
                                valid_image_labels.append(label)
                            else:
                                print(f"Warning: Invalid image label format '{label}' in scene {num}. Skipping label.")
                        
                        # Limit to max 2 labels
                        if len(valid_image_labels) > 2:
                            print(f"Warning: Scene {num} had > 2 image labels. Truncating to first 2.")
                            valid_image_labels = valid_image_labels[:2]
                    
                        seen_numbers.add(num)
                        validated_scenes.append({
                            "scene_number": total_scenes + len(validated_scenes) + 1,  # Ensure sequential numbering
                            "content": scene["content"].strip(),
                            "images_in_scene": valid_image_labels # Store validated & truncated list
                        })
                    else:
                        print(f"Warning: Skipping invalid scene structure: {scene}")

                # Update the scenes and counter
                all_scenes.extend(validated_scenes)
                total_scenes += len(validated_scenes)
                print(f"Chunk {chunk_index+1}: Generated {len(validated_scenes)} scenes. Total: {total_scenes}")
            else:
                print(f"Warning: Unexpected JSON structure for scenes response in chunk {chunk_index+1}: {response_json}")

        except Exception as e:
            print(f"Error generating scenes from Gemini for chunk {chunk_index+1}: {e}")
    
    # Final validation
    if not all_scenes:
        print("Warning: No valid scenes were generated from any chunk.")
        return []

    # Ensure scenes are properly sorted and numbered
    all_scenes.sort(key=lambda x: x["scene_number"])
    for i, scene in enumerate(all_scenes):
        scene["scene_number"] = i + 1
        
    print(f"Successfully generated and validated {len(all_scenes)} scenes across {len(content_chunks)} chunks.")
    return all_scenes

# --- Structured Dialogue Generation Function (Updated) ---
def generate_structured_dialogue(original_text: str, scenes: List[Dict[str, Any]], images_info: List[Dict[str, str]], custom_instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generates a structured dialogue based on scenes, instructing the model 
    to incorporate images directly.

    Args:
        original_text: The full text extracted from PDFs (for broader context).
        scenes: The list of scene objects (including scene_number, content, images_in_scene).
        images_info: List of dictionaries with info about extracted images.
        custom_instructions: Optional custom instructions for the Gemini model.

    Returns:
        A list of dialogue block dictionaries:
        [{
           "scene_number": int,
            "speaker": str (e.g., "Narrator", "Character A"), 
            "dialogue": str, 
            "actions_or_visuals": str
        }, ...]
        Returns empty list on failure.
    """
    if not scenes:
        print("Error: Cannot generate dialogue without scenes.")
        return []

    # We'll process by scene groups to maintain context while limiting image count
    all_dialogue_blocks = []
    
    # Map image labels to their paths for quick lookup
    image_path_map = {img["label"]: img["path"] for img in images_info if "label" in img and "path" in img}
    
    # Group scenes to keep under MAX_IMAGES_PER_CHUNK limit
    scene_groups = []
    current_group = []
    current_group_images = set()
    
    for scene in scenes:
        scene_images = scene.get("images_in_scene", [])
        # If adding this scene would exceed the image limit, start a new group
        if len(current_group_images.union(scene_images)) > MAX_IMAGES_PER_CHUNK and current_group:
            scene_groups.append(current_group)
            current_group = []
            current_group_images = set()
        
        current_group.append(scene)
        current_group_images.update(scene_images)
    
    # Add the last group if not empty
    if current_group:
        scene_groups.append(current_group)
    
    # Prepare custom instructions if provided
    custom_instructions_text = ""
    if custom_instructions:
        custom_instructions_text = f"""
CUSTOM INSTRUCTIONS FROM USER:
{custom_instructions}

Please follow these custom instructions while generating the dialogue content.
"""
        
    # Process each scene group
    for group_idx, scene_group in enumerate(scene_groups):
        print(f"Processing scene group {group_idx + 1}/{len(scene_groups)} containing {len(scene_group)} scenes.")
        
        # Collect all image labels from this group for prompt context
        all_group_image_labels = []
        for scene in scene_group:
            all_group_image_labels.extend(scene.get("images_in_scene", []))
        all_group_image_labels = list(set(all_group_image_labels))  # Remove duplicates
        
        # Collect paths for images in this group
        group_image_paths = [image_path_map[label] for label in all_group_image_labels if label in image_path_map]
        
        # Construct scene info for this group
        scenes_text = "\n".join([
            f"Scene {scene['scene_number']}: {scene['content']}\nImages: {', '.join(scene['images_in_scene']) if scene['images_in_scene'] else 'None'}"
            for scene in scene_group
        ])
        
        # Define the sequence of scene numbers we want dialogue for
        scene_numbers = [scene["scene_number"] for scene in scene_group]
        
        # Build the prompt
        image_context = ""
        if all_group_image_labels:
            image_context = f"""\
IMPORTANT: I'm providing {len(all_group_image_labels)} images that appear in these scenes, referenced by labels like [IMG_1]. Your dialogue should reference these images when relevant to enhance the educational experience.
"""

        prompt = f"""\
You are creating an educational video script with dialogue blocks for each scene. The dialogue should explain the educational content in a clear, engaging way based on the scene content. Each scene can have multiple dialogue blocks.

{image_context}

{custom_instructions_text}

Scenes to create dialogue for:
{scenes_text}

Task:
Generate a JSON response with an array of dialogue blocks. Each block should include:
1. "scene_number": The scene this dialogue belongs to ({', '.join(map(str, scene_numbers))})
2. "speaker": Always use "Narrator"  
3. "dialogue": What the narrator will say - clear, educational explanation of the scene content
4. "actions_or_visuals": Brief, visual-focused instructions (e.g., "Show diagram of [IMG_1]" or "Highlight key points on screen")

Format your response as a JSON array where each object is a dialogue block:
[
  {{
    "scene_number": 1,
    "speaker": "Narrator",
    "dialogue": "Welcome to our lesson on...",
    "actions_or_visuals": "Display title slide with [IMG_1]"
  }},
  {{
    "scene_number": 1,
    "speaker": "Narrator", 
    "dialogue": "Let's begin by exploring...",
    "actions_or_visuals": "Point to the diagram in [IMG_2]"
  }},
  ...
]

IMPORTANT GUIDELINES:
- Each scene should have 1-3 dialogue blocks to thoroughly explain its content
- Dialogue should be natural and conversational educational content
- Each dialogue block should focus on ONE clear teaching point
- Dialogue content should be 2-5 sentences per block (not too long)
- Consider appropriate uses of images when mentioned in the scene
- Ensure ALL dialogue blocks include ALL required fields (scene_number, speaker, dialogue, actions_or_visuals)
"""

        try:
            # Call Gemini with or without images
            if group_image_paths:
                print(f"Calling Gemini with {len(group_image_paths)} images for dialogue generation...")
                result_json = _call_gemini_with_images(prompt, group_image_paths, is_json_output=True)
            else:
                print("Calling Gemini for dialogue generation (no images)...")
                result_json = _call_gemini_text_only(prompt, is_json_output=True)
                
            # Handle different JSON response formats
            dialogue_blocks = []
            if isinstance(result_json, list):
                # Direct array format
                dialogue_blocks = result_json
            elif isinstance(result_json, dict) and "dialogueBlocks" in result_json:
                # Wrapped in dialogueBlocks key
                dialogue_blocks = result_json["dialogueBlocks"]
            elif isinstance(result_json, dict) and "dialogue" in result_json:
                # Wrapped in dialogue key
                dialogue_blocks = result_json["dialogue"]
            elif isinstance(result_json, dict) and "blocks" in result_json:
                # Wrapped in blocks key
                dialogue_blocks = result_json["blocks"]
            else:
                print(f"Warning: Unexpected JSON structure for dialogue in group {group_idx + 1}: {result_json}")
                
            # Validate each dialogue block
            valid_blocks = []
            for block in dialogue_blocks:
                if (isinstance(block, dict) and
                    "scene_number" in block and isinstance(block["scene_number"], int) and
                    "speaker" in block and isinstance(block["speaker"], str) and
                    "dialogue" in block and isinstance(block["dialogue"], str) and
                    "actions_or_visuals" in block and isinstance(block["actions_or_visuals"], str)):
                    
                    # Ensure the scene number is in our group
                    if block["scene_number"] in scene_numbers:
                        valid_blocks.append(block)
                    else:
                        print(f"Warning: Dialogue block references scene {block['scene_number']} which is not in the current group. Skipping.")
                else:
                    print(f"Warning: Invalid dialogue block structure: {block}")
                    
            print(f"Group {group_idx + 1}: Generated {len(valid_blocks)} valid dialogue blocks.")
            all_dialogue_blocks.extend(valid_blocks)
                
        except Exception as e:
            print(f"Error generating dialogue for scene group {group_idx + 1}: {e}")
            # Continue with other groups if one fails
            
    # Final validation and sorting
    if not all_dialogue_blocks:
        print("Warning: No valid dialogue blocks were generated.")
        return []
        
    # Sort by scene number only
    all_dialogue_blocks.sort(key=lambda x: x["scene_number"])
    
    print(f"Successfully generated {len(all_dialogue_blocks)} dialogue blocks across {len(scene_groups)} scene groups.")
    return all_dialogue_blocks

# --- REMOVED OLD HELPER FUNCTIONS --- 
# chunk_text_old and _call_gemini_old removed as they are no longer used.

# --- Ensure Old Functions are Removed ---
# The old generate_lesson_and_dialogue and generate_scenes functions should not be present below this line. 