import fitz  # PyMuPDF
import os
import io
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict

def _add_label_to_image(image_bytes: bytes, label: str, output_path: str, font_size: int = 20, padding: int = 5):
    """Adds a text label to the top of an image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = img.size

        # Use a basic font (Pillow's default)
        try:
            # Try loading a system font (more reliable if available)
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            # Fallback to Pillow's default bitmap font
            print("Arial font not found. Using default PIL font.")
            font = ImageFont.load_default()
            # Adjust font size estimate for default font if needed
            # font_size = 20 # Or adjust based on default font appearance


        # Calculate text size (handle potential AttributeError for default font)
        try:
            text_bbox = font.getbbox(label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Estimate size for default font
            # This is less accurate
            text_width = len(label) * (font_size // 2) # Rough estimate
            text_height = font_size # Rough estimate


        label_height = text_height + 2 * padding
        new_height = original_height + label_height

        # Create new image with space for the label
        new_img = Image.new("RGB", (original_width, new_height), "white")

        # Paste original image
        new_img.paste(img, (0, label_height))

        # Draw the label text
        draw = ImageDraw.Draw(new_img)
        text_x = (original_width - text_width) // 2
        text_y = padding
        draw.text((text_x, text_y), label, fill="black", font=font)

        # Determine save format
        img_format = img.format or 'PNG' # Default to PNG if format unknown
        if img_format not in ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP', 'TIFF']:
            print(f"Original format {img_format} not ideal for saving, using PNG.")
            img_format = 'PNG'
            # Ensure output path has the correct extension
            output_path = os.path.splitext(output_path)[0] + ".png"


        new_img.save(output_path, format=img_format)
        return output_path # Return the actual saved path

    except Exception as e:
        print(f"Error processing/annotating image for label {label}: {e}")
        # Fallback: Save raw image bytes without label if annotation fails
        try:
            fallback_path = os.path.splitext(output_path)[0] + "_raw" + os.path.splitext(output_path)[1]
            with open(fallback_path, "wb") as f_raw:
                f_raw.write(image_bytes)
            print(f"Saved raw image bytes to {fallback_path}")
            return fallback_path # Return the raw path as fallback
        except Exception as e_raw:
            print(f"Error saving raw image bytes for label {label}: {e_raw}")
            return None # Indicate failure


def extract_text_and_images_from_pdf(pdf_path: str, output_image_dir: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Extracts text and images from a PDF. Places placeholders in text and saves annotated images.

    Args:
        pdf_path: Path to the PDF file.
        output_image_dir: Directory to save annotated images.

    Returns:
        A tuple containing:
            - Extracted text with image placeholders (e.g., "[IMG_1]").
            - A list of dictionaries, each containing info about an extracted image
              (e.g., {"label": "[IMG_1]", "path": "/path/to/IMG_1.png"}).
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return "", []

    os.makedirs(output_image_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    combined_text = ""
    extracted_images_info = []
    image_counter = 0

    print(f"Processing PDF: {pdf_path}")

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)

        # Extract text for the page
        page_text = page.get_text("text")
        combined_text += page_text + "\n" # Add page text

        # Extract images
        image_list = page.get_images(full=True)
        if image_list:
            print(f"Found {len(image_list)} images on page {page_index + 1}")
            for img_index, img_info in enumerate(image_list):
                image_counter += 1
                label = f"[IMG_{image_counter}]"
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Define image filename based on label
                    image_filename = f"{label[1:-1]}.{image_ext}" # e.g., IMG_1.png
                    annotated_image_path = os.path.join(output_image_dir, image_filename)

                    # Add label to image and save
                    saved_path = _add_label_to_image(image_bytes, label, annotated_image_path)

                    if saved_path:
                        # Insert placeholder at the end of the current page's text
                        combined_text += f" {label} \n"
                        extracted_images_info.append({"label": label, "path": saved_path})
                        print(f"Saved annotated image: {saved_path}")
                    else:
                        print(f"Failed to save image {label} for page {page_index + 1}")

                except Exception as e:
                    print(f"Error extracting image xref {xref} on page {page_index + 1}: {e}")


    doc.close()
    print(f"Finished processing {pdf_path}. Extracted {image_counter} images.")
    return combined_text.strip(), extracted_images_info 