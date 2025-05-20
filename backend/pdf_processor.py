import os
from typing import List, Tuple, Dict
# Remove PyPDF2 import if no longer needed directly
# import PyPDF2
from .pdf_utils import extract_text_and_images_from_pdf # Use the new util

def process_uploaded_pdfs(pdf_paths: List[str], session_id: str) -> Tuple[str, List[Dict[str, str]]]:
    """Extracts text and images from a list of PDF files.

    Processes each PDF to extract text and images, places placeholders in the text,
    annotates images with labels, and saves them.

    Args:
        pdf_paths: A list of paths to the uploaded PDF files.
        session_id: A unique identifier for the current session/request.

    Returns:
        A tuple containing:
         - A single string containing the concatenated text from all PDFs,
           with image placeholders.
         - A list of dictionaries for all extracted and annotated images across all PDFs.
    """
    combined_text = ""
    all_images_info = []
    base_output_dir = os.path.join("results", session_id)
    image_output_dir = os.path.join(base_output_dir, "annotated_images")

    os.makedirs(image_output_dir, exist_ok=True)

    # Use a global counter for image labels across multiple PDFs in the same session
    # (Alternatively, could reset per PDF if desired)
    # Let pdf_utils handle the counter for simplicity for now.

    print(f"Starting PDF processing for session {session_id}")

    for pdf_path in pdf_paths:
        try:
            print(f"Processing PDF: {pdf_path}")
            # Define a specific output dir per PDF if needed, or use the common one
            # For simplicity, using the common one for all images in this session
            pdf_text, pdf_images_info = extract_text_and_images_from_pdf(pdf_path, image_output_dir)

            if pdf_text:
                combined_text += pdf_text
                combined_text += "\n\n---\n\n" # Add a separator between documents
            if pdf_images_info:
                all_images_info.extend(pdf_images_info)

        except Exception as e:
            print(f"Error processing {pdf_path} in process_uploaded_pdfs: {e}")
            # Add error placeholder to text?
            combined_text += f"\n\n--- ERROR PROCESSING {os.path.basename(pdf_path)} ---\n\n"

    print(f"Finished PDF processing for session {session_id}. Total images: {len(all_images_info)}")
    return combined_text.strip(), all_images_info 