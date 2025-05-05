import PyPDF2
from typing import List

def extract_text_from_pdfs(pdf_paths: List[str]) -> str:
    """Extracts text from a list of PDF files and concatenates it.

    Args:
        pdf_paths: A list of paths to the PDF files.

    Returns:
        A single string containing the concatenated text from all PDFs.
    """
    combined_text = ""
    for pdf_path in pdf_paths:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    combined_text += page.extract_text() or "" # Add empty string if extraction returns None
                combined_text += "\n\n---\n\n" # Add a separator between documents
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            # Optionally, re-raise the exception or handle it differently
            # For now, just print an error and continue
    return combined_text 