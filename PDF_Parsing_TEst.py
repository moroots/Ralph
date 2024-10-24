# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:01:05 2024

@author: Magnolia
"""


import pdfplumber
import fitz  # PyMuPDF
from PyPDF2 import PdfReader

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import re

#%%

def _plot_image(pil_image, **kwargs):
    # Display the image using matplotlib
    plt.imshow(pil_image)
    plt.axis("off")  # No axis for clean display
    plt.show()
    return

def _image_bytes_to_image(image_bytes, **kwargs):
    # Convert image bytes into an Image object using PIL
    image_stream = BytesIO(image_bytes)
    pil_image = Image.open(image_stream)
    return pil_image


class PDFProcessor:

    def __init__(self, pdf_path):
        self.patterns = {"figure_captions": [r"Figure \d+", r"Fig\. \d+", r"Figure", r"Fig\."],
                   "references":  ["References \n", "Reference \n", "Bibliography \n", "Citations \n", "Works Cited \n", "Literature Cited \n", "References\n"],
                   "acknowledgments": ["Acknowledgments \n"]
                   }
        self.troubleshoot = []
        self.result = self._parse(pdf_path)
        return

    def filter_caption_text(self, caption_text):
        """
        Filters the extracted caption text to keep only the portion after 'Figure' or 'Fig.'
        """

        # Search for the first occurrence of any of the patterns
        regex_pattern = r"(" + "|".join(self.patterns["figure_captions"]) + ")"
        match = re.search(regex_pattern, caption_text, re.IGNORECASE)
        if match:
            # Return the text after the match
            return caption_text[match.start():].strip()
        return "Caption Unknown"

    # Define a function to extract text around an image
    def _get_fig_caption(self, page, img_rect, margin=50):
        """
        Extracts text from the full width of the page below the image.
        img_rect: the rectangle around the image.
        margin: the vertical margin below the image to look for text (default is 50).
        """
        # Get the full page width
        page_width = page.rect.width

        # Define a rectangle that spans the full width of the page, starting below the image
        caption_rect = fitz.Rect(
            0,                       # Start at the left edge (x0)
            img_rect.y1,             # Start from the bottom of the image (y1)
            page_width,              # Full page width (x1)
            img_rect.y1 + margin     # Only a small margin below the image (adjust margin as needed)
        )

        text = page.get_text("text", clip=caption_rect)
        # Extract text within this area
        return self.filter_caption_text(text)

    def _get_figure(self, page, img, **kwargs):

        # Grab the Figure bbox and figure data
        # img_rect = page.get_image_bbox(img[7])

        try:
            figure = self.pdf.extract_image(img[0]) # Get the bounding box of the image (img[7] refers to the image name in PyMuPDF)
            img_rect = page.get_image_bbox(img[7])
            figure["caption"] = self._get_fig_caption(page, img_rect)

            # rename image too image_bytes
            figure["image_bytes"] = figure.pop("image")
            figure["img"] = img

        except ValueError as e:
             self.troubleshoot.append(f"Warning: {e} - Skipping image with name {img[7]}.")

        return figure

    def _get_references(self, all_text):

        # Search for the first occurrence of any of the patterns
        regex_pattern = r"(" + "|".join(self.patterns["references"]) + ")"
        match = re.search(regex_pattern, all_text, re.IGNORECASE)

        # Extract text starting from the references section
        if match:
            ref = all_text[match.start():].strip()
            ref = '\n'.join(ref.split("\n")[1:])
            return '||'.join(re.split(r'\.\s*\n(?=[A-Za-z]+,)', ref))
        return "References Unknown"

    def _get_acknowledgments(self, all_text):

        """BROKEN"""

        # Search for the first occurrence of any of the patterns
        regex_pattern = r"(" + "|".join(self.patterns["acknowledgments"]) + ")"
        match = re.search(regex_pattern, all_text, re.IGNORECASE)

        # Extract text starting from the references section
        if match:
            return all_text[match.start():].strip()
        return "References Unknown"

    def _remove_references(self, all_text):
        # Search for the first occurrence of any of the patterns
        regex_pattern = r"(" + "|".join(self.patterns["references"]) + ")"
        match = re.search(regex_pattern, all_text, re.IGNORECASE)

        # Extract text starting from the references section
        if match:
            return all_text[:match.start()].strip()
        return all_text

    def _remove_acknowledgments(self, all_text):
        # Search for the first occurrence of any of the patterns
        regex_pattern = r"(" + "|".join(self.patterns["acknowledgments"]) + ")"
        match = re.search(regex_pattern, all_text, re.IGNORECASE)

        # Extract text starting from the references section
        if match:
            return all_text[:match.start()].strip()
        return all_text


    def _filter_text(self, all_text):
        parsed_text = {"references": self._get_references(all_text),
                       "text": self._remove_acknowledgments(self._remove_references(all_text))}
        return parsed_text

    def _parse(self, pdf_path):
        result = {"text": {"all_text": ''}, "tables": {}, "images": {}}
        self.pdf = fitz.open(pdf_path)

        img_num = 0; page_num = 0
        for page_num in range(self.pdf.page_count):
            page_num += 1
            page = self.pdf.load_page(page_num-1)

            # Extract text
            result["text"]["all_text"] += page.get_text("text")


            # Extract images
            images = page.get_images(full=True)
            for img in images:
                img_num+=1
                result["images"][f"image_{img_num}"] = self._get_figure(page, img)
                result["images"][f"image_{img_num}"]["page_num"] = page_num

        result["text"].update(self._filter_text(result["text"]["all_text"]))
        result["metadata"] = self.pdf.metadata
        self.pdf.close()
        return result


#%%
import chromadb
from sentence_transformers import SentenceTransformer
import json
import base64

class PDFVectorStorage:
    def __init__(self, collection_name):

        # Initialize Chroma
        self.client = chromadb.PersistentClient(path="./db")

        # Check if collection already exists and use it, otherwise create it
        try:
            self.collection = self.client.create_collection(collection_name)
        except chromadb.errors.UniqueConstraintError:
            self.collection = self.client.get_collection(collection_name)

        # Load a pre-trained model to generate embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return

    def _create_embeddings(self, text):
        return self.model.encode(text)

    def _update_db(self, doc_id, filepath, pdf: dict):
        # Add main text embedding
        text_embedding = self._create_embeddings(pdf["text"]["text"])

        metadata = {"type": "text",
                    "filepath": filepath,
                    "doc_id": doc_id
                    }
        metadata.update(pdf["metadata"])
        metadata = {key: str(value) for key, value in metadata.items()}

        # Convert the dictionary into a JSON string
        document_json = json.dumps(pdf["text"])

        self.collection.add(ids=[f"{doc_id}_text"],
                            embeddings=[text_embedding],
                            metadatas=[metadata],
                            documents=[document_json]
                            )

        # Add figure captions with file path and citation
        metadata.update({"type": "image"})
        for key in pdf["images"].keys():  # figures is now a list of captions (strings)
            if "caption" in pdf["images"][key].keys():
                caption_embedding = self._create_embeddings(pdf["images"][key]["caption"])  # Treat caption as a string
                pdf["images"][key]["image_bytes"] = base64.b64encode(pdf["images"][key]["image_bytes"]).decode('utf-8')

                # Convert the dictionary into a JSON string
                document_json = json.dumps(pdf["images"][key])
                self.collection.add(ids=[f"{doc_id}_{key}"],
                                    embeddings=[caption_embedding],
                                    metadatas=[metadata],
                                    documents=[document_json]
                                    )

        # Add figure captions with file path and citation
        metadata.update({"type": "table"})
        for key in pdf["tables"].keys():  # figures is now a list of captions (strings)
            if "caption" in pdf["tables"][key].keys():
                caption_embedding = self._create_embeddings(pdf["tables"][key]["caption"])  # Treat caption as a string

                # Convert the dictionary into a JSON string
                document_json = json.dumps(pdf["tables"][key])
                self.collection.add(ids=[f"{doc_id}_{key}"],
                                    embeddings=[caption_embedding],
                                    metadatas=[metadata],
                                    documents=[document_json]
                                    )
        return

    def _query_db(self, query, collection, num_results=100):
        query_embedding = self._create_embeddings(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results,
        )
        return results

    def _unique_filepaths(self, query_results):
        filepaths = [metadata['filepath'] for metadata in query_results['metadatas'][0]]
        return list(set(filepaths))

#%%
import time
from pathlib import Path

# Start timing
start_time = time.time()

pdf_files = [x for x in Path("./pdfs").glob("*.pdf")]  # Replace with the path to your PDF file

storage = PDFVectorStorage("research_papers")

test = {}
for file in pdf_files:
    filepath = str(file.resolve())
    test[file.name] = PDFProcessor(file).result
    storage._update_db(doc_id=filepath, filepath=filepath, pdf=test[file.name])

# End timing
end_time = time.time()

# Print elapsed time
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

#%%




#%%

# # Initialize Chroma
# client = chromadb.PersistentClient(path="./db")

# # Delete the "research_papers" collection
# try:
#     client.delete_collection("research_papers")
#     print("Collection 'research_papers' deleted successfully.")
# except Exception as e:
#     print(f"Error deleting collection: {e}")























