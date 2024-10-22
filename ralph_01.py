# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:56:23 2024

@author: Magnolia
"""

import re
import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

#%%

# Initialize Chroma
client = chromadb.Client()

# Check if collection already exists and use it, otherwise create it
try:
    collection = client.create_collection("research_papers")
except chromadb.errors.UniqueConstraintError:
    collection = client.get_collection("research_papers")
# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

#%%
def extract_content_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        figures = []
        tables = []
        title, authors, year, journal = None, None, None, None

        for page_num, page in enumerate(pdf.pages):
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text += page_text

                # Extract title (Assume it's the first large text block on the first page)
                if page_num == 0 and not title:
                    title = extract_title(page_text)

                # Extract authors (Common formats include "by", "Authors:", etc.)
                if not authors:
                    authors = extract_authors(page_text)

                # Extract year of publication
                if not year:
                    year = extract_year(page_text)

                # Extract journal name
                if not journal:
                    journal = extract_journal(page_text)

            # Extract figures and their captions
            figures_on_page = page.images
            for figure in figures_on_page:
                caption = None
                if 'bbox' in figure:
                    caption = page.within_bbox(figure['bbox']).extract_text()
                figures.append({
                    "page": page_num,
                    "bbox": figure.get('bbox', 'N/A'),  # Provide 'N/A' if bbox is missing
                    "caption": caption if caption else 'No caption available'
                })

            # Extract tables using pdfplumber
            table_data = page.extract_tables()
            if table_data:
                for table in table_data:
                    df = pd.DataFrame(table)
                    tables.append(df)

    # Default values if extraction failed
    if not authors:
        authors = ["Unknown Author"]
    if not title:
        title = "Unknown Title"
    if not year:
        year = "Unknown Year"
    if not journal:
        journal = "Unknown Journal"

    return text, figures, tables, title, authors, year, journal


# Helper function to extract title
def extract_title(text):
    # Assuming the title is the first capitalized text block, often found at the start of the document
    title_match = re.search(r'^[A-Z][^\n]+(?:\n|$)', text)
    if title_match:
        return title_match.group(0).strip()
    return None

# Helper function to extract authors
def extract_authors(text):
    # Simple regex pattern for authors (can be improved based on document structure)
    authors_match = re.search(r'(?i)by\s+(.*?)(?=\n|\r|\.)', text)
    if authors_match:
        authors_text = authors_match.group(1)
        authors = [author.strip() for author in authors_text.split(",")]
        return authors
    return None

# Helper function to extract year of publication
def extract_year(text):
    # Look for a four-digit year (e.g., 2020, 2021, etc.)
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        return year_match.group(0)
    return None

# Helper function to extract journal name
def extract_journal(text):
    # Attempt to extract journal name, often found in "Journal of ..." format
    journal_match = re.search(r'Journal of [^\n]+', text, re.IGNORECASE)
    if journal_match:
        return journal_match.group(0).strip()
    return None


# Generate a citation string
def generate_citation(authors, title, journal, year):
    return f"{', '.join(authors)}. {year}. {title}. {journal}."

# Create embeddings for text
def create_embeddings(text):
    return model.encode([text])[0]

def add_to_chroma_with_metadata(collection, doc_id, text, figures, tables, file_path, citation):
    # Add main text embedding
    text_embedding = create_embeddings(text)
    collection.add(
        ids=[f"{doc_id}_text"],
        embeddings=[text_embedding],
        metadatas=[{
            "type": "text",
            "content": text,
            "file_path": file_path,
            "citation": citation,
            "doc_id": doc_id
        }]
    )

    # Add figure captions with file path and citation
    for idx, caption in enumerate(figures):  # figures is now a list of captions (strings)
        caption_embedding = create_embeddings(caption)  # Treat caption as a string
        collection.add(
            ids=[f"{doc_id}_figure_{idx}"],
            embeddings=[caption_embedding],
            metadatas=[{
                "type": "figure",
                "caption": caption,
                "file_path": file_path,
                "citation": citation,
                "doc_id": doc_id
            }]
        )

    # Add table embeddings with file path and citation
    for idx, table in enumerate(tables):
        table_embedding = create_embeddings(table.to_string())
        collection.add(
            ids=[f"{doc_id}_table_{idx}"],
            embeddings=[table_embedding],
            metadatas=[{
                "type": "table",
                "content": table.to_string(),
                "file_path": file_path,
                "citation": citation,
                "doc_id": doc_id
            }]
        )


# Query Chroma for relevant documents
def query_chroma(query, collection, num_results=5):
    query_embedding = create_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
    )
    return results['ids'], results['metadatas']

#%% PyQt GUI for uploading PDFs and performing queries

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar,
    QFileDialog, QDialog, QPushButton, QListWidget, QHBoxLayout
)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import chromadb

# Initialize Chroma client
client = chromadb.Client()

class FileProcessingThread(QThread):
    # Custom signals
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, file_paths, collection):
        super().__init__()
        self.file_paths = file_paths
        self.collection = collection

    def run(self):
        # Process each file and emit progress
        for index, pdf_path in enumerate(self.file_paths):
            # Simulate file processing (replace this with real logic)
            print(f"Processing file: {pdf_path}")
            self.collection.add(
                ids=[pdf_path],  # Using the path as ID
                metadatas=[{'file_path': pdf_path}]
            )

            # Emit progress signal
            self.progress.emit(index + 1)

        # Emit finished signal when done
        self.finished.emit()


class ProgressDialog(QDialog):
    def __init__(self, total_files):
        super().__init__()
        self.total_files = total_files
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Loading Documents')
        self.setFixedSize(400, 100)
        self.setWindowModality(Qt.ApplicationModal)

        self.label = QLabel('Loading documents, please wait...', self)
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(self.total_files)
        self.progress_bar.setValue(0)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def update_progress(self, value):
        self.progress_bar.setValue(value)


class ReferenceManager(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Reference Manager')
        self.setGeometry(100, 100, 1000, 600)

        # Set the window icon
        self.setWindowIcon(QIcon('assets/icon.png'))

        # Create a QLabel for the title and set a larger font size
        title_label = QLabel("Reference Manager", self)
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)

        # Create a QLabel to display the larger icon
        icon_label = QLabel(self)
        pixmap = QPixmap('assets/icon.png')
        scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignCenter)

        # Create Upload Button
        upload_button = QPushButton('Upload Documents', self)
        upload_button.clicked.connect(self.upload_pdf)

        # Create a QListWidget to display the unique file paths in the database
        self.file_list_widget = QListWidget(self)
        self.update_file_list()  # Populate the file list at startup

        # Create the main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(icon_label)
        main_layout.addWidget(upload_button)
        main_layout.setAlignment(upload_button, Qt.AlignCenter)

        # Create a horizontal layout to add the file list panel
        layout_with_list = QHBoxLayout()
        layout_with_list.addLayout(main_layout)
        layout_with_list.addWidget(self.file_list_widget)  # Add the file list panel to the right side

        self.setLayout(layout_with_list)
        self.show()

    def upload_pdf(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, 'Upload PDFs', '', 'PDF Files (*.pdf)')
        if file_paths:
            self.process_files(file_paths)
            return

    def process_files(self, file_paths):
        total_files = len(file_paths)
        if total_files == 0:
            return

        # Create the progress dialog
        progress_dialog = ProgressDialog(total_files)
        progress_dialog.show()

        # Start a QThread to process files
        collection = client.get_or_create_collection("research_papers")
        self.worker_thread = FileProcessingThread(file_paths, collection)

        # Connect signals from the thread to update the progress bar
        self.worker_thread.progress.connect(progress_dialog.update_progress)
        self.worker_thread.finished.connect(progress_dialog.accept)  # Close the dialog when done

        # Start the worker thread
        self.worker_thread.start()

        # Update the file list after processing
        self.worker_thread.finished.connect(self.update_file_list)

    def update_file_list(self):
        # Clear the current list
        self.file_list_widget.clear()

        # Get the collection from Chroma
        collection = client.get_collection("research_papers")

        # Fetch all the documents and extract unique file paths
        unique_file_paths = set()
        all_documents = collection.get(ids=None)  # Fetch all documents (None retrieves all)
        if 'metadatas' in all_documents:
            for metadata in all_documents['metadatas']:
                file_path = metadata.get('file_path')
                if file_path and file_path not in unique_file_paths:
                    unique_file_paths.add(file_path)

        # Populate the QListWidget with unique file paths
        for file_path in unique_file_paths:
            self.file_list_widget.addItem(file_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReferenceManager()
    sys.exit(app.exec_())







#%%

# # Initialize Chroma
# client = chromadb.Client()

# # Delete the "research_papers" collection
# try:
#     client.delete_collection("research_papers")
#     print("Collection 'research_papers' deleted successfully.")
# except Exception as e:
#     print(f"Error deleting collection: {e}")
