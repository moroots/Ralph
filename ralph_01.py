# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:56:23 2024

@author: Magnolia
"""

#%%
import sys
import re
import os
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar,
    QFileDialog, QDialog, QPushButton, QListWidget, QHBoxLayout
)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Initialize Chroma
client = chromadb.Client()

# Check if collection already exists and use it, otherwise create it
try:
    collection = client.create_collection("research_papers")
except chromadb.errors.UniqueConstraintError:
    collection = client.get_collection("research_papers")
# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

                # Extract authors
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
                    "bbox": figure.get('bbox', 'N/A'),
                    "caption": caption if caption else 'No caption available'
                })

            # Extract tables using pdfplumber
            table_data = page.extract_tables()
            if table_data:
                for table in table_data:
                    df = pd.DataFrame(table)
                    tables.append(df)

    if not authors:
        authors = ["Unknown Author"]
    if not title:
        title = "Unknown Title"
    if not year:
        year = "Unknown Year"
    if not journal:
        journal = "Unknown Journal"

    return text, figures, tables, title, authors, year, journal


# Helper functions to extract title, authors, year, and journal
def extract_title(text):
    title_match = re.search(r'^[A-Z][^\n]+(?:\n|$)', text)
    if title_match:
        return title_match.group(0).strip()
    return None

def extract_authors(text):
    authors_match = re.search(r'(?i)by\s+(.*?)(?=\n|\r|\.)', text)
    if authors_match:
        authors_text = authors_match.group(1)
        authors = [author.strip() for author in authors_text.split(",")]
        return authors
    return None

def extract_year(text):
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        return year_match.group(0)
    return None

def extract_journal(text):
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
    for idx, figure in enumerate(figures):
        caption = figure['caption']
        caption_embedding = create_embeddings(caption)
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

# Multithreaded processing of PDFs
class FileProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, file_paths, collection):
        super().__init__()
        self.file_paths = file_paths
        self.collection = collection

    def run(self):
        for index, pdf_path in enumerate(self.file_paths):
            print(f"Processing file: {pdf_path}")
            text, figures, tables, title, authors, year, journal = extract_content_from_pdf(pdf_path)
            citation = generate_citation(authors, title, journal, year)
            doc_id = os.path.basename(pdf_path).split('.')[0]
            add_to_chroma_with_metadata(self.collection, doc_id, text, figures, tables, pdf_path, citation)

            self.progress.emit(index + 1)
        self.finished.emit()

# PyQt GUI
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

        self.setWindowIcon(QIcon('assets/icon.png'))

        title_label = QLabel("Reference Manager", self)
        title_font = QFont("Arial", 20, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)

        icon_label = QLabel(self)
        pixmap = QPixmap('assets/icon.png')
        scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignCenter)

        upload_button = QPushButton('Upload Documents', self)
        upload_button.clicked.connect(self.upload_pdf)

        self.file_list_widget = QListWidget(self)
        self.update_file_list()

        main_layout = QVBoxLayout()
        main_layout.addWidget(title_label)
        main_layout.addWidget(icon_label)
        main_layout.addWidget(upload_button)
        main_layout.setAlignment(upload_button, Qt.AlignCenter)

        layout_with_list = QHBoxLayout()
        layout_with_list.addLayout(main_layout)
        layout_with_list.addWidget(self.file_list_widget)

        self.setLayout(layout_with_list)
        self.show()

    def upload_pdf(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, 'Upload PDFs', '', 'PDF Files (*.pdf)')
        if file_paths:
            self.process_files(file_paths)

    def process_files(self, file_paths):
        total_files = len(file_paths)
        if total_files == 0:
            return

        progress_dialog = ProgressDialog(total_files)
        progress_dialog.show()

        collection = client.get_or_create_collection("research_papers")
        self.worker_thread = FileProcessingThread(file_paths, collection)

        self.worker_thread.progress.connect(progress_dialog.update_progress)
        self.worker_thread.finished.connect(progress_dialog.accept)
        self.worker_thread.finished.connect(self.update_file_list)
        self.worker_thread.start()

    def update_file_list(self):
        self.file_list_widget.clear()
        collection = client.get_collection("research_papers")
        unique_file_paths = set()
        all_documents = collection.get(ids=None)

        if 'metadatas' in all_documents:
            for metadata in all_documents['metadatas']:
                file_path = metadata.get('file_path')
                if file_path and file_path not in unique_file_paths:
                    unique_file_paths.add(file_path)

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
