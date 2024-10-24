# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:47:38 2024

@author: Magnolia
"""
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar,
    QFileDialog, QDialog, QPushButton, QListWidget, QHBoxLayout
)
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import chromadb

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