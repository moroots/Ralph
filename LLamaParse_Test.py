# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:55:57 2024

@author: Magnolia
"""


# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# set up parser
parser = LlamaParse(
    result_type="text"  # "markdown" and "text" are available
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['Roots_et_al_2023.pdf'], file_extractor=file_extractor).load_data()
print(documents)





