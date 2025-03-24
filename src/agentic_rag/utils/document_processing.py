"""Document processing utilities for the Agentic RAG system."""

import os
from typing import List, Dict, Any, Optional, Union

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_document(file_path: str) -> List[Document]:
    """
    Load a document from various file formats.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects
        
    Raises:
        ValueError: If the file format is not supported
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.csv': CSVLoader,
        '.md': UnstructuredMarkdownLoader
    }
    
    if file_extension not in loaders:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    loader_cls = loaders[file_extension]
    loader = loader_cls(file_path)
    
    return loader.load()

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects after splitting
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return text_splitter.split_documents(documents)

def load_and_split_documents(
    file_paths: List[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Load and split documents into chunks.
    
    Args:
        file_paths: List of paths to documents
        chunk_size: Size of each document chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    documents = []
    
    for file_path in file_paths:
        # Determine the appropriate loader based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        elif ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Load the document
        documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    return text_splitter.split_documents(documents) 