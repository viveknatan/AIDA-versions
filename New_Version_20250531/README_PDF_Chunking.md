# PDF Section-Based Chunking for RAG

This repository contains a comprehensive solution for chunking PDF documents based on their sections, specifically designed for Retrieval-Augmented Generation (RAG) applications.

## 🎯 Overview

The PDF chunker analyzes document structure (font sizes, formatting, headers) to intelligently split content into meaningful sections rather than arbitrary text blocks. This preserves document context and creates more useful chunks for vector databases and RAG systems.

## 📁 Files Created

- **`pdf_section_chunker.py`** - Main chunking library with `PDFSectionChunker` class
- **`rag_chunking_example.py`** - Complete demonstration script
- **`northwind_rag_ready.json`** - RAG-ready chunks with full metadata (for vector databases)
- **`northwind_chunks_analysis.csv`** - Tabular analysis data
- **`northwind_chunks_readable.txt`** - Human-readable text export

## 🚀 Quick Start

### Basic Usage

```python
from pdf_section_chunker import chunk_northwind_pdf

# Simple one-line chunking
chunks = chunk_northwind_pdf("data/Northwind_Traders_Database_Overview.pdf")

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"Section: {chunk.title}")
    print(f"Content: {chunk.content[:100]}...")
```

### Custom Parameters

```python
from pdf_section_chunker import PDFSectionChunker

# Create chunker with custom settings
chunker = PDFSectionChunker(
    min_chunk_size=200,  # Minimum characters per chunk
    max_chunk_size=1000  # Maximum characters per chunk
)

chunks = chunker.chunk_by_sections("your_pdf_file.pdf")
```

## 🔧 Key Features

### Intelligent Section Detection
- **Font Analysis**: Identifies headers based on font size and formatting
- **Style Recognition**: Detects bold text, numbering, and capitalization patterns
- **Hierarchical Structure**: Maintains document outline with section levels

### RAG-Optimized Output
- **Rich Metadata**: Page numbers, section levels, character counts
- **Search Keywords**: Automatically extracted for better retrieval
- **Context Preservation**: Maintains document structure and relationships
- **Vector DB Ready**: Formatted for immediate use in vector databases

### Flexible Chunking
- **Size Control**: Configurable min/max chunk sizes
- **Smart Splitting**: Large sections split at sentence boundaries
- **Content Cleaning**: Removes artifacts and normalizes text

## 📊 Chunk Structure

Each `DocumentChunk` contains:

```python
@dataclass
class DocumentChunk:
    title: str           # Section title/header
    content: str         # Main text content
    page_number: int     # Source page
    section_level: int   # Hierarchical level (1=top, 4=sub-sub-section)
    metadata: Dict       # Additional information
```

## 🗄️ RAG Database Integration

### For Vector Databases (Qdrant, Pinecone, Weaviate)

```python
# Load the RAG-ready JSON
import json
with open('northwind_rag_ready.json', 'r') as f:
    rag_documents = json.load(f)

# Each document has:
# - id: Unique identifier
# - text: Main content for embedding
# - embedding_text: Title + content for better vectors
# - All metadata fields for filtering/context
```

### For LangChain

```python
from langchain.schema import Document

documents = [
    Document(
        page_content=doc['text'],
        metadata={k: v for k, v in doc.items() if k != 'text'}
    )
    for doc in rag_documents
]
```

## 📈 Analysis Results

For the Northwind PDF, the chunker created:
- **41 chunks** total
- **Average size**: ~1,000 characters (160 words)
- **Size range**: 126-1,471 characters
- **Coverage**: All 11 pages
- **Section levels**: 1-4 (hierarchical structure preserved)

## 🔍 Search Capabilities

The system includes keyword search for content discovery:

```python
from pdf_section_chunker import search_chunks_by_keyword

# Find chunks about customers
customer_chunks = search_chunks_by_keyword(chunks, "customer")
print(f"Found {len(customer_chunks)} chunks about customers")
```

## 🎛️ Configuration Options

### Chunking Parameters
- `min_chunk_size`: Minimum characters (default: 100)
- `max_chunk_size`: Maximum characters (default: 2000)

### Header Detection Tuning
The system automatically adjusts header detection based on:
- Font size relative to document average
- Bold/italic formatting
- Numbering patterns (1., 2., etc.)
- All-caps text patterns

## 📝 Export Formats

### JSON (Vector Database Ready)
```bash
northwind_rag_ready.json    # Full metadata, ready for vector DB
```

### CSV (Analysis)
```bash
northwind_chunks_analysis.csv    # Tabular data for analysis
```

### Text (Human Readable)
```bash
northwind_chunks_readable.txt    # Formatted for human review
```

## 🔌 Vector Database Examples

### Qdrant Integration
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create collection and insert documents
# (See rag_chunking_example.py for full code)
```

### FAISS Integration
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("northwind_vectorstore")
```

## 🛠️ Dependencies

Required packages (already in your pyproject.toml):
- `pymupdf` - PDF processing
- `pandas` - Data analysis
- `sentence-transformers` - Embeddings (optional)
- `qdrant-client` - Vector database (optional)

## 🚦 Running the Examples

```bash
# Run the main demonstration
python rag_chunking_example.py

# Run just the chunker
python pdf_section_chunker.py
```

## 📋 Use Cases

This chunking approach is ideal for:
- **Technical Documentation**: Preserves section structure
- **Research Papers**: Maintains academic organization
- **Business Reports**: Keeps logical flow intact
- **Manuals**: Preserves procedural sequences
- **Any Structured PDFs**: Where section boundaries matter

## 🎯 Next Steps

1. **Load into Vector Database**: Use `northwind_rag_ready.json`
2. **Generate Embeddings**: Process with your preferred embedding model
3. **Build RAG Pipeline**: Integrate with your LLM application
4. **Customize for Your PDFs**: Adjust parameters for different document types

The chunking function is designed to be adaptable to different PDF types while maintaining the intelligent section-based approach that makes RAG more effective. 