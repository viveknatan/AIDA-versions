import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os

# Try to import tiktoken, make it optional
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available. Install with: pip install tiktoken")

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    title: str
    content: str
    page_number: int
    section_level: int
    metadata: Dict
    
    def get_token_count(self, encoding_name: str = "cl100k_base") -> int:
        """Get token count for this chunk using tiktoken."""
        if not TIKTOKEN_AVAILABLE:
            # Rough approximation: 1 token ≈ 4 characters for English
            return len(self.content) // 4
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(self.content))
        except Exception:
            # Fallback to approximation
            return len(self.content) // 4

class PDFSectionChunker:
    """
    A class to chunk PDF documents based on sections for RAG applications.
    Now supports both character-based and token-based chunking.
    """
    
    def __init__(self, 
                 min_chunk_size: int = 100, 
                 max_chunk_size: int = 2000,
                 use_tokens: bool = True,
                 encoding_name: str = "cl100k_base",
                 min_token_size: int = 50,
                 max_token_size: int = 500):
        """
        Initialize the chunker with size constraints.
        
        Args:
            min_chunk_size: Minimum size for a chunk (in characters)
            max_chunk_size: Maximum size for a chunk (in characters)
            use_tokens: Whether to use token-based chunking (recommended)
            encoding_name: Tiktoken encoding to use (cl100k_base for GPT-4, GPT-3.5)
            min_token_size: Minimum tokens per chunk (if use_tokens=True)
            max_token_size: Maximum tokens per chunk (if use_tokens=True)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.use_tokens = use_tokens and TIKTOKEN_AVAILABLE
        self.encoding_name = encoding_name
        self.min_token_size = min_token_size
        self.max_token_size = max_token_size
        
        # Initialize tiktoken encoder if available
        self.encoder = None
        if self.use_tokens:
            try:
                self.encoder = tiktoken.get_encoding(encoding_name)
                print(f"✓ Using token-based chunking with {encoding_name} encoding")
            except Exception as e:
                print(f"Warning: Could not initialize tiktoken encoder: {e}")
                self.use_tokens = False
        
        if not self.use_tokens:
            print("ℹ️ Using character-based chunking")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation
            return len(text) // 4
    
    def get_chunk_size_limits(self) -> Tuple[int, int]:
        """Get the appropriate min/max limits based on chunking mode."""
        if self.use_tokens:
            return self.min_token_size, self.max_token_size
        else:
            return self.min_chunk_size, self.max_chunk_size
    
    def get_content_size(self, text: str) -> int:
        """Get size of content based on chunking mode."""
        if self.use_tokens:
            return self.count_tokens(text)
        else:
            return len(text)
    
    def extract_text_with_formatting(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF while preserving formatting information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text blocks with formatting info
        """
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                "text": span["text"],
                                "page": page_num + 1,
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],  # Bold, italic, etc.
                                "bbox": span["bbox"]  # Bounding box
                            })
        
        doc.close()
        return text_blocks
    
    def identify_headers(self, text_blocks: List[Dict]) -> List[Dict]:
        """
        Identify header blocks based on font size, style, and formatting.
        
        Args:
            text_blocks: List of text blocks with formatting information
            
        Returns:
            List of header blocks with hierarchical levels
        """
        # Analyze font sizes to determine header hierarchy
        font_sizes = [block["size"] for block in text_blocks if block["text"].strip()]
        avg_font_size = sum(font_sizes) / len(font_sizes)
        
        headers = []
        for block in text_blocks:
            text = block["text"].strip()
            if not text:
                continue
                
            # Check if this could be a header based on:
            # 1. Font size larger than average
            # 2. Bold formatting (flags & 2^4 = 16)
            # 3. Text patterns (numbers, capitalization)
            is_large_font = block["size"] > avg_font_size * 1.1
            is_bold = bool(block["flags"] & 16)
            is_numbered = bool(re.match(r'^\d+\.?\s+', text))
            is_caps = text.isupper() and len(text) < 100
            
            # Header detection logic
            if (is_large_font and is_bold) or is_numbered or (is_caps and len(text.split()) <= 10):
                # Determine header level based on font size
                if block["size"] > avg_font_size * 1.5:
                    level = 1
                elif block["size"] > avg_font_size * 1.3:
                    level = 2
                elif block["size"] > avg_font_size * 1.1:
                    level = 3
                else:
                    level = 4
                
                headers.append({
                    "text": text,
                    "page": block["page"],
                    "level": level,
                    "size": block["size"],
                    "bbox": block["bbox"]
                })
        
        return headers
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'Page \d+|\d+\s*$', '', text)
        # Clean up special characters
        text = text.replace('\x00', '').replace('\uf0b7', '•')
        return text.strip()
    
    def chunk_by_sections(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Main method to chunk PDF by sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text with formatting
        text_blocks = self.extract_text_with_formatting(pdf_path)
        
        # Identify headers
        headers = self.identify_headers(text_blocks)
        
        # Get size limits
        min_size, max_size = self.get_chunk_size_limits()
        
        # Group text blocks by sections
        chunks = []
        current_section = {"title": "Introduction", "content": "", "page": 1, "level": 0}
        
        # Create a mapping of page positions to headers
        header_positions = {}
        for header in headers:
            page = header["page"]
            if page not in header_positions:
                header_positions[page] = []
            header_positions[page].append(header)
        
        # Sort headers by page and position
        for page in header_positions:
            header_positions[page].sort(key=lambda x: x["bbox"][1])  # Sort by y-position
        
        current_header_idx = 0
        next_header = headers[0] if headers else None
        
        for block in text_blocks:
            text = block["text"].strip()
            if not text:
                continue
            
            # Check if this block is a header
            is_header = False
            if next_header and block["page"] == next_header["page"]:
                # Check if this text matches the next header
                if text == next_header["text"]:
                    is_header = True
                    
                    # Save previous section if it has content
                    if current_section["content"].strip():
                        chunk_content = self.clean_text(current_section["content"])
                        content_size = self.get_content_size(chunk_content)
                        
                        if content_size >= min_size:
                            # Create metadata with both char and token counts
                            metadata = {
                                "source": pdf_path,
                                "chunk_type": "section",
                                "char_count": len(chunk_content),
                                "token_count": self.count_tokens(chunk_content),
                                "encoding": self.encoding_name if self.use_tokens else None,
                                "chunking_method": "tokens" if self.use_tokens else "characters"
                            }
                            
                            chunks.append(DocumentChunk(
                                title=current_section["title"],
                                content=chunk_content,
                                page_number=current_section["page"],
                                section_level=current_section["level"],
                                metadata=metadata
                            ))
                    
                    # Start new section
                    current_section = {
                        "title": text,
                        "content": "",
                        "page": block["page"],
                        "level": next_header["level"]
                    }
                    
                    # Move to next header
                    current_header_idx += 1
                    next_header = headers[current_header_idx] if current_header_idx < len(headers) else None
            
            if not is_header:
                # Add to current section content
                current_section["content"] += " " + text
        
        # Don't forget the last section
        if current_section["content"].strip():
            chunk_content = self.clean_text(current_section["content"])
            content_size = self.get_content_size(chunk_content)
            
            if content_size >= min_size:
                metadata = {
                    "source": pdf_path,
                    "chunk_type": "section",
                    "char_count": len(chunk_content),
                    "token_count": self.count_tokens(chunk_content),
                    "encoding": self.encoding_name if self.use_tokens else None,
                    "chunking_method": "tokens" if self.use_tokens else "characters"
                }
                
                chunks.append(DocumentChunk(
                    title=current_section["title"],
                    content=chunk_content,
                    page_number=current_section["page"],
                    section_level=current_section["level"],
                    metadata=metadata
                ))
        
        return self._split_large_chunks(chunks)
    
    def _split_large_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Split chunks that are too large while preserving section boundaries.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with large chunks split
        """
        result = []
        min_size, max_size = self.get_chunk_size_limits()
        
        for chunk in chunks:
            chunk_size = self.get_content_size(chunk.content)
            
            if chunk_size <= max_size:
                result.append(chunk)
            else:
                # Split large chunk into smaller ones
                sentences = re.split(r'(?<=[.!?])\s+', chunk.content)
                current_chunk = ""
                part_num = 1
                
                for sentence in sentences:
                    test_content = current_chunk + sentence + " "
                    test_size = self.get_content_size(test_content)
                    
                    if test_size <= max_size:
                        current_chunk = test_content
                    else:
                        if current_chunk.strip():
                            # Create metadata for split chunk
                            metadata = {
                                **chunk.metadata,
                                "part_number": part_num,
                                "is_split": True,
                                "char_count": len(current_chunk.strip()),
                                "token_count": self.count_tokens(current_chunk.strip())
                            }
                            
                            result.append(DocumentChunk(
                                title=f"{chunk.title} (Part {part_num})",
                                content=current_chunk.strip(),
                                page_number=chunk.page_number,
                                section_level=chunk.section_level,
                                metadata=metadata
                            ))
                            part_num += 1
                        current_chunk = sentence + " "
                
                # Add the last part
                if current_chunk.strip():
                    final_title = f"{chunk.title} (Part {part_num})" if part_num > 1 else chunk.title
                    final_metadata = {
                        **chunk.metadata,
                        "part_number": part_num if part_num > 1 else None,
                        "is_split": part_num > 1,
                        "char_count": len(current_chunk.strip()),
                        "token_count": self.count_tokens(current_chunk.strip())
                    }
                    
                    result.append(DocumentChunk(
                        title=final_title,
                        content=current_chunk.strip(),
                        page_number=chunk.page_number,
                        section_level=chunk.section_level,
                        metadata=final_metadata
                    ))
        
        return result

def chunk_northwind_pdf(pdf_path: str = "data/Northwind_Traders_Database_Overview.pdf", 
                       use_tokens: bool = True,
                       encoding_name: str = "cl100k_base") -> List[DocumentChunk]:
    """
    Convenience function to chunk the Northwind PDF specifically.
    
    Args:
        pdf_path: Path to the Northwind PDF file
        use_tokens: Whether to use token-based chunking
        encoding_name: Tiktoken encoding to use
        
    Returns:
        List of DocumentChunk objects
    """
    if use_tokens and TIKTOKEN_AVAILABLE:
        chunker = PDFSectionChunker(
            min_token_size=25, 
            max_token_size=400,
            use_tokens=True,
            encoding_name=encoding_name
        )
    else:
        chunker = PDFSectionChunker(
            min_chunk_size=50, 
            max_chunk_size=1500,
            use_tokens=False
        )
    
    return chunker.chunk_by_sections(pdf_path)

def get_encoding_info() -> Dict[str, str]:
    """Get information about available tiktoken encodings."""
    if not TIKTOKEN_AVAILABLE:
        return {"error": "tiktoken not available"}
    
    encodings = {
        "cl100k_base": "GPT-4, GPT-3.5-turbo, text-embedding-ada-002",
        "p50k_base": "text-davinci-002, text-davinci-003",
        "r50k_base": "GPT-3 models (davinci, curie, babbage, ada)",
        "gpt2": "GPT-2 models"
    }
    
    return encodings

def save_chunks_to_text(chunks: List[DocumentChunk], output_path: str = "northwind_chunks.txt"):
    """
    Save chunks to a text file for inspection.
    
    Args:
        chunks: List of DocumentChunk objects
        output_path: Path to save the output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"PDF CHUNKING RESULTS\n")
        f.write(f"==================\n")
        
        if chunks:
            chunking_method = chunks[0].metadata.get('chunking_method', 'unknown')
            encoding = chunks[0].metadata.get('encoding', 'N/A')
            f.write(f"Chunking method: {chunking_method}\n")
            f.write(f"Encoding: {encoding}\n")
        
        f.write(f"Total chunks: {len(chunks)}\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            f.write(f"=== CHUNK {i}: {chunk.title} ===\n")
            f.write(f"Page: {chunk.page_number}, Level: {chunk.section_level}\n")
            f.write(f"Characters: {chunk.metadata.get('char_count', len(chunk.content))}\n")
            f.write(f"Tokens: {chunk.metadata.get('token_count', 'N/A')}\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write("-" * 50 + "\n")
            f.write(chunk.content)
            f.write("\n" + "=" * 80 + "\n\n")

if __name__ == "__main__":
    # Example usage
    try:
        print("🚀 PDF Section Chunker with Tiktoken Support")
        print("=" * 50)
        
        # Show encoding info
        if TIKTOKEN_AVAILABLE:
            print("\n📋 Available Encodings:")
            for encoding, models in get_encoding_info().items():
                print(f"  {encoding}: {models}")
        
        print(f"\n🔄 Processing Northwind PDF...")
        chunks = chunk_northwind_pdf()
        print(f"✅ Created {len(chunks)} chunks")
        
        if chunks:
            # Calculate statistics
            char_counts = [chunk.metadata.get('char_count', len(chunk.content)) for chunk in chunks]
            token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
            
            print(f"\n📊 Statistics:")
            print(f"   Character count - Min: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts)//len(char_counts)}")
            
            if any(token_counts):
                print(f"   Token count - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")
                print(f"   Chunking method: {chunks[0].metadata.get('chunking_method', 'unknown')}")
                print(f"   Encoding: {chunks[0].metadata.get('encoding', 'N/A')}")
        
        # Show first 3 chunks
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n--- Chunk {i}: {chunk.title} ---")
            print(f"Page: {chunk.page_number}, Characters: {chunk.metadata.get('char_count', len(chunk.content))}, Tokens: {chunk.metadata.get('token_count', 'N/A')}")
            print(f"Content preview: {chunk.content[:200]}...")
        
        # Save all chunks to file for inspection
        save_chunks_to_text(chunks)
        print(f"\n💾 All chunks saved to 'northwind_chunks.txt'")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}") 