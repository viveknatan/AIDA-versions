"""
Test PDF Section Chunker with Any Document
==========================================

This script demonstrates how to use the PDF chunker with different types of documents
and how to tune parameters for optimal results. Now includes tiktoken support!
"""

from pdf_section_chunker import PDFSectionChunker, DocumentChunk, get_encoding_info, TIKTOKEN_AVAILABLE
import os
from typing import List

def analyze_pdf_structure(pdf_path: str):
    """
    Analyze a PDF's structure to understand its formatting before chunking.
    This helps determine optimal parameters.
    """
    print(f"\n🔍 ANALYZING PDF STRUCTURE: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Create a chunker for analysis
    chunker = PDFSectionChunker(use_tokens=False)  # Use chars for analysis
    
    try:
        # Extract text with formatting
        text_blocks = chunker.extract_text_with_formatting(pdf_path)
        
        # Analyze font sizes
        font_sizes = [block["size"] for block in text_blocks if block["text"].strip()]
        font_styles = [block["font"] for block in text_blocks if block["text"].strip()]
        
        if font_sizes:
            avg_font = sum(font_sizes) / len(font_sizes)
            min_font = min(font_sizes)
            max_font = max(font_sizes)
            
            print(f"📊 Font Analysis:")
            print(f"   Average font size: {avg_font:.1f}")
            print(f"   Font range: {min_font:.1f} - {max_font:.1f}")
            print(f"   Unique font sizes: {sorted(set(font_sizes))}")
            print(f"   Unique fonts: {len(set(font_styles))} different fonts")
        
        # Identify potential headers
        headers = chunker.identify_headers(text_blocks)
        print(f"\n📋 Header Detection:")
        print(f"   Found {len(headers)} potential headers")
        
        if headers:
            print("   Header samples:")
            for i, header in enumerate(headers[:5]):  # Show first 5
                print(f"     {i+1}. Level {header['level']}: '{header['text'][:50]}...'")
        
        # Analyze pages
        pages = set(block["page"] for block in text_blocks)
        print(f"\n📖 Document Info:")
        print(f"   Total pages: {len(pages)}")
        print(f"   Total text blocks: {len(text_blocks)}")
        
        # Token analysis if available
        if TIKTOKEN_AVAILABLE:
            # Sample some text to estimate token ratios
            sample_text = " ".join([block["text"] for block in text_blocks[:100] if block["text"].strip()])
            if sample_text:
                token_chunker = PDFSectionChunker(use_tokens=True)
                token_count = token_chunker.count_tokens(sample_text)
                char_count = len(sample_text)
                ratio = char_count / token_count if token_count > 0 else 4
                
                print(f"\n🔢 Token Analysis (sample):")
                print(f"   Sample chars: {char_count}")
                print(f"   Sample tokens: {token_count}")
                print(f"   Chars per token: {ratio:.1f}")
        
        return {
            "avg_font_size": avg_font if font_sizes else 12,
            "font_range": (min_font, max_font) if font_sizes else (10, 14),
            "header_count": len(headers),
            "page_count": len(pages)
        }
        
    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")
        return None

def test_chunking_with_different_params(pdf_path: str, analysis_result: dict):
    """
    Test different chunking parameters to find optimal settings.
    Now tests both character and token-based chunking.
    """
    print(f"\n🧪 TESTING DIFFERENT CHUNKING PARAMETERS")
    
    # Test configurations
    test_configs = []
    
    # Character-based configs
    test_configs.extend([
        {"name": "Char-Conservative", "use_tokens": False, "min_size": 200, "max_size": 2000},
        {"name": "Char-Balanced", "use_tokens": False, "min_size": 100, "max_size": 1500},
        {"name": "Char-Fine", "use_tokens": False, "min_size": 50, "max_size": 1000},
    ])
    
    # Token-based configs (if available)
    if TIKTOKEN_AVAILABLE:
        test_configs.extend([
            {"name": "Token-Conservative", "use_tokens": True, "min_size": 100, "max_size": 500},
            {"name": "Token-Balanced", "use_tokens": True, "min_size": 50, "max_size": 300},
            {"name": "Token-Fine", "use_tokens": True, "min_size": 25, "max_size": 200},
        ])
    
    results = []
    
    for config in test_configs:
        try:
            if config["use_tokens"]:
                chunker = PDFSectionChunker(
                    min_token_size=config["min_size"],
                    max_token_size=config["max_size"],
                    use_tokens=True
                )
            else:
                chunker = PDFSectionChunker(
                    min_chunk_size=config["min_size"],
                    max_chunk_size=config["max_size"],
                    use_tokens=False
                )
            
            chunks = chunker.chunk_by_sections(pdf_path)
            
            if chunks:
                if config["use_tokens"]:
                    sizes = [chunk.metadata.get('token_count', 0) for chunk in chunks]
                    avg_size = sum(sizes) / len(sizes)
                    unit = "tokens"
                else:
                    sizes = [chunk.metadata.get('char_count', len(chunk.content)) for chunk in chunks]
                    avg_size = sum(sizes) / len(sizes)
                    unit = "chars"
                
                result = {
                    "config": config["name"],
                    "chunk_count": len(chunks),
                    "avg_size": int(avg_size),
                    "unit": unit,
                    "size_range": (min(sizes), max(sizes))
                }
                results.append(result)
                
                print(f"   {config['name']:15} → {len(chunks):2d} chunks, avg {int(avg_size):4d} {unit}")
        
        except Exception as e:
            print(f"   {config['name']:15} → Error: {e}")
    
    return results

def chunk_any_pdf(pdf_path: str, 
                  use_tokens: bool = None,
                  min_chunk_size: int = None, 
                  max_chunk_size: int = None,
                  min_token_size: int = None,
                  max_token_size: int = None,
                  encoding_name: str = "cl100k_base",
                  show_samples: bool = True) -> List[DocumentChunk]:
    """
    Chunk any PDF with optional parameter customization.
    
    Args:
        pdf_path: Path to PDF file
        use_tokens: Whether to use token-based chunking (auto-detected if None)
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        min_token_size: Minimum chunk size in tokens
        max_token_size: Maximum chunk size in tokens
        encoding_name: Tiktoken encoding to use
        show_samples: Whether to display sample chunks
    
    Returns:
        List of DocumentChunk objects
    """
    print(f"\n🔄 CHUNKING PDF: {os.path.basename(pdf_path)}")
    
    # Analyze structure first
    analysis = analyze_pdf_structure(pdf_path)
    if not analysis:
        return []
    
    # Auto-detect token usage if not specified
    if use_tokens is None:
        use_tokens = TIKTOKEN_AVAILABLE
    
    # Auto-detect parameters if not provided
    if use_tokens:
        if min_token_size is None or max_token_size is None:
            if analysis["header_count"] > 20:  # Lots of headers = fine-grained
                min_token_size = min_token_size or 25
                max_token_size = max_token_size or 200
            elif analysis["header_count"] < 5:  # Few headers = larger chunks
                min_token_size = min_token_size or 100
                max_token_size = max_token_size or 500
            else:  # Balanced
                min_token_size = min_token_size or 50
                max_token_size = max_token_size or 300
        
        chunker = PDFSectionChunker(
            min_token_size=min_token_size,
            max_token_size=max_token_size,
            use_tokens=True,
            encoding_name=encoding_name
        )
        print(f"📝 Using TOKEN-based chunking: min={min_token_size}, max={max_token_size} tokens")
        
    else:
        if min_chunk_size is None or max_chunk_size is None:
            if analysis["header_count"] > 20:  # Lots of headers = fine-grained
                min_chunk_size = min_chunk_size or 50
                max_chunk_size = max_chunk_size or 1000
            elif analysis["header_count"] < 5:  # Few headers = larger chunks
                min_chunk_size = min_chunk_size or 200
                max_chunk_size = max_chunk_size or 2500
            else:  # Balanced
                min_chunk_size = min_chunk_size or 100
                max_chunk_size = max_chunk_size or 1500
        
        chunker = PDFSectionChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            use_tokens=False
        )
        print(f"📝 Using CHARACTER-based chunking: min={min_chunk_size}, max={max_chunk_size} chars")
    
    try:
        chunks = chunker.chunk_by_sections(pdf_path)
        
        if chunks:
            print(f"✅ Created {len(chunks)} chunks")
            
            # Statistics
            char_counts = [chunk.metadata.get('char_count', len(chunk.content)) for chunk in chunks]
            token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
            
            print(f"📊 Chunk Statistics:")
            print(f"   Character count - Min: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts) / len(char_counts):.0f}")
            
            if any(token_counts):
                print(f"   Token count - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts) / len(token_counts):.0f}")
                print(f"   Chunking method: {chunks[0].metadata.get('chunking_method', 'unknown')}")
                print(f"   Encoding: {chunks[0].metadata.get('encoding', 'N/A')}")
            
            # Show samples
            if show_samples and len(chunks) > 0:
                print(f"\n📋 Sample Chunks:")
                for i, chunk in enumerate(chunks[:3]):
                    char_count = chunk.metadata.get('char_count', len(chunk.content))
                    token_count = chunk.metadata.get('token_count', 'N/A')
                    
                    print(f"\n   Chunk {i+1}: {chunk.title}")
                    print(f"   Page {chunk.page_number} | Level {chunk.section_level} | {char_count} chars | {token_count} tokens")
                    preview = chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                    print(f"   Preview: {preview}")
            
            return chunks
        else:
            print("❌ No chunks created - check PDF structure")
            return []
            
    except Exception as e:
        print(f"❌ Error chunking PDF: {e}")
        return []

def test_with_sample_pdfs():
    """Test the chunker with available PDF files."""
    
    # Look for PDFs in common locations
    potential_pdfs = [
        "data/Northwind_Traders_Database_Overview.pdf",
        "data/Northwind_Database_Schema.pdf",
        # Add other PDFs you might have
    ]
    
    print("🚀 TESTING PDF SECTION CHUNKER WITH TIKTOKEN SUPPORT")
    
    # Show encoding information
    if TIKTOKEN_AVAILABLE:
        print("\n📋 Available Tiktoken Encodings:")
        for encoding, models in get_encoding_info().items():
            print(f"   {encoding}: {models}")
    else:
        print("\n⚠️ Tiktoken not available - using character-based chunking only")
    
    found_pdfs = [pdf for pdf in potential_pdfs if os.path.exists(pdf)]
    
    if not found_pdfs:
        print("❌ No test PDFs found. Please ensure PDFs are in the data/ directory.")
        print("   Available files:")
        if os.path.exists("data"):
            for file in os.listdir("data"):
                if file.endswith(".pdf"):
                    print(f"   - data/{file}")
        return
    
    for pdf_path in found_pdfs:
        print("\n" + "="*80)
        
        # Test token-based chunking if available
        if TIKTOKEN_AVAILABLE:
            chunks = chunk_any_pdf(pdf_path, use_tokens=True, show_samples=True)
        else:
            chunks = chunk_any_pdf(pdf_path, use_tokens=False, show_samples=True)
        
        if chunks:
            # Test different parameters
            test_chunking_with_different_params(pdf_path, {"header_count": len(chunks)})

def main():
    """Main function to demonstrate PDF chunking capabilities."""
    test_with_sample_pdfs()
    
    print(f"\n" + "="*80)
    print("💡 USAGE TIPS FOR TOKEN-BASED CHUNKING:")
    print("""
    🎯 Token vs Character Chunking:
       - TOKENS: More accurate for LLM applications (recommended)
       - CHARACTERS: Simpler, works without tiktoken
    
    📏 Recommended Token Sizes:
       - GPT-4/GPT-3.5 context: 25-500 tokens per chunk
       - Embedding models: 50-300 tokens per chunk
       - Very detailed docs: 100-500 tokens per chunk
    
    🔧 Encoding Selection:
       - cl100k_base: GPT-4, GPT-3.5-turbo (most common)
       - p50k_base: text-davinci-002, text-davinci-003
       - r50k_base: Older GPT-3 models
    
    📚 Document-Specific Tips:
       - Academic Papers: 100-400 tokens (preserve context)
       - Technical Docs: 50-200 tokens (granular sections)
       - Business Reports: 100-300 tokens (balanced)
       - Reference Material: 200-500 tokens (comprehensive)
    
    ⚡ Performance:
       - Token counting is slower than character counting
       - Cache token counts in metadata for reuse
       - Use character-based for quick prototyping
    """)

if __name__ == "__main__":
    main() 