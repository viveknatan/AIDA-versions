"""
RAG Chunking Example for Northwind PDF
======================================

This script demonstrates how to use the PDF section-based chunking function
for Retrieval-Augmented Generation (RAG) applications with tiktoken support.
"""

from pdf_section_chunker import (PDFSectionChunker, DocumentChunk, chunk_northwind_pdf, 
                                save_chunks_to_text, get_encoding_info, TIKTOKEN_AVAILABLE)
import json
import pandas as pd
from typing import List, Dict

def analyze_chunks(chunks: List[DocumentChunk]):
    """Analyze the generated chunks and print statistics."""
    print(f"\n=== CHUNK ANALYSIS ===")
    print(f"Total chunks: {len(chunks)}")
    
    # Calculate statistics
    char_counts = [chunk.metadata.get('char_count', len(chunk.content)) for chunk in chunks]
    token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
    word_counts = [len(chunk.content.split()) for chunk in chunks]
    
    print(f"Character count - Min: {min(char_counts)}, Max: {max(char_counts)}, Avg: {sum(char_counts)//len(char_counts)}")
    print(f"Word count - Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)//len(word_counts)}")
    
    if any(token_counts):
        print(f"Token count - Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts)//len(token_counts)}")
        print(f"Chunking method: {chunks[0].metadata.get('chunking_method', 'unknown')}")
        print(f"Encoding: {chunks[0].metadata.get('encoding', 'N/A')}")
        
        # Token efficiency metrics
        if min(token_counts) > 0:
            chars_per_token = [char_counts[i] / token_counts[i] for i in range(len(chunks)) if token_counts[i] > 0]
            avg_chars_per_token = sum(chars_per_token) / len(chars_per_token)
            print(f"Average chars per token: {avg_chars_per_token:.1f}")
    
    # Section levels
    levels = [chunk.section_level for chunk in chunks]
    print(f"Section levels: {sorted(set(levels))}")
    
    # Pages
    pages = [chunk.page_number for chunk in chunks]
    print(f"Pages: {min(pages)} to {max(pages)}")

def display_sample_chunks(chunks: List[DocumentChunk], num_samples: int = 3):
    """Display sample chunks for inspection."""
    print(f"\n=== SAMPLE CHUNKS (First {num_samples}) ===")
    
    for i, chunk in enumerate(chunks[:num_samples]):
        char_count = chunk.metadata.get('char_count', len(chunk.content))
        token_count = chunk.metadata.get('token_count', 'N/A')
        
        print(f"\nChunk {i+1}: {chunk.title}")
        print(f"Page: {chunk.page_number} | Level: {chunk.section_level} | {char_count} chars | {token_count} tokens")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 60)
        content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(content_preview)

def prepare_for_rag_database(chunks: List[DocumentChunk]) -> List[Dict]:
    """
    Prepare chunks for insertion into a RAG/vector database.
    
    Returns:
        List of dictionaries with all necessary fields for RAG
    """
    rag_documents = []
    
    for i, chunk in enumerate(chunks):
        doc = {
            # Unique identifier
            'id': f"northwind_chunk_{i+1:03d}",
            
            # Main content for vector embedding
            'text': chunk.content,
            'title': chunk.title,
            
            # Metadata for filtering and context
            'source_document': 'Northwind_Traders_Database_Overview.pdf',
            'page_number': chunk.page_number,
            'section_level': chunk.section_level,
            'char_count': chunk.metadata.get('char_count', len(chunk.content)),
            'token_count': chunk.metadata.get('token_count', 0),
            'word_count': len(chunk.content.split()),
            'chunk_type': chunk.metadata.get('chunk_type', 'section'),
            
            # Token-related metadata
            'chunking_method': chunk.metadata.get('chunking_method', 'unknown'),
            'encoding': chunk.metadata.get('encoding'),
            'chars_per_token': (chunk.metadata.get('char_count', len(chunk.content)) / 
                               chunk.metadata.get('token_count', 1) if chunk.metadata.get('token_count', 0) > 0 else None),
            
            # Additional fields for better retrieval
            'search_keywords': extract_keywords(chunk.content),
            'section_hierarchy': f"Level {chunk.section_level}: {chunk.title}",
            
            # Context for LLM
            'retrieval_context': f"This content is from section '{chunk.title}' "
                                f"(page {chunk.page_number}) of the Northwind Traders Database Overview.",
            
            # Combined text for embedding (title + content)
            'embedding_text': f"{chunk.title}. {chunk.content}",
            
            # Split information if applicable
            'is_split_chunk': chunk.metadata.get('is_split', False),
            'part_number': chunk.metadata.get('part_number')
        }
        rag_documents.append(doc)
    
    return rag_documents

def extract_keywords(text: str) -> List[str]:
    """Extract simple keywords from text (basic implementation)."""
    import re
    
    # Remove common words and extract meaningful terms
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an'}
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [word for word in words if word not in common_words]
    
    # Get most frequent words (simple approach)
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return top keywords
    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    return [word for word, freq in top_keywords]

def search_chunks_by_keyword(chunks: List[DocumentChunk], keyword: str) -> List[DocumentChunk]:
    """Simple keyword search in chunks."""
    keyword_lower = keyword.lower()
    matching_chunks = []
    
    for chunk in chunks:
        if (keyword_lower in chunk.title.lower() or 
            keyword_lower in chunk.content.lower()):
            matching_chunks.append(chunk)
    
    return matching_chunks

def export_chunks(rag_documents: List[Dict], chunks: List[DocumentChunk]):
    """Export chunks in various formats."""
    
    # Export as JSON for vector databases
    with open('northwind_rag_ready.json', 'w', encoding='utf-8') as f:
        json.dump(rag_documents, f, indent=2, ensure_ascii=False)
    print("✓ Exported to 'northwind_rag_ready.json' (for vector databases)")
    
    # Export as CSV for analysis
    df = pd.DataFrame(rag_documents)
    df.to_csv('northwind_chunks_analysis.csv', index=False)
    print("✓ Exported to 'northwind_chunks_analysis.csv' (for analysis)")
    
    # Export detailed text version
    save_chunks_to_text(chunks, 'northwind_chunks_readable.txt')
    print("✓ Exported to 'northwind_chunks_readable.txt' (human-readable)")

def demonstrate_token_optimization():
    """Show how to optimize chunking for different token limits."""
    
    print(f"\n=== TOKEN OPTIMIZATION EXAMPLES ===")
    
    if not TIKTOKEN_AVAILABLE:
        print("⚠️ Tiktoken not available - skipping token optimization")
        return
    
    # Test different token configurations
    configs = [
        {"name": "Embedding Model (OpenAI)", "max_tokens": 300, "desc": "Optimal for text-embedding-ada-002"},
        {"name": "GPT-3.5 Context", "max_tokens": 500, "desc": "Good balance for GPT-3.5-turbo"},
        {"name": "GPT-4 Detailed", "max_tokens": 800, "desc": "Rich context for GPT-4"},
        {"name": "Fine-grained Analysis", "max_tokens": 150, "desc": "Granular for detailed analysis"}
    ]
    
    for config in configs:
        print(f"\n🔧 {config['name']} ({config['desc']}):")
        
        chunker = PDFSectionChunker(
            min_token_size=25,
            max_token_size=config['max_tokens'],
            use_tokens=True
        )
        
        try:
            chunks = chunker.chunk_by_sections("data/Northwind_Traders_Database_Overview.pdf")
            
            if chunks:
                token_counts = [chunk.metadata.get('token_count', 0) for chunk in chunks]
                avg_tokens = sum(token_counts) / len(token_counts)
                
                print(f"   → {len(chunks)} chunks, avg {avg_tokens:.0f} tokens")
                print(f"   → Range: {min(token_counts)}-{max(token_counts)} tokens")
                
                # Check if any chunks exceed embedding limits
                over_limit = [t for t in token_counts if t > config['max_tokens']]
                if over_limit:
                    print(f"   ⚠️ {len(over_limit)} chunks exceed {config['max_tokens']} token limit")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def demonstrate_rag_integration():
    """Show how to integrate with common vector databases."""
    
    integration_examples = f"""
    
=== VECTOR DATABASE INTEGRATION WITH TOKENS ===

1. QDRANT INTEGRATION (Token-aware):
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Load token-optimized chunks
with open('northwind_rag_ready.json', 'r') as f:
    rag_documents = json.load(f)

# Initialize
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create collection with metadata
client.create_collection(
    collection_name="northwind_docs",
    vectors_config={{"size": 384, "distance": "Cosine"}}
)

# Insert documents with token metadata
for doc in rag_documents:
    # Filter by token count if needed
    if doc['token_count'] <= 300:  # Embedding model limit
        embedding = model.encode(doc['embedding_text'])
        client.upsert(
            collection_name="northwind_docs",
            points=[{{
                "id": doc['id'],
                "vector": embedding.tolist(),
                "payload": {{
                    **doc,
                    "optimized_for": "embedding_model"
                }}
            }}]
        )
```

2. LANGCHAIN WITH TOKEN FILTERING:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Filter documents by token count
max_tokens = 400  # For your specific model
filtered_docs = [doc for doc in rag_documents if doc['token_count'] <= max_tokens]

documents = [
    Document(
        page_content=doc['text'],
        metadata={{
            **{{k: v for k, v in doc.items() if k != 'text'}},
            "token_optimized": True
        }}
    )
    for doc in filtered_docs
]

# Create vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("northwind_vectorstore_token_optimized")
```

3. TOKEN-AWARE RETRIEVAL:
```python
def retrieve_with_token_budget(query: str, token_budget: int = 2000):
\"\"\"Retrieve chunks that fit within a token budget.\"\"\"
results = vectorstore.similarity_search(query, k=20)
    
selected_chunks = []
total_tokens = 0
    
for result in results:
    chunk_tokens = result.metadata.get('token_count', 0)
    if total_tokens + chunk_tokens <= token_budget:
        selected_chunks.append(result)
        total_tokens += chunk_tokens
    else:
        break
    
return selected_chunks, total_tokens
```
    """
    
    print(integration_examples)

def compare_chunking_methods():
    """Compare character vs token-based chunking."""
    
    print(f"\n=== CHUNKING METHOD COMPARISON ===")
    
    if not TIKTOKEN_AVAILABLE:
        print("⚠️ Tiktoken not available - skipping comparison")
        return
    
    # Character-based chunking
    print("📝 Character-based chunking:")
    char_chunker = PDFSectionChunker(
        min_chunk_size=100,
        max_chunk_size=1500,
        use_tokens=False
    )
    char_chunks = char_chunker.chunk_by_sections("data/Northwind_Traders_Database_Overview.pdf")
    
    # Token-based chunking
    print("🔢 Token-based chunking:")
    token_chunker = PDFSectionChunker(
        min_token_size=50,
        max_token_size=300,
        use_tokens=True
    )
    token_chunks = token_chunker.chunk_by_sections("data/Northwind_Traders_Database_Overview.pdf")
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"   Character-based: {len(char_chunks)} chunks")
    print(f"   Token-based: {len(token_chunks)} chunks")
    
    if char_chunks and token_chunks:
        char_sizes = [len(chunk.content) for chunk in char_chunks]
        token_sizes = [chunk.metadata.get('token_count', 0) for chunk in token_chunks]
        
        print(f"\n   Character method:")
        print(f"     → Avg size: {sum(char_sizes) / len(char_sizes):.0f} characters")
        print(f"     → Range: {min(char_sizes)}-{max(char_sizes)} characters")
        
        print(f"\n   Token method:")
        print(f"     → Avg size: {sum(token_sizes) / len(token_sizes):.0f} tokens")
        print(f"     → Range: {min(token_sizes)}-{max(token_sizes)} tokens")
        
        print(f"\n💡 Recommendation: Use token-based for LLM applications!")

def main():
    """Main function demonstrating the PDF chunking workflow with tiktoken."""
    
    print("🔄 Processing Northwind PDF for RAG with Tiktoken Support...")
    
    # Show encoding info
    if TIKTOKEN_AVAILABLE:
        print("\n📋 Available Tiktoken Encodings:")
        for encoding, models in get_encoding_info().items():
            print(f"   {encoding}: {models}")
    else:
        print("\n⚠️ Tiktoken not available - using character-based chunking")
    
    # 1. Chunk the PDF with token support
    try:
        chunks = chunk_northwind_pdf(use_tokens=TIKTOKEN_AVAILABLE)
        print(f"✅ Successfully created {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Error chunking PDF: {e}")
        return
    
    # 2. Analyze the chunks
    analyze_chunks(chunks)
    
    # 3. Display sample chunks
    display_sample_chunks(chunks)
    
    # 4. Compare chunking methods
    compare_chunking_methods()
    
    # 5. Show token optimization
    demonstrate_token_optimization()
    
    # 6. Prepare for RAG database
    print(f"\n🔄 Preparing chunks for RAG database...")
    rag_documents = prepare_for_rag_database(chunks)
    print(f"✅ Prepared {len(rag_documents)} documents for RAG")
    
    # 7. Export in various formats
    print(f"\n🔄 Exporting chunks...")
    export_chunks(rag_documents, chunks)
    
    # 8. Demonstrate search functionality
    print(f"\n=== SEARCH DEMONSTRATION ===")
    search_terms = ["customer", "order", "product", "table", "relationship"]
    
    for term in search_terms:
        results = search_chunks_by_keyword(chunks, term)
        if results:
            avg_tokens = sum(chunk.metadata.get('token_count', 0) for chunk in results) / len(results)
            print(f"'{term}': {len(results)} chunks found (avg {avg_tokens:.0f} tokens)")
        else:
            print(f"'{term}': No chunks found")
    
    # 9. Show integration examples
    demonstrate_rag_integration()
    
    print(f"\n✅ RAG chunking workflow with tiktoken completed!")
    print(f"📁 Check the exported files for your RAG implementation.")
    
    if TIKTOKEN_AVAILABLE:
        print(f"🎯 Your chunks are now token-optimized for LLM applications!")

if __name__ == "__main__":
    main() 