"""
RAG Tool for AI Data Analyst Agent
Combines PDF and database document processing with vector search for enhanced question answering.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# LangChain imports
from langchain.schema import Document, BaseRetriever
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from operator import itemgetter
from pydantic import Field

# Local imports
from config import Configuration
from RAG.pdf_section_chunker import chunk_northwind_pdf

@dataclass
class RAGResult:
    """Result from RAG tool execution"""
    answer: str
    sources: List[str]
    retrieved_docs: List[Dict[str, Any]]
    confidence: float
    source_types: List[str]  # ['pdf', 'database', 'hybrid']

class ScoreFilteredRetriever(BaseRetriever):
    """Simple retriever that filters results by similarity score."""
    
    vectorstore: Any = Field()
    score_threshold: float = Field(default=0.3)
    k: int = Field(default=8)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.k)
        
        # Filter by score threshold and return with metadata about scores
        filtered_docs = []
        for doc, score in docs_with_scores:
            if score >= self.score_threshold:
                # Add score to metadata for later use
                doc.metadata['retrieval_score'] = score
                filtered_docs.append(doc)
        
        return filtered_docs

class RAGTool:
    """
    RAG Tool that combines PDF and database documents for enhanced question answering.
    Integrates with the AI Data Analyst agent as a complementary knowledge source.
    """
    
    def __init__(self, config: Configuration):
        self.config = config
        self.embedding_model = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.llm = None
        
        # Document tracking
        self.pdf_documents = []
        self.db_documents = []
        self.all_documents = []
        
        # Initialize the RAG system
        self._initialize_rag_system()
    
    def _initialize_rag_system(self):
        """Initialize the complete RAG system with PDF and database documents."""
        try:
            print("🔄 Initializing RAG system...")
            
            # Initialize OpenAI components
            self.embedding_model = OpenAIEmbeddings(
                api_key=self.config.OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            
            self.llm = ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.llm_model,
                temperature=0.1
            )
            
            # Load and process documents
            self._load_pdf_documents()
            self._load_database_documents()
            self._create_vector_store()
            self._create_rag_chain()
            
            print("✅ RAG system initialized successfully!")
            
        except Exception as e:
            print(f"⚠️ RAG system initialization failed: {e}")
            # Create a minimal fallback system
            self._create_fallback_system()
    
    def _load_pdf_documents(self):
        """Load and process PDF documents using intelligent chunking."""
        try:
            pdf_path = "data/Northwind_Traders_Database_Overview.pdf"
            
            if not os.path.exists(pdf_path):
                print(f"⚠️ PDF not found at {pdf_path}")
                return
            
            print("📄 Processing PDF documents...")
            
            # Use the existing PDF chunker
            chunks = chunk_northwind_pdf(
                pdf_path=pdf_path,
                use_tokens=True,
                encoding_name="cl100k_base"
            )
            
            # Convert to LangChain Documents
            self.pdf_documents = self._convert_chunks_to_langchain_docs(chunks)
            
            print(f"✅ Loaded {len(self.pdf_documents)} PDF document chunks")
            
        except Exception as e:
            print(f"⚠️ PDF processing failed: {e}")
            self.pdf_documents = []
    
    def _load_database_documents(self):
        """Load database-generated business documents."""
        try:
            print("🗃️ Generating database business documents...")
            
            # Use the existing database manager to avoid duplicate connection logic
            try:
                from database_test import DatabaseManager
                db_manager = DatabaseManager()
                
                # Check if we have a successful database connection
                if db_manager.get_schema_name() == "northwind":
                    # Use the schema info to generate summary documents
                    schema_info = db_manager.get_schema_info()
                    
                    # Create comprehensive business documents from schema and sample data
                    business_docs = self._generate_docs_from_schema(schema_info, db_manager)
                    
                    # Convert to LangChain format
                    self.db_documents = self._convert_business_docs_to_langchain(business_docs)
                    
                    print(f"✅ Generated {len(self.db_documents)} database business documents")
                else:
                    print("⚠️ Database not connected to Northwind schema, skipping database documents")
                    self.db_documents = []
                    
            except Exception as e:
                print(f"⚠️ Database document generation failed: {e}")
                self.db_documents = []
                
        except Exception as e:
            print(f"⚠️ Database processing failed: {e}")
            self.db_documents = []
    
    def _generate_docs_from_schema(self, schema_info: Dict[str, Any], db_manager) -> List[str]:
        """Generate business documents from database schema and sample data."""
        documents = []
        
        try:
            # Document 1: Database Schema Overview
            schema_doc = "NORTHWIND DATABASE SCHEMA OVERVIEW:\n\n"
            schema_doc += f"The Northwind database contains {len(schema_info)} main tables:\n\n"
            
            for table_name, table_info in schema_info.items():
                schema_doc += f"**{table_name.upper()} TABLE:**\n"
                schema_doc += f"- Columns: {len(table_info['columns'])}\n"
                
                # List key columns
                key_columns = []
                for col in table_info['columns']:
                    if col.get('primary_key'):
                        key_columns.append(f"{col['name']} (PK)")
                    elif not col['nullable']:
                        key_columns.append(f"{col['name']} (Required)")
                
                if key_columns:
                    schema_doc += f"- Key columns: {', '.join(key_columns[:5])}\n"
                
                # Foreign key relationships
                if table_info['foreign_keys']:
                    schema_doc += f"- Relationships: {len(table_info['foreign_keys'])} foreign keys\n"
                
                schema_doc += "\n"
            
            documents.append(schema_doc)
            
            # Document 2: Sample Data Analysis for key tables
            for table_name in ['customers', 'products', 'orders']:
                if table_name in schema_info:
                    try:
                        sample_data = db_manager.get_sample_data(table_name, limit=10)
                        if not sample_data.empty:
                            sample_doc = f"SAMPLE {table_name.upper()} DATA ANALYSIS:\n\n"
                            sample_doc += f"The {table_name} table contains the following structure and sample data:\n\n"
                            
                            # Add column information
                            sample_doc += "**Column Information:**\n"
                            for col in schema_info[table_name]['columns']:
                                sample_doc += f"- {col['name']}: {col['type']}\n"
                            
                            sample_doc += f"\n**Sample Records (showing {len(sample_data)} of many):**\n"
                            sample_doc += sample_data.head().to_string(index=False)
                            sample_doc += "\n\n"
                            
                            documents.append(sample_doc)
                    except Exception as e:
                        print(f"⚠️ Could not get sample data for {table_name}: {e}")
            
            print(f"📊 Generated {len(documents)} schema-based documents")
            return documents
            
        except Exception as e:
            print(f"⚠️ Schema document generation failed: {e}")
            return []
    
    def _convert_chunks_to_langchain_docs(self, chunks) -> List[Document]:
        """Convert PDF chunks to LangChain Document format."""
        langchain_docs = []
        
        for chunk in chunks:
            metadata = {
                "source": chunk.metadata.get('source', 'Northwind_Traders_Database_Overview.pdf'),
                "page": chunk.page_number,
                "section_title": chunk.title,
                "section_level": chunk.section_level,
                "char_count": chunk.metadata.get('char_count', len(chunk.content)),
                "token_count": chunk.metadata.get('token_count', 0),
                "chunking_method": chunk.metadata.get('chunking_method', 'section'),
                "chunk_type": "pdf_section",
                "content_type": "documentation"
            }
            
            doc = Document(
                page_content=chunk.content,
                metadata=metadata
            )
            langchain_docs.append(doc)
        
        return langchain_docs
    
    def _convert_business_docs_to_langchain(self, business_docs) -> List[Document]:
        """Convert database business documents to LangChain Document format."""
        langchain_docs = []
        
        doc_types = [
            "customer_analysis", "customer_behavior", "product_catalog", 
            "supplier_analysis", "employee_performance", "shipping_logistics",
            "financial_performance", "business_intelligence", "operational_efficiency"
        ]
        
        for i, doc in enumerate(business_docs):
            doc_type = doc_types[i] if i < len(doc_types) else f"business_doc_{i}"
            
            metadata = {
                "source": f"northwind_database_{doc_type}",
                "type": "business_analysis",
                "document_id": i,
                "section_title": doc_type.replace('_', ' ').title(),
                "char_count": len(doc),
                "chunk_type": "database_generated",
                "content_type": "business_data"
            }
            
            langchain_doc = Document(
                page_content=doc,
                metadata=metadata
            )
            langchain_docs.append(langchain_doc)
        
        return langchain_docs
    
    def _smart_split_documents(self, all_documents: List[Document]) -> List[Document]:
        """Split large database documents while preserving PDF chunks."""
        final_documents = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
        )
        
        for doc in all_documents:
            # Split large database documents
            if (doc.metadata.get('content_type') == 'business_data' and 
                len(doc.page_content) > 2000):
                splits = text_splitter.split_documents([doc])
                # Update metadata for splits
                for i, split in enumerate(splits):
                    split.metadata.update({
                        'split_index': i,
                        'original_length': len(doc.page_content),
                        'is_split': True
                    })
                final_documents.extend(splits)
            else:
                # Keep PDF chunks and smaller DB docs as-is
                final_documents.append(doc)
        
        return final_documents
    
    def _create_vector_store(self):
        """Create the vector store with all documents."""
        try:
            # Combine all documents
            self.all_documents = self.pdf_documents + self.db_documents
            
            if not self.all_documents:
                print("⚠️ No documents available for vector store")
                return
            
            # Apply smart splitting
            processed_documents = self._smart_split_documents(self.all_documents)
            
            print(f"📊 Document processing: {len(self.all_documents)} → {len(processed_documents)} chunks")
            
            # Create vector store
            self.vectorstore = Qdrant.from_documents(
                processed_documents,
                self.embedding_model,
                location=":memory:",
                collection_name="northwind_comprehensive_rag",
            )
            
            # Create retriever
            self.retriever = ScoreFilteredRetriever(
                vectorstore=self.vectorstore,
                score_threshold=0.3,
                k=8
            )
            
            print(f"✅ Created vector store with {len(processed_documents)} documents")
            
        except Exception as e:
            print(f"⚠️ Vector store creation failed: {e}")
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering."""
        if not self.retriever or not self.llm:
            return
        
        try:
            # Enhanced RAG prompt
            RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant with access to comprehensive Northwind Traders information from both:
1. Database Overview PDF documentation (structural and design information)
2. Live database analysis reports (current business data and performance metrics)

Instructions:
- Use the provided context to answer the question thoroughly
- If you can't answer based on the context, clearly state that you don't know
- When possible, distinguish between structural/design information and actual business performance data
- Cite specific sources when providing information (mention if from PDF docs or database analysis)
- If the question requires current data analysis that isn't in the context, mention that live database querying might provide more current information

Answer:
"""
            
            rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
            
            # Create the chain
            self.rag_chain = (
                {"context": itemgetter("question") | self.retriever, "question": itemgetter("question")}
                | rag_prompt 
                | self.llm 
                | StrOutputParser()
            )
            
            print("✅ Created RAG chain")
            
        except Exception as e:
            print(f"⚠️ RAG chain creation failed: {e}")
    
    def _create_fallback_system(self):
        """Create a minimal fallback system when full RAG initialization fails."""
        print("🔄 Creating fallback RAG system...")
        
        try:
            self.llm = ChatOpenAI(
                api_key=self.config.OPENAI_API_KEY,
                model=self.config.llm_model,
                temperature=0.1
            )
            print("✅ Fallback system ready (LLM only)")
        except Exception as e:
            print(f"⚠️ Even fallback system failed: {e}")
    
    def query(self, question: str) -> RAGResult:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to answer using RAG
            
        Returns:
            RAGResult with answer and metadata
        """
        try:
            if not self.rag_chain:
                # Fallback to direct LLM if RAG chain not available
                if self.llm:
                    fallback_answer = self.llm.invoke(f"Answer this question about Northwind Traders business: {question}")
                    return RAGResult(
                        answer=fallback_answer.content if hasattr(fallback_answer, 'content') else str(fallback_answer),
                        sources=["llm_fallback"],
                        retrieved_docs=[],
                        confidence=0.5,
                        source_types=["llm_only"]
                    )
                else:
                    return RAGResult(
                        answer="RAG system not available. Please ensure proper configuration.",
                        sources=[],
                        retrieved_docs=[],
                        confidence=0.0,
                        source_types=[]
                    )
            
            # Get retrieved documents first for analysis
            retrieved_docs = self.retriever._get_relevant_documents(question)
            
            # Run the RAG chain
            answer = self.rag_chain.invoke({"question": question})
            
            # Analyze sources
            sources = []
            source_types = []
            doc_metadata = []
            
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'unknown')
                sources.append(source)
                
                content_type = doc.metadata.get('content_type', 'unknown')
                if content_type == 'documentation':
                    source_types.append('pdf')
                elif content_type == 'business_data':
                    source_types.append('database')
                else:
                    source_types.append('unknown')
                
                doc_metadata.append({
                    'source': source,
                    'content_type': content_type,
                    'score': doc.metadata.get('retrieval_score', 0.0),
                    'section': doc.metadata.get('section_title', ''),
                    'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })
            
            # Calculate confidence based on retrieval scores and source diversity
            if retrieved_docs:
                avg_score = sum(doc.metadata.get('retrieval_score', 0.0) for doc in retrieved_docs) / len(retrieved_docs)
                source_diversity = len(set(source_types)) / max(len(source_types), 1)
                confidence = min(avg_score * source_diversity, 1.0)
            else:
                confidence = 0.3  # Low confidence if no docs retrieved
            
            return RAGResult(
                answer=answer,
                sources=list(set(sources)),  # Remove duplicates
                retrieved_docs=doc_metadata,
                confidence=confidence,
                source_types=list(set(source_types))
            )
            
        except Exception as e:
            print(f"⚠️ RAG query failed: {e}")
            return RAGResult(
                answer=f"RAG query failed: {str(e)}",
                sources=[],
                retrieved_docs=[],
                confidence=0.0,
                source_types=[]
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the RAG system."""
        return {
            "pdf_documents_loaded": len(self.pdf_documents),
            "database_documents_loaded": len(self.db_documents),
            "total_documents": len(self.all_documents),
            "vector_store_ready": self.vectorstore is not None,
            "rag_chain_ready": self.rag_chain is not None,
            "embedding_model_ready": self.embedding_model is not None,
            "llm_ready": self.llm is not None
        } 