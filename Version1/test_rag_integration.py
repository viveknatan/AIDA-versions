#!/usr/bin/env python3
"""
Test script for RAG integration with the AI Data Analyst Agent
"""

from agent import DataAnalystAgent
from config import Configuration

def test_rag_integration():
    """Test the RAG integration with various types of questions."""
    
    print("🚀 Testing RAG Integration with AI Data Analyst Agent")
    print("=" * 60)
    
    # Initialize agent
    print("🔄 Initializing agent...")
    agent = DataAnalystAgent(Configuration())
    
    # Test questions that should trigger different combinations
    test_questions = [
        {
            "question": "What are the top 5 customers by revenue?",
            "description": "Database-focused question"
        },
        {
            "question": "What can you tell me about the Northwind business model and database structure?",
            "description": "RAG-focused question"
        },
        {
            "question": "Show me the top customers and explain how the Northwind database is designed",
            "description": "Combined database + RAG question"
        }
    ]
    
    for i, test_case in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"Question: {test_case['question']}")
        print("-" * 60)
        
        try:
            # Process the question
            result = agent.process_question(test_case['question'])
            
            # Show results
            print("📊 Results:")
            
            # Database analysis
            if result.get('analysis'):
                print(f"✅ Database Analysis: {len(result['analysis'])} characters")
            
            # RAG results
            if result.get('rag_result'):
                rag = result['rag_result']
                print(f"✅ RAG Analysis: Confidence {rag.confidence:.2f}, Sources: {rag.source_types}")
                print(f"   Retrieved {len(rag.retrieved_docs)} documents")
            
            # Combined analysis
            if result.get('combined_analysis'):
                print(f"✅ Combined Analysis: {len(result['combined_analysis'])} characters")
                print("\n📝 Combined Analysis Preview:")
                preview = result['combined_analysis'][:300] + "..." if len(result['combined_analysis']) > 300 else result['combined_analysis']
                print(preview)
            
            # Error handling
            if result.get('error'):
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n✅ RAG Integration Testing Complete!")

if __name__ == "__main__":
    test_rag_integration() 