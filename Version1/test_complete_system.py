#!/usr/bin/env python3
"""
Complete system test for RAG-integrated AI Data Analyst
"""

from agent import DataAnalystAgent
from config import Configuration

def test_complete_system():
    """Test the complete integrated system."""
    print("🚀 Testing Complete RAG-Integrated AI Data Analyst System")
    print("=" * 70)
    
    # Initialize agent
    print("🔄 Initializing agent...")
    agent = DataAnalystAgent(Configuration())
    
    # Test with a comprehensive question
    question = "What are the top 3 customers by revenue and how does this relate to the Northwind business model?"
    print(f"\n🔍 Testing Question:")
    print(f"'{question}'")
    print("-" * 70)
    
    try:
        result = agent.process_question(question)
        
        print("📊 RESULTS SUMMARY:")
        print(f"✅ Database analysis: {len(result.get('analysis', ''))} characters")
        
        if result.get('rag_result'):
            rag = result['rag_result']
            print(f"✅ RAG confidence: {rag.confidence:.2f}")
            print(f"✅ RAG sources: {rag.source_types}")
            print(f"✅ Retrieved documents: {len(rag.retrieved_docs)}")
        
        print(f"✅ Combined analysis: {len(result.get('combined_analysis', ''))} characters")
        print(f"✅ Has visualization: {result.get('visualization') is not None}")
        print(f"✅ Error status: {result.get('error', 'None')}")
        
        # Show key components
        print("\n🎯 COMBINED ANALYSIS PREVIEW:")
        combined = result.get('combined_analysis', '')
        preview = combined[:500] + '...' if len(combined) > 500 else combined
        print(preview)
        
        if result.get('sql_query'):
            print("\n🔍 GENERATED SQL:")
            print(result['sql_query'])
        
        # Show RAG details if available
        if result.get('rag_result') and result['rag_result'].confidence > 0.1:
            print(f"\n🧠 RAG SYSTEM DETAILS:")
            rag = result['rag_result']
            print(f"- Answer length: {len(rag.answer)} characters")
            print(f"- Sources: {rag.sources}")
            print(f"- Confidence: {rag.confidence:.3f}")
        
        print("\n✅ SYSTEM TEST COMPLETED SUCCESSFULLY!")
        
        # Test RAG system status
        if hasattr(agent, 'rag_tool') and agent.rag_tool:
            status = agent.rag_tool.get_system_status()
            print(f"\n🔧 RAG SYSTEM STATUS:")
            for key, value in status.items():
                print(f"- {key}: {value}")
        
    except Exception as e:
        print(f"❌ SYSTEM TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_system() 