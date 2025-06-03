#!/usr/bin/env python3
"""
Test database question handling
"""

from agent import DataAnalystAgent
from config import Configuration

def test_database_question():
    """Test that database questions work correctly."""
    print("📊 Testing Database Question Handling")
    print("=" * 50)
    
    agent = DataAnalystAgent(Configuration())
    result = agent.process_question("Show me all customers")
    
    print("📊 RESULTS:")
    print(f"✅ Error: {result.get('error', 'None')}")
    print(f"✅ SQL Query: {result.get('sql_query', 'None')[:100]}...")
    print(f"✅ Has Query Results: {not result.get('query_results', pd.DataFrame()).empty}")
    print(f"✅ Combined Analysis: {result.get('combined_analysis', 'None')[:100]}...")
    
    rag_result = result.get('rag_result')
    if rag_result:
        print(f"✅ RAG Confidence: {rag_result.confidence:.3f}")
        print(f"✅ RAG Source Types: {rag_result.source_types}")
    else:
        print("✅ RAG Result: None")
    
    # Test classification
    intent = result.get('intent', {})
    print(f"✅ Database Related: {intent.get('is_database_related', 'Unknown')}")
    print(f"✅ Intent Confidence: {intent.get('confidence', 'Unknown')}")
    
    # Check if it generated SQL and got results
    if result.get('sql_query') and not result.get('query_results', pd.DataFrame()).empty:
        print("🎯 SUCCESS: System correctly generated SQL and retrieved data!")
    else:
        print("❌ ISSUE: System failed to generate SQL or retrieve data")

if __name__ == "__main__":
    import pandas as pd
    test_database_question() 