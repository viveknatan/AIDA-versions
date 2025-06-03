#!/usr/bin/env python3
"""
Test Batman question handling
"""

from agent import DataAnalystAgent
from config import Configuration

def test_batman_question():
    """Test that Batman question gets proper 'I don't know' response."""
    print("🦇 Testing Batman Question Handling")
    print("=" * 50)
    
    config = Configuration()
    config.DEBUG = True
    agent = DataAnalystAgent(config)
    result = agent.process_question("Who is Batman?")
    
    print("📊 RESULTS:")
    print(f"✅ Error: {result.get('error', 'None')}")
    print(f"✅ SQL Query: {result.get('sql_query', 'None')}")
    combined_analysis = result.get('combined_analysis', 'None')
    print(f"✅ Combined Analysis: {combined_analysis[:200] if combined_analysis != 'None' else 'None'}...")
    print(f"✅ Has Combined Analysis: {bool(combined_analysis and combined_analysis != 'None')}")
    print(f"✅ Full Result Keys: {list(result.keys())}")
    if combined_analysis and combined_analysis != 'None':
        print(f"✅ Full Combined Analysis: {combined_analysis}")
    
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
    
    # Check if it avoided SQL generation
    if not result.get('sql_query') and result.get('combined_analysis'):
        print("🎯 SUCCESS: System correctly avoided SQL generation and provided direct response!")
    elif result.get('sql_query'):
        print("❌ ISSUE: System still attempted SQL generation")
    else:
        print("❌ ISSUE: System provided neither SQL nor direct response")

if __name__ == "__main__":
    test_batman_question() 