import sys
import os
import numpy as np
import collections.abc
import datetime

# Add the project root directory to sys.path
# This allows finding the top-level 'RAG' module when app.py is run from backend/
# or when backend.app is run as a module from the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from agent import DataAnalystAgent
import pandas as pd
import re

app = Flask(__name__)
CORS(app)

# Initialize the DataAnalystAgent
# We might need to adjust initialization based on how Config and DataAnalystAgent are structured.
agent = DataAnalystAgent()

response_cache = {}

# --- Helper function to make dictionary serializable ---
def make_value_serializable(value):
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, pd.Timestamp):
        return value.isoformat()
    elif isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    # Add other specific type conversions if needed
    return value

def make_dict_serializable(d):
    if not isinstance(d, collections.abc.Mapping):
        if isinstance(d, collections.abc.Iterable) and not isinstance(d, (str, bytes)):
            return [make_dict_serializable(item) for item in d]
        return make_value_serializable(d)

    new_dict = {}
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            new_dict[k] = make_dict_serializable(v)
        elif isinstance(v, collections.abc.Iterable) and not isinstance(v, (str, bytes)):
            new_dict[k] = [make_dict_serializable(item) for item in v]
        else:
            new_dict[k] = make_value_serializable(v)
    return new_dict
# --- End helper function ---

def get_database_info():
    db_url = Config.NEON_DATABASE_URL
    if not db_url:
        return None
    
    try:
        url_without_protocol = db_url.split('://', 1)[1] if '://' in db_url else db_url
        pattern = r'([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/([^?]+)'
        match = re.match(pattern, url_without_protocol)
        
        if match:
            username, password, host, port, database = match.groups()
            masked_password = '*' * len(password) if password else 'None'
            
            return {
                'host': host,
                'port': port or '5432',
                'database': database,
                'username': username,
                'password_masked': masked_password,
                'connection_type': 'PostgreSQL' if 'postgresql' in db_url.lower() else 'Unknown',
                'schema': 'northwind'
            }
    except Exception:
        pass
    
    return {'raw_url': db_url[:50] + '...' if len(db_url) > 50 else db_url}

def normalize_question(question: str) -> str:
    return question.lower().strip().replace("?", "").replace(".", "")

def find_cached_response(question: str, cache: dict) -> dict:
    normalized_q = normalize_question(question)
    if normalized_q in cache:
        # Ensure cached items are also fully serializable if they are re-sent
        # This might be redundant if they were serialized before caching, but good for safety
        cached_item = cache[normalized_q]
        # Process query_results specifically if it was stored as DataFrame and needs re-conversion
        if isinstance(cached_item.get('query_results'), pd.DataFrame):
             cached_item_copy = cached_item.copy()
             cached_item_copy['query_results'] = cached_item_copy['query_results'].to_dict(orient='records')
             return make_dict_serializable(cached_item_copy) # Serialize the whole item
        return make_dict_serializable(cached_item) # Serialize if not DataFrame case
    return None

def save_to_cache(question: str, result: dict, cache: dict):
    cache_key = normalize_question(question)
    # Ensure result is serializable BEFORE caching, or handle upon retrieval.
    # For simplicity, let's assume `result` passed here has already been processed for serialization for the live response.
    # However, query_results for caching should ideally be the DataFrame if needed later by agent logic.
    # Let's make sure we cache the DataFrame version of query_results but other things are serializable.
    
    entry_to_cache = result.copy() # result should have DataFrame for query_results if from live agent run

    # If query_results is already a list of dicts (e.g. from a previous cache hit being re-cached), convert to DF
    if isinstance(entry_to_cache.get('query_results'), list) and entry_to_cache.get('query_results'):
        entry_to_cache['query_results'] = pd.DataFrame(entry_to_cache['query_results'])
    
    cache_entry = {
        "original_question": entry_to_cache.get("original_question", question),
        "sql_query": entry_to_cache.get("sql_query", ""),
        "query_results": entry_to_cache.get("query_results"), # Store as DataFrame
        "analysis": entry_to_cache.get("analysis", ""),
        "combined_analysis": entry_to_cache.get("combined_analysis", ""),
        "visualization": make_dict_serializable(entry_to_cache.get("visualization")), # ensure viz is serializable
        "rag_result": make_dict_serializable(entry_to_cache.get("rag_result")), # ensure RAG is serializable
        "error": entry_to_cache.get("error", ""),
        "cached_at": datetime.datetime.now().isoformat(),
        "usage_count": 1
    }
    cache[cache_key] = cache_entry

@app.route('/db-info', methods=['GET'])
def db_info():
    info = get_database_info()
    if info:
        return jsonify(make_dict_serializable(info))
    return jsonify({"error": "Database URL not configured"}), 500

@app.route('/schema-info', methods=['GET'])
def schema_info():
    try:
        schema = agent.db_manager.get_schema_info()
        if schema:
            return jsonify(make_dict_serializable(schema))
        return jsonify({"error": "Could not retrieve schema"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Try to find in cache first
    cached_response_data = find_cached_response(question, response_cache)
    if cached_response_data:
        cached_response_data["usage_count"] = cached_response_data.get("usage_count", 0) + 1
        # NOTE: The actual update of usage_count in the persisted cache happens if/when a cached item is re-saved via positive feedback.
        # This in-memory increment is for the current response only if not re-saved.
        return jsonify({"response": cached_response_data, "type": "cache"})

    try:
        result = agent.process_question(question) # This is AgentState dict

        response_data = result.copy() # result is the raw AgentState

        # 1. Handle query_results: Convert DataFrame to list of dicts for JSON response
        if isinstance(response_data.get('query_results'), pd.DataFrame):
            response_data['query_results'] = response_data['query_results'].to_dict(orient='records')
        
        # 2. Simplify rag_result for frontend (exclude raw Document objects)
        simplified_rag_for_response = None
        raw_rag_result_from_agent = response_data.get('rag_result')

        if raw_rag_result_from_agent is not None:
            temp_rag_data_as_dict = None
            if isinstance(raw_rag_result_from_agent, dict):
                temp_rag_data_as_dict = raw_rag_result_from_agent
            elif hasattr(raw_rag_result_from_agent, 'model_dump'):  # Pydantic v2+
                temp_rag_data_as_dict = raw_rag_result_from_agent.model_dump()
            elif hasattr(raw_rag_result_from_agent, 'dict'):  # Pydantic v1
                temp_rag_data_as_dict = raw_rag_result_from_agent.dict()
            else:
                print(f"[Debug] RAGResult is type {type(raw_rag_result_from_agent)}, attempting getattr fallback.")
                try:
                    temp_rag_data_as_dict = {
                        "answer": getattr(raw_rag_result_from_agent, 'answer', "RAG answer not available"),
                        "sources": getattr(raw_rag_result_from_agent, 'sources', []),
                        "confidence": getattr(raw_rag_result_from_agent, 'confidence', 0.0),
                        "source_types": getattr(raw_rag_result_from_agent, 'source_types', [])
                    }
                except AttributeError as e:
                    print(f"[Error] Failed to access RAGResult attributes via getattr: {e}")
                    temp_rag_data_as_dict = {"answer": "Error processing RAG data", "sources": [], "confidence": 0.0, "source_types": []}
            
            if temp_rag_data_as_dict is not None:
                simplified_rag_for_response = {
                    "answer": temp_rag_data_as_dict.get('answer'),
                    "sources": temp_rag_data_as_dict.get('sources'),
                    "confidence": temp_rag_data_as_dict.get('confidence'),
                    "source_types": temp_rag_data_as_dict.get('source_types')
                    # Exclude 'retrieved_docs' from the response to frontend
                }
        response_data['rag_result'] = simplified_rag_for_response
        
        # IMPORTANT: Remove automatic caching from /query endpoint
        # Caching will now be handled by the /feedback endpoint upon positive feedback.
        # Original line: save_to_cache(question, result, response_cache) # result has DataFrame

        return jsonify({"response": make_dict_serializable(response_data), "type": "live"})

    except Exception as e:
        app.logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.json
    question = data.get('question')
    rating = data.get('rating')
    result_to_cache = data.get('result') # This is the AI response data from the frontend

    if not all([question, rating, result_to_cache]):
        return jsonify({"error": "Missing question, rating, or result data"}), 400

    if rating == 'positive':
        try:
            # Ensure query_results is a DataFrame before caching if it exists and is a list of dicts
            # The result_to_cache comes from frontend, so query_results will be list of dicts.
            if isinstance(result_to_cache.get('query_results'), list):
                result_to_cache_copy = result_to_cache.copy()
                result_to_cache_copy['query_results'] = pd.DataFrame(result_to_cache_copy['query_results'])
            else:
                result_to_cache_copy = result_to_cache # If not list, use as is (might be None or already df if logic changes)
            
            save_to_cache(question, result_to_cache_copy, response_cache)
            return jsonify({"message": "Feedback received and response cached"}), 200
        except Exception as e:
            app.logger.error(f"Error saving to cache: {e}", exc_info=True)
            return jsonify({"error": f"Failed to cache response: {str(e)}"}), 500
    elif rating == 'negative':
        # For now, we just acknowledge negative feedback. Could be used for logging or model retraining later.
        return jsonify({"message": "Negative feedback received"}), 200
    else:
        return jsonify({"error": "Invalid rating"}), 400

@app.route('/cache-summary', methods=['GET'])
def cache_summary():
    cache_size = len(response_cache)
    total_usage = sum(entry.get("usage_count", 0) for entry in response_cache.values())
    
    recent_cached_items_serializable = []
    for key, value in response_cache.items():
        # Value from cache might have DataFrame for query_results
        entry_copy = value.copy()
        if isinstance(entry_copy.get("query_results"), pd.DataFrame):
            entry_copy["query_results"] = entry_copy["query_results"].to_dict(orient='records')
        recent_cached_items_serializable.append(make_dict_serializable(entry_copy))

    return jsonify({
        "cache_size": cache_size,
        "cache_hits": total_usage - cache_size if total_usage > cache_size else 0,
        "cached_items": recent_cached_items_serializable[:5]
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001) # Running on a different port than default 5000 