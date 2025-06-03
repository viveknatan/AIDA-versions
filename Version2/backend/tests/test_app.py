import pytest
import json
import importlib
from unittest.mock import MagicMock, patch

# Delay app-specific imports to be handled by fixtures

@pytest.fixture(scope='function') # Ensure this runs fresh for each test
def app_and_config_reloaded(monkeypatch):
    """
    Mocks environment, prevents .env loading, reloads config and app, and returns them.
    This ensures that Config object and app are initialized with mocked environment variables.
    """
    monkeypatch.setenv("NEON_DATABASE_URL", "postgresql://testuser:testpass@testhost/testdb_fixture_reloaded")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture_openai_key_reloaded")
    # Mock other necessary env vars if your Config or app initialization depends on them
    # For example, if LANGCHAIN_TRACING_V2 enables features that need other keys:
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false") # Disable tracing for tests unless specifically testing it
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy_langsmith_key_for_tests")
    monkeypatch.setenv("LANGCHAIN_PROJECT", "test_project")

    with patch('dotenv.load_dotenv', return_value=True) as mock_load_dotenv:
        # Import modules here to ensure they are loaded *after* env mocks are set
        from backend import config as reloaded_config_module
        from backend import app as reloaded_flask_app_module
        from backend.app import response_cache as reloaded_global_response_cache

        importlib.reload(reloaded_config_module)
        importlib.reload(reloaded_flask_app_module)
        
        # Clear cache before each test
        reloaded_global_response_cache.clear()
        
        return reloaded_flask_app_module, reloaded_config_module, reloaded_global_response_cache

@pytest.fixture
def client(app_and_config_reloaded):
    reloaded_flask_app_module, _, _ = app_and_config_reloaded
    return reloaded_flask_app_module.app.test_client()

@pytest.fixture
def mock_agent(app_and_config_reloaded, monkeypatch):
    reloaded_flask_app_module, _, _ = app_and_config_reloaded
    
    # Attempt to import DataAnalystAgent for spec, handle if not found for some test runner setups
    try:
        from backend.agent import DataAnalystAgent
    except ImportError:
        DataAnalystAgent = None # Fallback
        
    if DataAnalystAgent:
        mocked_agent_instance = MagicMock(spec=DataAnalystAgent)
    else:
        mocked_agent_instance = MagicMock()

    mocked_agent_instance.db_manager = MagicMock()
    mocked_agent_instance.db_manager.get_schema_info = MagicMock(return_value={'users': {'columns': [{'name': 'id', 'type': 'integer', 'primary_key': True}], 'foreign_keys':[]}})
    
    default_run_result = {
        'sql_query': 'SELECT * FROM mocked_users',
        'query_results': [{'id': 1, 'name': 'Mocked Test User'}], 
        'analysis': 'This is a mocked analysis.',
        'combined_analysis': 'This is a RAG mocked analysis.',
        'visualization': {'type': 'bar', 'data': [{'id':1, 'name':'Mocked Test User'}], 'x_axis': 'name', 'y_axis':'id'},
        'rag_result': 'Mocked RAG data here'
    }
    mocked_agent_instance.run = MagicMock(return_value=default_run_result)
    
    monkeypatch.setattr(reloaded_flask_app_module, 'agent', mocked_agent_instance)
    return mocked_agent_instance

# --- Test Cases ---
# All tests will now implicitly use the app instance configured by app_and_config_reloaded via the client fixture.

def test_db_info(client):
    response = client.get('/db-info')
    data = json.loads(response.data)
    assert response.status_code == 200
    # Check against value from app_and_config_reloaded fixture
    assert data['host'] == 'testhost' 
    assert data['database'] == 'testdb_fixture_reloaded' 
    assert data['username'] == 'testuser'

def test_schema_info(client, mock_agent):
    response = client.get('/schema-info')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'users' in data
    mock_agent.db_manager.get_schema_info.assert_called_once()

def test_handle_query_live(client, mock_agent):
    question = 'Show me users'
    response = client.post('/query', json={'question': question})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['type'] == 'live'
    assert data['response']['sql_query'] == 'SELECT * FROM mocked_users'
    assert data['response']['query_results'][0]['name'] == 'Mocked Test User' 
    mock_agent.run.assert_called_once_with(question)

def test_handle_query_cached(client, mock_agent):
    question = 'Cache this question for test'
    
    live_response = client.post('/query', json={'question': question})
    live_data = json.loads(live_response.data)
    assert live_data['type'] == 'live'
    mock_agent.run.assert_called_with(question)

    feedback_payload = {
        'question': question,
        'rating': 'positive',
        'result': live_data['response']
    }
    feedback_response = client.post('/feedback', json=feedback_payload)
    assert feedback_response.status_code == 200

    mock_agent.run.reset_mock() 

    cached_response = client.post('/query', json={'question': question})
    cached_data = json.loads(cached_response.data)
    assert cached_data['type'] == 'cache'
    assert cached_data['response']['original_question'] == question
    mock_agent.run.assert_not_called() 

def test_handle_query_no_question(client):
    response = client.post('/query', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'No question provided' in data['error']

def test_feedback_positive_and_caches(client, mock_agent):
    question = 'Test positive feedback caching'
    
    query_response = client.post('/query', json={'question': question})
    query_data = json.loads(query_response.data)
    assert query_response.status_code == 200
    
    feedback_payload = {
        'question': question,
        'rating': 'positive',
        'result': query_data['response'] 
    }
    response = client.post('/feedback', json=feedback_payload)
    assert response.status_code == 200
    assert 'Feedback received and response cached' in json.loads(response.data)['message']

    mock_agent.run.reset_mock() 
    cache_query_response = client.post('/query', json={'question': question})
    cache_query_data = json.loads(cache_query_response.data)
    assert cache_query_data['type'] == 'cache'
    mock_agent.run.assert_not_called()

def test_feedback_missing_data(client):
    response = client.post('/feedback', json={'question': 'Missing some data'})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'Missing feedback data' in data['error']

def test_cache_summary_after_caching(client, mock_agent):
    question_for_cache = 'Question for cache summary test'

    live_response = client.post('/query', json={'question': question_for_cache})
    live_data = json.loads(live_response.data)

    feedback_payload = {
        'question': question_for_cache,
        'rating': 'positive',
        'result': live_data['response']
    }
    client.post('/feedback', json=feedback_payload)

    summary_response = client.get('/cache-summary')
    assert summary_response.status_code == 200
    summary_data = json.loads(summary_response.data)
    
    assert summary_data['cache_size'] == 1
    assert summary_data['cache_hits'] == 0 
    assert len(summary_data['cached_items']) == 1
    assert summary_data['cached_items'][0]['original_question'] == question_for_cache
    assert summary_data['cached_items'][0]['usage_count'] == 1

# Note: The 'query_results' in the mocked agent.run now returns a list of dicts,
# because the app route expects a DataFrame which it then converts using .to_dict(orient='records').
# For simplicity in mock, we provide the already dict-oriented list.
# The save_to_cache in the app handles pd.DataFrame(result['query_results']) if it's a list from JSON.

# Note on running:
# 1. Ensure .env exists in backend/ or critical env vars are otherwise available if not mocked by autouse.
#    (mock_env_vars should cover NEON_DATABASE_URL and OPENAI_API_KEY for config validation)
# 2. cd backend
# 3. python -m pytest (or uv run pytest) 