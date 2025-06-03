import streamlit as st
import pandas as pd
from agent import DataAnalystAgent
from config import Config
import re

def get_database_info():
    """Extract database information from the connection URL
    
    Parses the NEON_DATABASE_URL to extract connection details like host, port,
    database name, etc. Masks the password for security when displaying.
    
    Returns:
        dict: Dictionary containing database connection info with masked password,
              or None if no URL is configured
    """
    db_url = Config.NEON_DATABASE_URL
    if not db_url:
        return None
    
    try:
        # Remove protocol prefix (postgresql://, etc.) from URL
        url_without_protocol = db_url.split('://', 1)[1] if '://' in db_url else db_url
        
        # Regex pattern to extract: username:password@host:port/database
        pattern = r'([^:]+):([^@]+)@([^:/]+)(?::(\d+))?/([^?]+)'
        match = re.match(pattern, url_without_protocol)
        
        if match:
            username, password, host, port, database = match.groups()
            # Mask password for security display
            masked_password = '*' * len(password) if password else 'None'
            
            return {
                'host': host,
                'port': port or '5432',  # Default PostgreSQL port
                'database': database,
                'username': username,
                'password_masked': masked_password,
                'connection_type': 'PostgreSQL' if 'postgresql' in db_url.lower() else 'Unknown',
                'schema': 'northwind'  # Hardcoded for this application
            }
    except Exception:
        # Silently handle any parsing errors
        pass
    
    return {'raw_url': db_url[:50] + '...' if len(db_url) > 50 else db_url}

def normalize_question(question: str) -> str:
    """Normalize question for cache key matching
    
    Converts question to lowercase, strips whitespace, and removes punctuation
    to create a consistent cache key for similar questions.
    
    Args:
        question: User's natural language question
        
    Returns:
        str: Normalized question string for use as cache key
    """
    return question.lower().strip().replace("?", "").replace(".", "")

def find_cached_response(question: str, cache: dict) -> dict:
    """Find a cached response for the given question (exact match only)
    
    Searches the response cache for a previously stored answer to the same
    normalized question. Only performs exact matches to ensure relevance.
    
    Args:
        question: User's natural language question
        cache: Dictionary containing cached responses
        
    Returns:
        dict: Cached response data if found, None otherwise
    """
    normalized_q = normalize_question(question)
    
    # Look for exact match in cache using normalized question
    if normalized_q in cache:
        return cache[normalized_q]
    
    return None

def save_to_cache(question: str, result: dict, cache: dict):
    """Save successful response to cache
    
    Stores a comprehensive cache entry containing the question, SQL query,
    results, analysis, visualization, and metadata for future retrieval.
    Only called when user provides positive feedback.
    
    Args:
        question: Original user question
        result: Complete response data from the agent
        cache: Dictionary to store the cached response
    """
    cache_key = normalize_question(question)
    
    # Create a comprehensive cache entry with all response components
    cache_entry = {
        "original_question": question,  # Store original for display
        "sql_query": result.get("sql_query", ""),
        "query_results": result.get("query_results"),
        "analysis": result.get("analysis", ""),
        "combined_analysis": result.get("combined_analysis", ""),  # RAG-enhanced analysis
        "visualization": result.get("visualization"),
        "rag_result": result.get("rag_result"),
        "error": result.get("error", ""),
        "cached_at": pd.Timestamp.now(),  # Timestamp for cache management
        "usage_count": 1  # Track how many times this cache entry is used
    }
    
    cache[cache_key] = cache_entry

def process_feedback(message_id: str, rating: str, question: str, response_content: str, data: pd.DataFrame, chart=None, is_live_feedback: bool = False, live_result: dict = None, live_prompt: str = None):
    """Callback to process feedback and update session state
    
    Handles user feedback (positive/negative) for responses. Stores feedback data
    and automatically caches responses that receive positive ratings for future reuse.
    Supports both historical message feedback and live response feedback.
    
    Args:
        message_id: Unique identifier for the message being rated
        rating: User rating ('positive' or 'negative')
        question: Original question (for historical feedback)
        response_content: Response text (for historical feedback)
        data: Query results DataFrame (for historical feedback)
        chart: Visualization object (for historical feedback)
        is_live_feedback: Whether this is feedback on a just-generated response
        live_result: Complete result data (for live feedback)
        live_prompt: Original question (for live feedback)
    """
    # Record that feedback was given for this message
    st.session_state.message_feedback_status[message_id] = rating

    # Determine actual data based on feedback type (live vs historical)
    actual_question = live_prompt if is_live_feedback else question
    actual_response_content = (live_result.get("combined_analysis") or live_result.get("analysis")) if is_live_feedback and live_result else response_content
    actual_data = live_result.get("query_results") if is_live_feedback and live_result else data
    actual_chart = live_result.get("visualization") if is_live_feedback and live_result else chart

    # Store feedback data for analytics
    feedback_entry = {
        "message_id": message_id,
        "rating": rating,
        "question": actual_question,
        "response": actual_response_content
    }
    st.session_state.feedback_data.append(feedback_entry)

    # Cache positive responses for future reuse
    if rating == "positive":
        cache_payload = {
            "query_results": actual_data,
            "combined_analysis": actual_response_content,
            "visualization": actual_chart,
            "sql_query": live_result.get("sql_query", "") if is_live_feedback and live_result else "",
            "analysis": actual_response_content,
            "rag_result": live_result.get("rag_result") if is_live_feedback and live_result else None,
            "error": ""
        }
        save_to_cache(actual_question, cache_payload, st.session_state.response_cache)

def main():
    """Main Streamlit application function
    
    Sets up the complete AI Data Analyst interface including:
    - Page configuration and title
    - Database connection sidebar with schema information
    - Cache and feedback summary displays
    - Agent initialization and session state management
    - Sample question buttons for user guidance
    - Chat history display with feedback options
    - Real-time question processing with caching
    - Response display with SQL, data, analysis, and visualizations
    """
    st.set_page_config(
        page_title="AI Data Analyst",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🤖 AI Data Analyst")
    st.markdown("Ask questions about your data in natural language!")
    
    # Database connection info sidebar
    with st.sidebar:
        st.header("🔗 Database Connection")
        
        db_info = get_database_info()
        if db_info:
            if 'host' in db_info:
                st.success("Connected to PostgreSQL")
                st.write(f"**Host:** {db_info['host']}")
                st.write(f"**Port:** {db_info['port']}")
                st.write(f"**Database:** {db_info['database']}")
                st.write(f"**Schema:** {db_info['schema']}")
                st.write(f"**Username:** {db_info['username']}")
                st.write(f"**Password:** {db_info['password_masked']}")
            else:
                st.info("Database URL configured")
                st.write(f"**URL:** {db_info['raw_url']}")
        else:
            st.error("No database URL configured")
        
        # Show current schema and available tables if agent is initialized
        if 'agent' in st.session_state:
            try:
                schema_name = st.session_state.agent.db_manager.get_schema_name()
                st.info(f"🎯 Active Schema: **{schema_name}**")
                
                schema_info = st.session_state.agent.db_manager.get_schema_info()
                st.write(f"Schema info retrieved: {len(schema_info)} tables found")
                
                if schema_info:
                    st.header("📋 Available Tables")
                    for table_name, table_info in schema_info.items():
                        with st.expander(f"📊 {table_name}"):
                            st.write("**Columns:**")
                            for col in table_info['columns']:
                                pk_indicator = " 🔑" if col.get('primary_key') else ""
                                st.write(f"• {col['name']} ({col['type']}){pk_indicator}")
                            
                            if table_info['foreign_keys']:
                                st.write("**Foreign Keys:**")
                                for fk in table_info['foreign_keys']:
                                    st.write(f"• {fk['constrained_columns']} → {fk['referred_table']}.{fk['referred_columns']}")
                else:
                    st.warning("No tables found in the current schema")
            except Exception as e:
                st.error(f"Could not load schema: {str(e)}")
                st.write(f"Error details: {e}")
        else:
            st.info("Agent not initialized yet")
        
        # Show cache summary
        if st.session_state.get('response_cache'):
            st.markdown("---")
            st.header("🧠 Response Cache")
            
            cache_size = len(st.session_state.response_cache)
            total_usage = sum(entry.get("usage_count", 0) for entry in st.session_state.response_cache.values())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📋 Cached Responses", cache_size)
            with col2:
                st.metric("🔄 Cache Hits", total_usage - cache_size if total_usage > 0 else 0)
            
            if cache_size > 0:
                st.write("**Recent cached questions:**")
                for i, (_, entry) in enumerate(list(st.session_state.response_cache.items())[-3:]):
                    st.write(f"• {entry.get('original_question', 'Unknown')[:50]}...")
                    
                if st.button("🗑️ Clear Cache", help="Clear all cached responses"):
                    st.session_state.response_cache = {}
                    st.success("Cache cleared!")
                    st.rerun()

        # Show feedback summary if there's any feedback
        if st.session_state.get('feedback_data'):
            st.markdown("---")
            st.header("📊 Feedback Summary")
            
            total_feedback = len(st.session_state.feedback_data)
            positive_feedback = len([f for f in st.session_state.feedback_data if f["rating"] == "positive"])
            negative_feedback = total_feedback - positive_feedback
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👍 Positive", positive_feedback)
            with col2:
                st.metric("👎 Negative", negative_feedback)
            
            if total_feedback > 0:
                satisfaction_rate = (positive_feedback / total_feedback) * 100
                st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    # Initialize the agent
    if 'agent' not in st.session_state:
        with st.spinner("Initializing AI Data Analyst..."):
            try:
                st.session_state.agent = DataAnalystAgent()
                
                db_info = get_database_info()
                if db_info and 'host' in db_info:
                    schema_name = st.session_state.agent.db_manager.get_schema_name()
                    st.success(f"✅ Connected to PostgreSQL database '{db_info['database']}' (schema: {schema_name}) on {db_info['host']}")
                else:
                    st.success("✅ Connected to database successfully!")
                    
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                st.info("💡 Check your database connection and API keys in the sidebar")
                return
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize feedback storage in memory
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    
    # Initialize response cache for satisfied responses
    if 'response_cache' not in st.session_state:
        st.session_state.response_cache = {}
    
    # Initialize simple feedback tracking
    if 'message_feedback_status' not in st.session_state:
        st.session_state.message_feedback_status = {}
    
    # Show sample questions
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try asking questions about the Northwind dataset:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Show all customers", use_container_width=True):
                prompt = "Show me all customers"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
        
        with col2:
            if st.button("🛒 Recent orders", use_container_width=True):
                prompt = "Show me the most recent orders"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("💰 Top selling products", use_container_width=True):
                prompt = "What are the top selling products by revenue?"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
                
        with col4:
            if st.button("👥 Employee performance", use_container_width=True):
                prompt = "Which employees have processed the most orders?"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
        
        col5, col6 = st.columns(2)
        with col5:
            if st.button("🌍 Sales by country", use_container_width=True):
                prompt = "Show me total sales by country"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
                
        with col6:
            if st.button("📈 Monthly sales trends", use_container_width=True):
                prompt = "Show me sales trends by month"
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "data" in message:
                st.dataframe(message["data"], use_container_width=True, hide_index=True)
            if "chart" in message:
                st.plotly_chart(message["chart"], use_container_width=True)
            
            # Simple feedback system - use buttons with on_click callbacks
            if (message["role"] == "assistant" and "data" in message):
                st.markdown("---")
                
                message_id = f"msg_{i}"
                
                if message_id in st.session_state.message_feedback_status:
                    # Show confirmation - feedback already given
                    feedback_type = st.session_state.message_feedback_status[message_id]
                    if feedback_type == "positive":
                        st.success("✅ Thanks for your positive feedback! Response saved to cache.")
                    else:
                        st.info("📝 Thanks for your feedback - we'll work on improving!")
                else:
                    # Show feedback buttons
                    st.markdown("**Was this response helpful?**")
                    question_content = st.session_state.messages[i-1]["content"] if i > 0 else "Unknown question for historical feedback"
                    response_content = message["content"]
                    response_data = message.get("data", pd.DataFrame())
                    response_chart = message.get("chart")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.button(
                            "👍 Yes, helpful", 
                            key=f"yes_helpful_hist_{i}", 
                            on_click=process_feedback, 
                            args=(message_id, "positive", question_content, response_content, response_data, response_chart),
                            use_container_width=True
                        )
                    with col2:
                        st.button(
                            "👎 No, not helpful", 
                            key=f"no_helpful_hist_{i}", 
                            on_click=process_feedback, 
                            args=(message_id, "negative", question_content, response_content, response_data, response_chart),
                            use_container_width=True
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # Check cache first
            cached_response = find_cached_response(prompt, st.session_state.response_cache)
            
            if cached_response:
                # Use cached response
                st.info("📋 Retrieved from cache (previously satisfied response)")
                result = cached_response.copy()
                
                # Update usage count
                cache_key = normalize_question(prompt)
                if cache_key in st.session_state.response_cache:
                    st.session_state.response_cache[cache_key]["usage_count"] += 1
            else:
                # Generate new response
                with st.spinner("Analyzing your question..."):
                    result = st.session_state.agent.process_question(prompt)
            
            if result.get("error"):
                st.error(f"❌ {result['error']}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"❌ {result['error']}"
                })
            else:
                # Check if we have a direct response (like "I don't know") without SQL
                if result.get("combined_analysis") and not result.get("sql_query"):
                    # Direct response case (e.g., "I don't know")
                    st.markdown("🤖 **Response:**")
                    st.markdown(result["combined_analysis"])
                    
                    # RAG system works behind the scenes - no UI display needed
                    
                    # Store the direct response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["combined_analysis"]
                    })
                
                else:
                    # Regular database query case
                    with st.expander("🔍 Generated SQL Query"):
                        st.code(result.get("sql_query", "No SQL query generated"), language="sql")
                    
                    if not result.get("query_results", pd.DataFrame()).empty:
                        st.markdown("📊 **Query Results:**")
                        st.dataframe(result["query_results"], use_container_width=True, hide_index=True)
                        
                        # Show combined analysis if available (new RAG-enhanced output)
                        if result.get("combined_analysis"):
                            st.markdown("🎯 **Comprehensive Analysis:**")
                            st.markdown(result["combined_analysis"])
                        elif result.get("analysis"):
                            st.markdown("🧠 **Analysis:**")
                            st.markdown(result["analysis"])
                        
                        # RAG system works behind the scenes to enhance responses
                        
                        if result.get("visualization"):
                            st.markdown("📈 **Visualization:**")
                            st.plotly_chart(result["visualization"], use_container_width=True)
                        
                        # Add immediate feedback buttons with on_click callbacks
                        # Determine the message_id for this new, not-yet-fully-stored message
                        # It will be the next index in st.session_state.messages
                        live_message_id = f"msg_{len(st.session_state.messages)}"

                        if live_message_id in st.session_state.message_feedback_status:
                            feedback_type = st.session_state.message_feedback_status[live_message_id]
                            if feedback_type == "positive":
                                st.success("✅ Thanks for your positive feedback! Response saved to cache.")
                            else:
                                st.info("📝 Thanks for your feedback - we'll work on improving!")
                        else:
                            st.markdown("**Was this response helpful?**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.button(
                                    "👍 Yes, helpful", 
                                    key=f"yes_helpful_live_{len(st.session_state.messages)}", 
                                    on_click=process_feedback,
                                    args=(live_message_id, "positive", None, None, None, None, True, result, prompt),
                                    use_container_width=True
                                )
                            with col2:
                                st.button(
                                    "👎 No, not helpful", 
                                    key=f"no_helpful_live_{len(st.session_state.messages)}", 
                                    on_click=process_feedback,
                                    args=(live_message_id, "negative", None, None, None, None, True, result, prompt),
                                    use_container_width=True
                                )
                        
                        # Store message with results and feedback flag
                        analysis_content = (result.get("combined_analysis") or 
                                          result.get("analysis") or 
                                          "Here are your results:")
                        
                        message_data = {
                            "role": "assistant",
                            "content": analysis_content,
                            "data": result["query_results"],
                            "feedback_given": False
                        }
                        if result.get("visualization"):
                            message_data["chart"] = result["visualization"]
                        
                        st.session_state.messages.append(message_data)
                    else:
                        st.info("No data found for your query.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "No data found for your query."
                        })

if __name__ == "__main__":
    main() 