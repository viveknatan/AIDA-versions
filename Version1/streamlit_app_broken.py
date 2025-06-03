import streamlit as st
import pandas as pd
from agent import DataAnalystAgent
from config import Config
import re

def get_database_info():
    """Extract database information from the connection URL"""
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

def main():
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
                st.write(f"Schema info retrieved: {len(schema_info)} tables found")  # Debug info
                
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
                st.write(f"Error details: {e}")  # More detailed error info
        else:
            st.info("Agent not initialized yet")
        
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
                st.dataframe(message["data"], use_container_width=True)
            if "chart" in message:
                st.plotly_chart(message["chart"], use_container_width=True)
            
            # Only show feedback buttons in history if no feedback was given live AND no feedback given yet
            if (message["role"] == "assistant" and 
                "data" in message and 
                not message.get("feedback_given", False) and
                not any(f["message_index"] == i for f in st.session_state.feedback_data)):
                
                st.markdown("---")
                st.markdown("**Was this response helpful?**")
                
                col1, col2, col3 = st.columns([1, 1, 4])
                
                with col1:
                    if st.button("👍", key=f"thumbs_up_{i}", help="Yes, helpful"):
                        # Save positive feedback
                        feedback = {
                            "message_index": i,
                            "rating": "positive",
                            "question": st.session_state.messages[i-1]["content"] if i > 0 else "",
                            "response": message["content"]
                        }
                        st.session_state.feedback_data.append(feedback)
                        st.session_state.messages[i]["feedback_given"] = True
                        st.success("Thanks for the feedback! 👍")
                        st.rerun()
                
                with col2:
                    if st.button("👎", key=f"thumbs_down_{i}", help="No, not helpful"):
                        # Save negative feedback
                        feedback = {
                            "message_index": i,
                            "rating": "negative", 
                            "question": st.session_state.messages[i-1]["content"] if i > 0 else "",
                            "response": message["content"]
                        }
                        st.session_state.feedback_data.append(feedback)
                        st.session_state.messages[i]["feedback_given"] = True
                        st.info("Thanks for the feedback! We'll work on improving. 👎")
                        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
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
                    
                    # Show RAG information if available
                    rag_result = result.get("rag_result")
                    if rag_result and rag_result.confidence > 0.05:
                        with st.expander(f"🧠 Knowledge Base Insights (Confidence: {rag_result.confidence:.2f})"):
                            st.write("**Sources used:**", ", ".join(rag_result.source_types))
                            st.write("**Documents retrieved:**", len(rag_result.retrieved_docs))
                            if rag_result.sources:
                                st.write("**Source files:**", ", ".join(rag_result.sources[:3]))
                    
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
                        st.dataframe(result["query_results"], use_container_width=True)
                        
                        # Show combined analysis if available (new RAG-enhanced output)
                        if result.get("combined_analysis"):
                            st.markdown("🎯 **Comprehensive Analysis:**")
                            st.markdown(result["combined_analysis"])
                        elif result.get("analysis"):
                            st.markdown("🧠 **Analysis:**")
                            st.markdown(result["analysis"])
                        
                        # Show RAG information if available
                        rag_result = result.get("rag_result")
                        if rag_result and rag_result.confidence > 0.1:
                            with st.expander(f"🧠 Knowledge Base Insights (Confidence: {rag_result.confidence:.2f})"):
                                st.write("**Sources used:**", ", ".join(rag_result.source_types))
                                st.write("**Documents retrieved:**", len(rag_result.retrieved_docs))
                                if rag_result.sources:
                                    st.write("**Source files:**", ", ".join(rag_result.sources[:3]))
                        
                        if result.get("visualization"):
                            st.markdown("📈 **Visualization:**")
                            st.plotly_chart(result["visualization"], use_container_width=True)
                    
                    # Add immediate feedback buttons
                    st.markdown("---")
                    st.markdown("**Was this response helpful?**")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    current_msg_index = len(st.session_state.messages)  # This will be the index after we append
                    
                    with col1:
                        if st.button("👍", key=f"live_thumbs_up_{current_msg_index}", help="Yes, helpful"):
                            # Save positive feedback
                            feedback = {
                                "message_index": current_msg_index,
                                "rating": "positive",
                                "question": prompt,
                                "response": result["analysis"] if result["analysis"] else "Query results provided"
                            }
                            st.session_state.feedback_data.append(feedback)
                            st.success("Thanks for the feedback! 👍")
                    
                    with col2:
                        if st.button("👎", key=f"live_thumbs_down_{current_msg_index}", help="No, not helpful"):
                            # Save negative feedback
                            feedback = {
                                "message_index": current_msg_index,
                                "rating": "negative", 
                                "question": prompt,
                                "response": result["analysis"] if result["analysis"] else "Query results provided"
                            }
                            st.session_state.feedback_data.append(feedback)
                            st.info("Thanks for the feedback! We'll work on improving. 👎")
                    
                    # Store message with results and feedback flag
                    analysis_content = (result.get("combined_analysis") or 
                                      result.get("analysis") or 
                                      "Here are your results:")
                    
                    message_data = {
                        "role": "assistant",
                        "content": analysis_content,
                        "data": result["query_results"],
                        "feedback_given": False  # Will be updated if feedback is given in history view
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