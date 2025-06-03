from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
import pandas as pd
from config import Configuration
import re
import os
import numpy as np

# Try to import the database manager - use whichever is available
try:
    from database_test import DatabaseManager
    print("✅ Using database_test.py")
except ImportError:
    try:
        from database import DatabaseManager
        print("✅ Using database.py")
    except ImportError:
        raise ImportError("❌ Could not find database.py or database_test.py. Please create one of these files.")

# Import the modern LLM handler
from llm_handler import LLMHandler
from visualization import VisualizationManager
from rag_tool import RAGTool, RAGResult

# Try to import LangChain for tracing
try:
    from langchain.callbacks.manager import CallbackManager
    from langchain_core.callbacks import StdOutCallbackHandler
    LANGCHAIN_CALLBACKS_AVAILABLE = True
except ImportError:
    LANGCHAIN_CALLBACKS_AVAILABLE = False

class AgentState(TypedDict):
    question: str
    schema_info: dict
    intent: dict  # Will store QuestionIntent result
    sql_query: str
    query_results: pd.DataFrame
    analysis: str
    visualization: object
    rag_result: Optional[RAGResult]  # RAG system result
    combined_analysis: str  # Final combined analysis
    error: str

class DataAnalystAgent:
    def __init__(self, config: Configuration = None):
        self.config = config or Configuration()
        self.db_manager = DatabaseManager()
        self.llm_handler = LLMHandler(self.config)
        self.viz_manager = VisualizationManager()
        
        # Setup tracing for the agent
        self._setup_agent_tracing()
        
        # Initialize RAG system
        try:
            self.rag_tool = RAGTool(self.config)
            print("✅ RAG system integrated into agent")
        except Exception as e:
            print(f"⚠️ RAG system initialization failed: {e}")
            self.rag_tool = None
        
        self.graph = self._build_graph()
    
    def _setup_agent_tracing(self):
        """Setup tracing for the agent workflow"""
        if self.config.LANGCHAIN_TRACING_V2.lower() == "true":
            # Set up environment variables for tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if self.config.LANGSMITH_API_KEY:
                os.environ["LANGSMITH_API_KEY"] = self.config.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.config.LANGCHAIN_PROJECT
            
            if self.config.DEBUG:
                print(f"🔍 Agent tracing enabled for project: {self.config.LANGCHAIN_PROJECT}")
    
    @staticmethod
    def is_simple_data_dump_query(sql_query: str) -> bool:
        """Check if the SQL query is a simple data dump (e.g., SELECT * FROM table)."""
        if not sql_query:
            return False
        
        query_lower = sql_query.lower().strip()
        
        # Must be a SELECT query
        if not query_lower.startswith("select "):
            return False
        
        # Check for absence of complex clauses
        if "where " in query_lower or \
           "group by" in query_lower or \
           "having " in query_lower:
            return False
        
        # Check for absence of common aggregate functions
        aggregate_patterns = ["count(", "sum(", "avg(", "min(", "max("]
        if any(agg_func in query_lower for agg_func in aggregate_patterns):
            return False
            
        # If it passed all checks, it's likely a simple data dump
        return True

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def get_schema(state: AgentState) -> AgentState:
            """Get database schema information"""
            try:
                state["schema_info"] = self.db_manager.get_schema_info()
            except Exception as e:
                state["error"] = f"Schema retrieval failed: {str(e)}"
            return state
        
        def classify_intent(state: AgentState) -> AgentState:
            """Classify whether the question requires database access"""
            try:
                intent_result = self.llm_handler.classify_question_intent(
                    state["question"], 
                    state["schema_info"]
                )
                
                # Convert Pydantic model to dict for state storage
                intent_dict = {
                    "is_database_related": intent_result.is_database_related,
                    "confidence": intent_result.confidence,
                    "reasoning": intent_result.reasoning,
                    "suggested_response": intent_result.suggested_response
                }
                
                state["intent"] = intent_dict
                
                # If not database-related, set a helpful response
                if not intent_result.is_database_related:
                    response = ( # Always use the canned message, ignore intent_result.suggested_response
                        "I'm an AI Data Analyst specialized in analyzing database information. "
                        f"Your question '{state['question']}' appears to be a general question that doesn't relate to the available data in our database or my specific knowledge base. "
                        "I can help you analyze data from our database which contains information about "
                        f"{', '.join(state['schema_info'].keys()) if state['schema_info'] else 'various business entities'}. "
                        "Please ask questions about the data in our database, such as showing records, calculating totals, or finding patterns in the data."
                    )
                    state["analysis"] = response
                    
            except Exception as e:
                state["error"] = f"Intent classification failed: {str(e)}"
            
            return state
        
        def generate_sql(state: AgentState) -> AgentState:
            """Generate SQL query from natural language, potentially using RAG context"""
            if state.get("error"):
                return state
            
            try:
                # Check if we have RAG context that might help with SQL generation
                rag_context = ""
                if state.get("rag_result") and state["rag_result"].confidence > 0.2:
                    rag_context = f"\nAdditional Context from Knowledge Base:\n{state['rag_result'].answer[:500]}..."
                
                # Generate SQL with potential RAG context
                enhanced_question = state["question"]
                if rag_context:
                    enhanced_question = f"{state['question']}\n\nContext: Use this additional context if relevant: {rag_context}"
                
                if self.config.DEBUG:
                    print(f"[DEBUG] Schema provided to LLM for SQL generation: {state['schema_info']}")

                state["sql_query"] = self.llm_handler.generate_sql_query(
                    enhanced_question, 
                    state["schema_info"]
                )
                
                if self.config.DEBUG and rag_context:
                    print(f"SQL generation enhanced with RAG context (confidence: {state['rag_result'].confidence:.2f})")
                    
            except Exception as e:
                state["error"] = f"SQL generation failed: {str(e)}"
            return state
        
        def execute_query(state: AgentState) -> AgentState:
            """Execute the generated SQL query"""
            if state.get("error"):
                return state
            
            sql_query = state.get("sql_query", "")
            if not sql_query or sql_query.strip().startswith("--"):
                if self.config.DEBUG:
                    print(f"Skipping execution for empty or comment-only query: {sql_query}")
                # If it's a comment (e.g., LLM saying it can't answer), 
                # set analysis to the comment and avoid DB execution error.
                if sql_query.strip().startswith("--"):
                    state["analysis"] = sql_query.strip()
                    state["query_results"] = pd.DataFrame() # Empty DataFrame
                    state["visualization"] = "Query was a comment, no data to visualize."
                else:
                    state["error"] = "No SQL query generated."
                return state
            
            if self.config.DEBUG:
                print(f"[DEBUG] Executing SQL Query: {sql_query}") # Log the query
            
            try:
                state["query_results"] = self.db_manager.execute_query(sql_query)
                
                # Attempt to sort the DataFrame for table display
                if not state["query_results"].empty:
                    df_to_sort = state["query_results"]
                    numeric_cols = df_to_sort.select_dtypes(include=np.number).columns.tolist()
                    
                    # Filter out potential ID columns from numeric columns to find a value to sort by
                    # This uses a simplified version of the logic in VisualizationManager
                    id_patterns = ['id', '_id', 'ID', '_ID']
                    value_like_cols = [
                        col for col in numeric_cols 
                        if not any(pattern in str(col).lower() for pattern in id_patterns)
                    ]
                    
                    sort_column = None
                    if value_like_cols:
                        # Prioritize columns that might indicate a sum or quantity if multiple exist
                        priority_keywords = ['total', 'sum', 'amount', 'count', 'value', 'quantity', 'sales', 'revenue', 'metric']
                        for keyword in priority_keywords:
                            for col in value_like_cols:
                                if keyword in col.lower():
                                    sort_column = col
                                    break
                            if sort_column:
                                break
                        if not sort_column: # If no priority keyword match, take the first value-like column
                            sort_column = value_like_cols[0]
                    
                    if sort_column:
                        try:
                            state["query_results"] = df_to_sort.sort_values(by=sort_column, ascending=False)
                            if self.config.DEBUG:
                                print(f"[DEBUG] Query results DataFrame sorted by '{sort_column}' descending.")
                        except Exception as e_sort:
                            if self.config.DEBUG:
                                print(f"[DEBUG] Failed to sort DataFrame by column '{sort_column}': {e_sort}. Using unsorted results.")
                                # state["query_results"] remains the unsorted df_to_sort
                    elif self.config.DEBUG:
                        print("[DEBUG] No suitable numeric column found for sorting query results table or all numeric columns look like IDs.")

                if self.config.DEBUG:
                    if not state["query_results"].empty:
                        print(f"[DEBUG] Query Results (first 5 rows):\n{state['query_results'].head().to_string()}")
                    else:
                        print("[DEBUG] Query returned no results (empty DataFrame).")
            except Exception as e:
                state["error"] = f"Query execution failed: {str(e)}"
            return state
        
        def analyze_results(state: AgentState) -> AgentState:
            """Analyze query results"""
            if state.get("error"):
                return state
            
            try:
                # Convert DataFrame to string representation for analysis
                data_str = state["query_results"].to_string()
                state["analysis"] = self.llm_handler.analyze_data(data_str, state["question"])
            except Exception as e:
                state["error"] = f"Analysis failed: {str(e)}"
            return state
        
        def query_rag_system(state: AgentState) -> AgentState:
            """Query the RAG system for additional context"""
            if state.get("error"):
                return state
            
            try:
                if self.rag_tool:
                    rag_result = self.rag_tool.query(state["question"])
                    state["rag_result"] = rag_result
                    
                    if self.config.DEBUG:
                        print(f"RAG result confidence: {rag_result.confidence}")
                        print(f"RAG sources: {rag_result.source_types}")
                else:
                    # Create empty RAG result if tool not available
                    state["rag_result"] = RAGResult(
                        answer="RAG system not available",
                        sources=[],
                        retrieved_docs=[],
                        confidence=0.0,
                        source_types=[]
                    )
            except Exception as e:
                state["error"] = f"RAG query failed: {str(e)}"
                
            return state
        
        def set_direct_response(state: AgentState) -> AgentState:
            """Set a direct 'I don't know' response for non-database questions"""
            if state.get("error"):
                return state
            
            rag_result = state.get("rag_result")
            
            # Set a direct "I don't know" response
            if rag_result and rag_result.confidence > 0.05:
                # Some minimal RAG context available
                state["combined_analysis"] = """I don't have information about that topic in my database or knowledge base. 

My expertise is focused on analyzing the Northwind Traders business data. I can help you with questions about:
- Customer analysis and sales data
- Product performance and inventory
- Employee performance metrics
- Order trends and patterns
- Financial analysis and reporting

Please ask a question related to the Northwind business data."""
            else:
                # No relevant context at all
                state["combined_analysis"] = """I don't know about that topic. 

I'm an AI Data Analyst specialized in analyzing the Northwind Traders business database. I can help you with questions about customers, orders, products, employees, and sales data.

Please ask a question related to the business data in our database."""
            
            if self.config.DEBUG:
                print(f"Set direct response: {state['combined_analysis'][:100]}...")
            
            return state
        
        def should_proceed_to_sql(state: AgentState) -> str:
            """Decide whether to proceed with SQL generation or respond early"""
            if state.get("error"):
                return "end"
            
            intent = state.get("intent", {})
            rag_result = state.get("rag_result")
            
            is_db_related = intent.get("is_database_related", True)
            intent_confidence = intent.get("confidence", 0.0)
            rag_confidence = rag_result.confidence if rag_result else 0.0
            
            # Debug logging
            if self.config.DEBUG:
                print(f"SQL routing: db_related={is_db_related}, intent_conf={intent_confidence:.2f}, rag_conf={rag_confidence:.2f}")
            
            # If clearly not database-related and RAG has low relevant context, respond directly
            if (not is_db_related and intent_confidence > 0.6 and rag_confidence < 0.4):
                if self.config.DEBUG:
                    print("Routing to set_direct_response - question not relevant to database or knowledge base")
                return "set_direct_response"
            
            # Proceed with SQL if it's database-related OR if RAG found relevant context
            if self.config.DEBUG:
                print("Routing to generate_sql - proceeding with database query")
            return "generate_sql"
        
        def create_visualization(state: AgentState) -> AgentState:
            """Create data visualization, unless it's a simple data dump query."""
            if state.get("error"):
                return state
            
            sql_query = state.get("sql_query", "")
            query_results_df = state.get("query_results")
            visualization_output = None # Default to None

            if not sql_query or query_results_df is None or query_results_df.empty:
                if self.config.DEBUG:
                    print("No query, no results, or empty results; skipping visualization.")
            elif DataAnalystAgent.is_simple_data_dump_query(sql_query):
                if self.config.DEBUG:
                    print(f"Skipping visualization for simple data dump query: {sql_query[:100]}...")
            else:
                try:
                    # viz_manager.auto_visualize returns a Plotly Figure object or None
                    fig = self.viz_manager.auto_visualize(
                        query_results_df, 
                        state["question"]
                    )
                    if fig: # If a figure object is created
                        visualization_output = fig.to_dict() # Convert Plotly fig to JSON-serializable dict
                        if self.config.DEBUG:
                            print("Successfully generated Plotly figure dictionary for visualization.")
                    else:
                        if self.config.DEBUG:
                            print("Visualization Manager returned None (e.g. single data point or no suitable chart).")
                        visualization_output = "Visualization not applicable or data unsuitable."

                except Exception as e:
                    print(f"[ERROR] Visualization failed: {str(e)}")
                    state["error"] = f"Visualization failed: {str(e)}" # Log error in state
                    visualization_output = "Visualization generation failed."
            
            state["visualization"] = visualization_output
            return state
        
        def combine_analysis(state: AgentState) -> AgentState:
            """Combine database analysis and RAG results into final response"""
            if state.get("error"): # If an error occurred in a previous step, propagate it
                # Ensure combined_analysis reflects the error if it's the final step for this path
                if not state.get("combined_analysis"): # if not already set by an earlier error handling
                     state["combined_analysis"] = state.get("error", "An unspecified error occurred before analysis combination.")
                return state
            
            try:
                question = state.get("question", "The user's question.")
                db_analysis = state.get("analysis", "").strip() # LLM's analysis of SQL results
                rag_result: Optional[RAGResult] = state.get("rag_result")
                
                # Check if db_analysis is substantial or just a placeholder like "No data found"
                is_db_analysis_substantial = db_analysis and len(db_analysis) > 50 and not db_analysis.lower().startswith("no data")

                final_combined_analysis = ""

                if rag_result and rag_result.confidence > 0.3:
                    # High confidence RAG result - combine with database analysis using LLM
                    
                    # Prepare context for the combination prompt
                    db_context = ""
                    if is_db_analysis_substantial:
                        db_context = f"""
DATABASE ANALYSIS (Based on live data):
{db_analysis}
"""
                    elif db_analysis: # db_analysis exists but is not substantial (e.g. "No results")
                        db_context = f"""
NOTE ON DATABASE QUERY:
{db_analysis} 
(This indicates the specific query returned no direct records or a summary of absence.)
"""
                    else: # No db_analysis at all (should be rare if previous steps are fine)
                        db_context = "NOTE ON DATABASE QUERY: No specific analysis of database results was performed or available for this question."

                    knowledge_base_context = f"""
KNOWLEDGE BASE INFORMATION (Background or related context):
Answer: {rag_result.answer}
Sources: {', '.join(rag_result.sources) if rag_result.sources else 'N/A'}
Confidence: {rag_result.confidence:.2f}
"""
                    
                    combined_prompt = f"""
You are an AI assistant tasked with synthesizing information from two sources: a live database query analysis and a knowledge base. Your goal is to provide a single, comprehensive, and clear answer to the user's question.

USER QUESTION: {question}

{db_context}

{knowledge_base_context}

INSTRUCTIONS FOR SYNTHESIS:
1.  **Primary Focus**: If the DATABASE ANALYSIS provides a direct and substantial answer to the user's question, prioritize that information.
2.  **Supplement with Knowledge Base**: Use the KNOWLEDGE BASE INFORMATION to add relevant background, context, definitions, or related insights that complement the database findings. Do not let it overshadow direct data facts unless the database analysis is minimal (e.g., 'no data found').
3.  **Acknowledge Sources (Implicitly)**: You don't need to explicitly say "The database says..." or "The knowledge base says...". Instead, weave the information together naturally. The user understands you have access to both.
4.  **Handle "No Data"**: If the DATABASE ANALYSIS indicates no specific data was found (e.g., "No results for your query"), and the KNOWLEDGE BASE INFORMATION is relevant, then the knowledge base answer can become the primary part of your response. Clearly state that no direct data was found if that's the case, then provide the knowledge base insights.
5.  **Clarity and Conciseness**: Provide a clear, well-structured, and concise answer. Avoid redundancy.
6.  **Contradictions**: If there's a clear contradiction, prioritize the live DATABASE ANALYSIS as it reflects current data. You might briefly acknowledge the discrepancy if significant (e.g., "While some general documentation suggests X, current data shows Y.").
7.  **Actionable Insights**: If applicable, derive actionable insights from the combined information.

Based on all the above, provide a comprehensive, synthesized response:
"""
                    
                    if self.config.DEBUG:
                        print(f"[DEBUG] Sending to LLM for combined analysis. DB substantial: {is_db_analysis_substantial}, RAG confidence: {rag_result.confidence:.2f}")
                        # print(f"[DEBUG] Combined prompt for LLM:\n{combined_prompt[:1000]}...") # Uncomment to see full prompt

                    combined_analysis_llm_call = self.llm_handler.chat_llm.invoke(combined_prompt)
                    final_combined_analysis = combined_analysis_llm_call.content if hasattr(combined_analysis_llm_call, 'content') else str(combined_analysis_llm_call)
                
                elif is_db_analysis_substantial: 
                    # DB analysis is good, RAG is not strong enough or absent
                    final_combined_analysis = db_analysis
                    if rag_result and rag_result.confidence > 0.05: # Very low confidence RAG, just append as a small note
                        final_combined_analysis += f"\n\n---\nAdditional context (low confidence): {rag_result.answer[:200]}..."
                
                elif rag_result and rag_result.answer and rag_result.confidence > 0.1: # No substantial DB analysis, but some RAG answer
                    final_combined_analysis = f"""While I couldn't find specific data in the database matching your query directly, here's some related information from my knowledge base:

{rag_result.answer}
(Sources: {', '.join(rag_result.sources) if rag_result.sources else 'N/A'})
"""
                    if db_analysis: # Add the (non-substantial) db_analysis note
                        final_combined_analysis = f"{db_analysis}\n\n{final_combined_analysis}"


                elif db_analysis: # Only db_analysis is present (and it's not substantial)
                    final_combined_analysis = db_analysis
                
                else: # Neither is substantial
                    final_combined_analysis = "I was unable to retrieve a specific answer from the database or find sufficiently relevant information in my knowledge base for your question."

                state["combined_analysis"] = final_combined_analysis.strip()
                    
                if self.config.DEBUG:
                    print(f"[DEBUG] Final Combined Analysis (first 100 chars): {state['combined_analysis'][:100]}...")
                    print(f"[DEBUG] RAG confidence: {rag_result.confidence if rag_result else 'N/A'}, DB analysis substantial: {is_db_analysis_substantial}")
                    
            except Exception as e:
                error_message = f"Error during analysis combination: {str(e)}"
                state["error"] = error_message
                # Fallback to db_analysis if available, otherwise the error itself
                state["combined_analysis"] = db_analysis if db_analysis else error_message
                if self.config.DEBUG:
                    print(f"[ERROR] combine_analysis failed: {error_message}")
                    import traceback
                    print(traceback.format_exc())
                
            return state
        
        def should_process_database_query(state: AgentState) -> str:
            """Decide whether to proceed with RAG + database query or return early"""
            if state.get("error"):
                return "end"
            
            intent = state.get("intent", {})
            is_db_related = intent.get("is_database_related", True)
            confidence = intent.get("confidence", 0.0)
            
            # Debug logging
            if self.config.DEBUG:
                print(f"Routing decision: is_db_related={is_db_related}, confidence={confidence}")
            
            # Always proceed to RAG first to check if we have relevant context
            # The RAG system will determine if we should proceed with SQL or respond directly
            if self.config.DEBUG:
                print("Routing to query_rag_system - checking RAG for relevant context")
            return "query_rag_system"
        
        # Build the graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("get_schema", get_schema)
        graph.add_node("classify_intent", classify_intent)
        graph.add_node("generate_sql", generate_sql)
        graph.add_node("execute_query", execute_query)
        graph.add_node("analyze_results", analyze_results)
        graph.add_node("query_rag_system", query_rag_system)  # New RAG node
        graph.add_node("set_direct_response", set_direct_response)  # New direct response node
        graph.add_node("create_visualization", create_visualization)
        graph.add_node("combine_analysis", combine_analysis)  # New combination node
        
        # Add edges with RAG-first workflow
        graph.add_edge("get_schema", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            should_process_database_query,
            {
                "query_rag_system": "query_rag_system",  # Run RAG first for database questions
                "end": END  # Non-database questions still end early
            }
        )
        
        # After RAG, decide whether to proceed with SQL or respond directly
        graph.add_conditional_edges(
            "query_rag_system",
            should_proceed_to_sql,
            {
                "generate_sql": "generate_sql",  # Proceed with database analysis
                "set_direct_response": "set_direct_response"  # Set "I don't know" response
            }
        )
        
        # Direct response goes to END
        graph.add_edge("set_direct_response", END)
        
        # Continue with SQL generation if proceeding
        graph.add_edge("generate_sql", "execute_query")
        graph.add_edge("execute_query", "analyze_results")
        graph.add_edge("analyze_results", "create_visualization")
        graph.add_edge("create_visualization", "combine_analysis")
        graph.add_edge("combine_analysis", END)
        
        # Set entry point
        graph.set_entry_point("get_schema")
        
        return graph.compile()
    
    def process_question(self, question: str) -> AgentState:
        """Process natural language question through the agent workflow"""
        initial_state = AgentState(
            question=question,
            schema_info={},
            intent={},
            sql_query="",
            query_results=pd.DataFrame(),
            analysis="",
            visualization=None,
            rag_result=None,
            combined_analysis="",
            error=""
        )
        
        # Add tracing metadata if tracing is enabled
        if self.config.LANGCHAIN_TRACING_V2.lower() == "true":
            from datetime import datetime
            run_metadata = {
                "question": question[:100],  # Truncate long questions
                "timestamp": datetime.now().isoformat(),
                "agent_version": "1.0",
                "model": self.config.llm_model
            }
            
            try:
                # Invoke with metadata
                result = self.graph.invoke(
                    initial_state,
                    config={
                        "metadata": run_metadata,
                        "tags": ["data-analyst", "question-processing"],
                        "run_name": f"DataAnalyst_Query_{datetime.now().strftime('%H%M%S')}"
                    }
                )
            except Exception as e:
                if self.config.DEBUG:
                    print(f"Tracing metadata failed, running without: {e}")
                result = self.graph.invoke(initial_state)
        else:
            result = self.graph.invoke(initial_state)
            
        if self.config.DEBUG:
            print(f"Final result combined_analysis: '{result.get('combined_analysis', 'NOT_FOUND')}'")
        return result