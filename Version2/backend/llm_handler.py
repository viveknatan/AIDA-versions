"""
Modern LLM Handler using LangChain with backward compatibility
"""
import os
from typing import Dict, Any, Optional
from config import Configuration
from models import SQLQuery, DataAnalysis, ErrorResponse, QuestionIntent

try:
    # Try LangChain imports
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
    print("✅ Using LangChain integration")
except ImportError:
    # Fallback to requests
    import requests
    import json
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available, using HTTP fallback")

class LLMHandler:
    def __init__(self, config: Optional[Configuration] = None):
        self.config = config or Configuration()
        self.config.validate()
        
        # Setup LangSmith tracing if enabled
        self._setup_tracing()
        
        if LANGCHAIN_AVAILABLE:
            self._init_langchain()
        else:
            self._init_http_fallback()
    
    def _setup_tracing(self):
        """Setup LangSmith tracing environment variables"""
        if self.config.LANGCHAIN_TRACING_V2.lower() == "true":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
            if self.config.LANGSMITH_API_KEY:
                os.environ["LANGSMITH_API_KEY"] = self.config.LANGSMITH_API_KEY
                
            os.environ["LANGCHAIN_PROJECT"] = self.config.LANGCHAIN_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.config.LANGCHAIN_ENDPOINT
            
            # Optional: Add session name with timestamp for better tracking
            from datetime import datetime
            session_name = f"data-analyst-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            os.environ["LANGCHAIN_SESSION"] = session_name
            
            if self.config.DEBUG:
                print(f"🔍 LangSmith tracing session: {session_name}")
        else:
            # Ensure tracing is disabled
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    def _init_langchain(self):
        """Initialize LangChain components"""
        # Set API key for potential HTTP fallback
        self.api_key = self.config.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        self.chat_llm = ChatOpenAI(
            api_key=self.config.OPENAI_API_KEY,
            model=self.config.llm_model,
            temperature=0.1,
            max_tokens=2000  # Limit response tokens
        )
        
        # Create structured LLMs for different tasks with function calling method
        self.sql_llm = self.chat_llm.with_structured_output(SQLQuery, method="function_calling")
        self.analysis_llm = self.chat_llm.with_structured_output(DataAnalysis, method="function_calling")
        self.intent_llm = self.chat_llm.with_structured_output(QuestionIntent, method="function_calling")
        
        # Add tags for different LLM tasks (for tracing)
        self.sql_llm = self.sql_llm.with_config({"run_name": "SQL_Generation", "tags": ["sql", "generation"]})
        self.analysis_llm = self.analysis_llm.with_config({"run_name": "Data_Analysis", "tags": ["analysis", "insights"]})
        self.intent_llm = self.intent_llm.with_config({"run_name": "Intent_Classification", "tags": ["intent", "classification"]})
        
        # Define prompt templates
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent assistant that determines whether questions require database analysis.

You have access to a database with the following schema information. Your job is to classify whether a user's question can be answered using this database or if it's a general question unrelated to the data.

Database-related questions typically:
- Ask for specific data, counts, totals, averages
- Request information about entities that exist in the database
- Want to see, find, show, list, or analyze data
- Ask about trends, patterns, or comparisons in the data

Non-database questions typically:
- Ask about general knowledge (e.g., "Who is Batman?")
- Request definitions or explanations of concepts
- Ask about current events or information not in the database
- Are conversational or personal questions

Be confident in your classification and provide clear reasoning."""),
            ("user", """Available Database Schema:
{schema}

User Question: {question}

Determine if this question requires database access or can be answered without the database.""")
        ])
        
        self.sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert PostgreSQL SQL analyst. Your task is to convert natural language questions into valid PostgreSQL queries based ONLY on the provided database schema. Adhere strictly to the following rules:

1.  **Exclusivity of Schema**: ONLY use the tables and columns explicitly defined in the 'Database Schema' section below. Do not invent tables or columns. If a requested entity is not in the schema, you MUST indicate that the query cannot be formed. 
2.  **Syntax**: Generate ONLY valid PostgreSQL syntax.
3.  **Statements**: Use ONLY SELECT statements. Do NOT use INSERT, UPDATE, DELETE, or any DDL statements.
4.  **Table Naming**: Use table and column names EXACTLY as they appear in the schema. Do NOT use schema prefixes (e.g., 'public.Orders' should be 'Orders').
5.  **Joins**: When joining tables, ensure the join conditions use columns that are explicitly linked by foreign keys in the schema, or are clearly analogous (e.g. tableA.id and tableB.tableA_id).
6.  **Clarity**: If the question is ambiguous or requires information not present in the schema, state that you cannot generate a query and briefly explain why.
7.  **Row Limits**: Only apply a LIMIT clause (e.g., LIMIT 1000) if the user explicitly asks to see 'all data', 'every record', or a very large number of records. For specific queries, return all matching results unless otherwise specified by a limit in the question itself.
8.  **Confidence**: You will also provide a confidence score and an explanation for your generated query as part of a structured output, but your primary output for this part of the chain is the SQL query itself.

Few-shot Examples (Illustrative - adapt to the actual schema provided later):

Schema Example:
Table: Customers
Columns: CustomerID (TEXT*), CompanyName (TEXT), City (TEXT)
Foreign Keys: 

Table: Orders
Columns: OrderID (INTEGER*), CustomerID (TEXT), OrderDate (DATETIME), Amount (NUMERIC)
Foreign Keys: CustomerID -> Customers.CustomerID

Question: Show me all orders for customer 'ALFKI'.
SQL: SELECT OrderID, OrderDate, Amount FROM Orders WHERE CustomerID = 'ALFKI';

Question: What is the total amount for each customer?
SQL: SELECT c.CompanyName, SUM(o.Amount) AS TotalAmount FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID GROUP BY c.CompanyName;

Question: How many products are there?
SQL: -- I cannot answer this question as there is no 'Products' table in the provided schema.

--- End of Illustrative Examples ---

Now, generate a query for the following actual request based on the REAL schema provided.
"""),
            ("user", """Database Schema:
{schema}

User Question: {question}

Provide the PostgreSQL query.""")
        ])
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst providing insights from query results.

Analyze the data and provide:
1. A concise summary of what the data shows
2. Key insights with their significance
3. Actionable recommendations
4. Notable patterns or trends

Keep insights specific and actionable."""),
            ("user", """Question: {question}

Data Results:
{data}

Analyze this data and provide comprehensive insights.""")
        ])
    
    def _init_http_fallback(self):
        """Initialize HTTP fallback"""
        self.api_key = self.config.OPENAI_API_KEY
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
    
    def classify_question_intent(self, question: str, schema_info: Dict[str, Any]) -> QuestionIntent:
        """Classify whether a question requires database access"""
        
        # Format schema information
        schema_description = self._format_schema_for_prompt(schema_info)
        
        try:
            if LANGCHAIN_AVAILABLE:
                return self._classify_intent_langchain(question, schema_description)
            else:
                return self._classify_intent_http(question, schema_description)
        
        except Exception as e:
            # Conservative fallback - assume it's database-related to avoid missing valid queries
            print(f"Intent classification failed: {e}")
            return QuestionIntent(
                is_database_related=True,
                confidence=0.5,
                reasoning="Intent classification failed, assuming database-related for safety",
                suggested_response=None
            )
    
    def _classify_intent_langchain(self, question: str, schema: str) -> QuestionIntent:
        """Classify intent using LangChain structured output"""
        try:
            formatted_prompt = self.intent_prompt.format_messages(
                question=question,
                schema=schema
            )
            
            response: QuestionIntent = self.intent_llm.invoke(formatted_prompt)
            
            if self.config.DEBUG:
                print(f"Intent Classification - DB Related: {response.is_database_related}")
                print(f"Confidence: {response.confidence}")
                print(f"Reasoning: {response.reasoning}")
            
            return response
            
        except Exception as e:
            if self.config.DEBUG:
                print(f"LangChain intent classification failed: {e}")
            raise e
    
    def _classify_intent_http(self, question: str, schema: str) -> QuestionIntent:
        """HTTP fallback for intent classification"""
        system_prompt = """You are an intelligent assistant that determines whether questions require database analysis.

You have access to a database. Your job is to classify whether a user's question can be answered using this database or if it's a general question unrelated to the data.

Database-related questions typically ask for specific data, counts, totals, averages, or information about entities that exist in the database.

Non-database questions typically ask about general knowledge, definitions, current events, or conversational topics.

Respond with a JSON object containing:
- is_database_related: boolean
- confidence: number between 0 and 1
- reasoning: brief explanation
- suggested_response: response for non-database questions (or null)"""

        user_prompt = f"""Available Database Schema:
{schema}

User Question: {question}

Determine if this question requires database access."""

        response_text = self._make_openai_request(system_prompt, user_prompt, temperature=0.1)
        
        # Parse JSON response (simplified parsing for fallback)
        try:
            import json
            response_data = json.loads(response_text)
            return QuestionIntent(
                is_database_related=response_data.get("is_database_related", True),
                confidence=response_data.get("confidence", 0.5),
                reasoning=response_data.get("reasoning", "HTTP fallback classification"),
                suggested_response=response_data.get("suggested_response")
            )
        except:
            # If parsing fails, assume database-related for safety
            return QuestionIntent(
                is_database_related=True,
                confidence=0.3,
                reasoning="Failed to parse intent classification, assuming database-related",
                suggested_response=None
            )
    
    def generate_sql_query(self, natural_language_question: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query from natural language question"""
        
        # Format schema information
        schema_description = self._format_schema_for_prompt(schema_info)

        if self.config.DEBUG: # Add this block to see the exact schema string
            print(f"""[DEBUG] Schema description provided to LLM for SQL generation:
{schema_description}
--- End of Schema Description ---""", flush=True) # Added flush=True
        
        try:
            if LANGCHAIN_AVAILABLE:
                return self._generate_sql_langchain(natural_language_question, schema_description)
            else:
                return self._generate_sql_http(natural_language_question, schema_description)
        
        except Exception as e:
            # Graceful fallback
            if LANGCHAIN_AVAILABLE:
                print(f"LangChain SQL generation failed: {e}")
                print("Falling back to HTTP method...")
                return self._generate_sql_http(natural_language_question, schema_description)
            else:
                raise Exception(f"SQL generation failed: {str(e)}")
    
    def _generate_sql_langchain(self, question: str, schema: str) -> str:
        """Generate SQL using LangChain structured output"""
        try:
            # Create the prompt
            formatted_prompt = self.sql_prompt.format_messages(
                question=question,
                schema=schema
            )
            
            # Get structured response
            response: SQLQuery = self.sql_llm.invoke(formatted_prompt)
            
            # Log confidence for debugging
            if self.config.DEBUG:
                print(f"SQL Generation Confidence: {response.confidence}")
                print(f"Explanation: {response.explanation}")
            
            return response.sql_query
            
        except Exception as e:
            # Create error response for debugging
            error = ErrorResponse(
                error_type="SQL_GENERATION",
                error_message=str(e),
                suggestion="Try rephrasing your question or check the database schema"
            )
            if self.config.DEBUG:
                print(f"SQL Generation Error: {error}")
            raise e
    
    def _generate_sql_http(self, question: str, schema: str) -> str:
        """HTTP fallback for SQL generation"""
        system_prompt = """You are an expert SQL analyst. Convert natural language questions to PostgreSQL queries.

Rules:
- Generate ONLY valid PostgreSQL syntax
- Limit results to 1000 rows maximum only when the question asks to display all rows of a table  
- Only use SELECT statements (no INSERT, UPDATE, DELETE)
- Use table and column names exactly as shown in the schema
- Do NOT use schema prefixes in table names"""

        user_prompt = f"""Question: {question}

Database Schema:
{schema}

Convert this to a PostgreSQL query."""

        response = self._make_openai_request(system_prompt, user_prompt, temperature=0.1)
        
        # Clean up response
        sql_query = response.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        return sql_query.strip()
    
    def analyze_data(self, data: str, question: str) -> str:
        """Analyze query results and provide insights"""
        
        # Truncate data if it's too long to avoid context length issues
        truncated_data = self._truncate_data_for_analysis(data)
        
        try:
            if LANGCHAIN_AVAILABLE:
                return self._analyze_data_langchain(truncated_data, question)
            else:
                return self._analyze_data_http(truncated_data, question)
        
        except Exception as e:
            # Graceful fallback
            if LANGCHAIN_AVAILABLE:
                print(f"LangChain analysis failed: {e}")
                print("Falling back to HTTP method...")
                return self._analyze_data_http(truncated_data, question)
            else:
                raise Exception(f"Analysis failed: {str(e)}")
    
    def _truncate_data_for_analysis(self, data: str) -> str:
        """Truncate data to fit within context limits"""
        max_chars = self.config.max_context_tokens * 3  # Rough estimate: 1 token ≈ 3 chars
        
        if len(data) <= max_chars:
            return data
        
        # Truncate and add note
        truncated = data[:max_chars]
        # Try to cut at a line break to avoid cutting mid-row
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars * 0.8:  # Only if we don't lose too much
            truncated = truncated[:last_newline]
        
        return truncated + f"\n\n[Note: Data truncated for analysis. Showing first {len(truncated)} characters of {len(data)} total characters.]"
    
    def _analyze_data_langchain(self, data: str, question: str) -> str:
        """Analyze data using LangChain structured output"""
        try:
            # Create the prompt
            formatted_prompt = self.analysis_prompt.format_messages(
                question=question,
                data=data
            )
            
            # Get structured response
            response: DataAnalysis = self.analysis_llm.invoke(formatted_prompt)
            
            # Format the structured response into readable text
            analysis_text = f"{response.summary}\n\n"
            
            if response.key_insights:
                analysis_text += "Key Findings:\n"
                for i, insight in enumerate(response.key_insights, 1):
                    analysis_text += f"{i}. {insight.finding}"
                    if insight.value:
                        analysis_text += f" ({insight.value})"
                    analysis_text += f" - {insight.significance}\n"
                analysis_text += "\n"
            
            if response.notable_patterns:
                analysis_text += "Notable Patterns:\n"
                for pattern in response.notable_patterns:
                    analysis_text += f"• {pattern}\n"
                analysis_text += "\n"
            
            if response.recommendations:
                analysis_text += "Recommendations:\n"
                for rec in response.recommendations:
                    analysis_text += f"• {rec}\n"
            
            return analysis_text.strip()
            
        except Exception as e:
            if self.config.DEBUG:
                print(f"Structured analysis failed: {e}")
            raise e
    
    def _analyze_data_http(self, data: str, question: str) -> str:
        """HTTP fallback for data analysis"""
        system_prompt = """You are a data analyst providing insights from query results.

Analyze the data and provide:
1. Key findings
2. Patterns or trends  
3. Notable insights
4. Summary statistics if relevant

Keep the response concise but informative."""

        user_prompt = f"""Question: {question}

Data:
{data}

Analyze this data and provide comprehensive insights."""

        return self._make_openai_request(system_prompt, user_prompt, temperature=0.3)
    
    def _make_openai_request(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """Make HTTP request to OpenAI API (fallback method)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        import requests
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API request failed with status {response.status_code}: {response.text}")
        
        response_data = response.json()
        
        if "error" in response_data:
            raise Exception(f"OpenAI API error: {response_data['error']}")
        
        if "choices" not in response_data or len(response_data["choices"]) == 0:
            raise Exception("No response choices returned from OpenAI API")
        
        return response_data["choices"][0]["message"]["content"]
    
    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for LLM prompt"""
        formatted_schema = []
        
        for table_name, table_info in schema_info.items():
            columns_list = []
            for col in table_info.get("columns", []):
                col_type = str(col.get("type"))
                # Simplify common type representations for brevity
                if "VARCHAR" in col_type or "TEXT" in col_type:
                    col_type = "TEXT"
                elif "INTEGER" in col_type or "BIGINT" in col_type or "SMALLINT" in col_type:
                    col_type = "INTEGER"
                elif "NUMERIC" in col_type or "DECIMAL" in col_type:
                    col_type = "NUMERIC"
                elif "TIMESTAMP" in col_type or "DATETIME" in col_type:
                    col_type = "DATETIME"
                # Add other simplifications as needed

                columns_list.append(
                    f"{col['name']} ({col_type}{'*' if col.get('primary_key') else ''})"
                )
            columns_str = ", ".join(columns_list)
            table_description = f"Table: {table_name}\nColumns: {columns_str}"

            foreign_keys = table_info.get("foreign_keys", [])
            if foreign_keys:
                fk_descriptions = []
                for fk in foreign_keys:
                    constrained_cols = ", ".join(fk.get("constrained_columns", []))
                    referred_table = fk.get("referred_table")
                    referred_cols = ", ".join(fk.get("referred_columns", []))
                    if constrained_cols and referred_table and referred_cols:
                        fk_descriptions.append(f"{constrained_cols} -> {referred_table}.{referred_cols}")
                if fk_descriptions:
                    table_description += "\nForeign Keys: " + "; ".join(fk_descriptions)
            
            formatted_schema.append(table_description + "\n")
        
        return "\n".join(formatted_schema).strip() # Ensure a single newline at the very end if any