"""
Pydantic models for structured LLM outputs
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionIntent(BaseModel):
    """Model for classifying question intent"""
    is_database_related: bool = Field(
        description="Whether this question requires access to the database"
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was made",
        max_length=1000
    )
    suggested_response: Optional[str] = Field(
        description="Suggested response for non-database questions",
        default=None
    )

class SQLQuery(BaseModel):
    """Model for SQL query generation"""
    sql_query: str = Field(
        description="The generated SQL query",
        min_length=1
    )
    explanation: str = Field(
        description="Brief explanation of what the query does",
        max_length=500
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
        default=0.8
    )

class DataInsight(BaseModel):
    """Model for individual data insights"""
    finding: str = Field(description="Key finding or insight")
    significance: str = Field(description="Why this finding is important")
    value: Optional[str] = Field(description="Specific value or metric", default=None)

class DataAnalysis(BaseModel):
    """Model for comprehensive data analysis"""
    summary: str = Field(
        description="High-level summary of the analysis",
        max_length=200
    )
    key_insights: List[DataInsight] = Field(
        description="List of key insights from the data",
        min_items=1,
        max_items=5
    )
    recommendations: List[str] = Field(
        description="Actionable recommendations based on the analysis",
        max_items=3,
        default=[]
    )
    notable_patterns: List[str] = Field(
        description="Notable patterns or trends in the data",
        max_items=3,
        default=[]
    )

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error_type: str = Field(description="Type of error encountered")
    error_message: str = Field(description="Human-readable error message")
    suggestion: Optional[str] = Field(description="Suggestion for resolving the error", default=None)

class RAGResponse(BaseModel):
    """Model for RAG system responses"""
    answer: str = Field(description="Answer from RAG system")
    sources: List[str] = Field(description="Sources used for the answer")
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    source_types: List[str] = Field(description="Types of sources used (pdf, database, etc.)")
    retrieved_docs_count: int = Field(description="Number of documents retrieved")
    has_relevant_info: bool = Field(description="Whether relevant information was found")