import os
from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class Configuration:
    """Configuration for the AI Data Analyst"""
    
    # Database Configuration
    NEON_DATABASE_URL: str = os.getenv("NEON_DATABASE_URL", "")
    
    # LLM Configuration
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"  # Use gpt-4o which supports structured outputs
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # LangSmith Tracing Configuration
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ai-data-analyst")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    # Analysis Configuration
    max_query_attempts: int = 3
    enable_visualization: bool = True
    max_context_tokens: int = 6000  # Leave room for response
    
    # App Configuration
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def from_runnable_config(cls, config: Optional[dict] = None) -> "Configuration":
        """Create a Configuration instance from a config dict or environment variables"""
        configurable = config or {}
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        # Filter out None values and use defaults
        filtered_values = {k: v for k, v in values.items() if v is not None}
        return cls(**filtered_values)
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.OPENAI_API_KEY and self.llm_provider == LLMProvider.OPENAI:
            raise ValueError("OpenAI API key is required when using OpenAI provider")
        if not self.NEON_DATABASE_URL:
            print("Warning: No database URL configured, will use SQLite fallback")
        
        # LangSmith tracing validation
        if self.LANGCHAIN_TRACING_V2.lower() == "true":
            if not self.LANGSMITH_API_KEY:
                print("Warning: LangSmith tracing enabled but LANGSMITH_API_KEY not found")
                print("Get your API key from: https://smith.langchain.com/")
            else:
                print(f"✅ LangSmith tracing enabled for project: {self.LANGCHAIN_PROJECT}")
        else:
            print("💡 To enable LangSmith tracing, set LANGCHAIN_TRACING_V2=true in your .env file")
            
        return True

# Backward compatibility - keep the old Config class
class Config:
    """Legacy Config class for backward compatibility"""
    def __init__(self):
        self._config = Configuration()
        self._config.validate()
    
    @property
    def NEON_DATABASE_URL(self):
        return self._config.NEON_DATABASE_URL
    
    @property
    def OPENAI_API_KEY(self):
        return self._config.OPENAI_API_KEY
    
    @property
    def DEBUG(self):
        return self._config.DEBUG

# Create default instance for backward compatibility
Config = Config()