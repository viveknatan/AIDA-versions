# LangSmith Tracing Setup Guide

This AI Data Analyst application now includes integrated LangSmith tracing for monitoring and debugging LLM interactions.

## Quick Setup

1. **Sign up for LangSmith** (if you haven't already):
   - Go to [https://smith.langchain.com/](https://smith.langchain.com/)
   - Create an account or sign in
   - Get your API key from the settings page

2. **Add to your environment variables** (`.env` file):
   ```env
   # LangSmith Tracing Configuration
   LANGCHAIN_TRACING_V2=true
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=ai-data-analyst
   ```

3. **Optional Configuration**:
   ```env
   # Custom project name
   LANGCHAIN_PROJECT=my-custom-project-name
   
   # Custom endpoint (usually not needed)
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   
   # Enable debug mode for more verbose tracing logs
   DEBUG=true
   ```

## What Gets Traced

With tracing enabled, you'll see:

### 🔍 **Complete Question Processing Flow**
- Intent classification (database vs. general questions)
- RAG system queries and results
- SQL query generation
- Database query execution
- Data analysis and insights
- Visualization creation decisions
- Combined analysis generation

### 🏷️ **Organized by Tags**
- `intent`, `classification` - Question intent analysis
- `sql`, `generation` - SQL query creation
- `analysis`, `insights` - Data analysis steps
- `data-analyst`, `question-processing` - Overall workflow

### 📊 **Detailed Metadata**
- User questions (truncated for privacy)
- Timestamps
- Model versions
- Confidence scores
- Error tracking

## Viewing Your Traces

1. **Go to LangSmith Dashboard**: [https://smith.langchain.com/](https://smith.langchain.com/)
2. **Select your project**: Look for `ai-data-analyst` or your custom project name
3. **Browse traces**: Each user question creates a complete trace showing the full workflow

## Benefits of Tracing

### 🐛 **Debugging**
- See exactly where failures occur
- Track model confidence scores
- Identify slow components
- Monitor token usage

### 📈 **Performance Monitoring**
- Track response times for each component
- Monitor success/failure rates
- Analyze user question patterns
- Optimize prompt performance

### 💡 **Insights**
- Understand which questions work best
- See RAG vs. SQL routing decisions
- Track visualization creation patterns
- Monitor cache hit rates

## Disabling Tracing

To disable tracing, simply set:
```env
LANGCHAIN_TRACING_V2=false
```

Or remove the LangSmith configuration entirely.

## Security Note

Tracing captures question content and model responses. Ensure your LangSmith project settings align with your privacy requirements.

## Troubleshooting

**Issue**: Tracing not working
- ✅ Check your `LANGSMITH_API_KEY` is correct
- ✅ Verify `LANGCHAIN_TRACING_V2=true`
- ✅ Ensure internet connectivity to LangSmith
- ✅ Check console for tracing-related error messages

**Issue**: Too much trace data
- Adjust trace sampling if needed
- Consider using different projects for dev/prod
- Use filters in the LangSmith dashboard

## Example .env Configuration

```env
# Database Configuration
NEON_DATABASE_URL=your_neon_database_url_here

# OpenAI Configuration  
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=ls_your_api_key_here
LANGCHAIN_PROJECT=ai-data-analyst

# Debug Mode
DEBUG=false
```

Happy tracing! 🚀 