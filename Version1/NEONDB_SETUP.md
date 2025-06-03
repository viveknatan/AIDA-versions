# NeonDB Northwind Schema Setup Guide

## Overview
This AI Data Analyst is configured to work specifically with the **Northwind schema** in your NeonDB database. Follow these steps to set up the connection properly.

## 1. NeonDB Setup

### Get Your NeonDB Connection String
1. Log into your [Neon Console](https://console.neon.tech/)
2. Navigate to your project/database
3. Copy the connection string (should look like):
   ```
   postgresql://username:password@ep-cool-forest-123456.us-east-2.aws.neon.tech/your_database_name
   ```

### Ensure Northwind Schema Exists
The app expects a `northwind` schema with the classic Northwind database tables. If you don't have this:

1. **Option A: Import Northwind Schema**
   - Download the Northwind SQL file
   - Import it into your NeonDB ensuring it creates tables in the `northwind` schema

2. **Option B: Create Schema and Import**
   ```sql
   CREATE SCHEMA IF NOT EXISTS northwind;
   -- Then import your Northwind tables into this schema
   ```

## 2. Environment Configuration

### Create .env File
1. Copy the template:
   ```bash
   cp env_template.txt .env
   ```

2. Update `.env` with your actual values:
   ```env
   NEON_DATABASE_URL=postgresql://your_username:your_password@your_neon_host.neon.tech/your_database_name
   OPENAI_API_KEY=your_openai_api_key_here
   DEBUG=false
   ```

## 3. Connection Features

The updated database connection now:
- ✅ **Automatically connects to the `northwind` schema**
- ✅ **Sets proper SSL mode for NeonDB**
- ✅ **Validates schema and table access**
- ✅ **Falls back to SQLite with sample data if NeonDB is unavailable**

## 4. Verification

Test your connection:
```bash
uv run python -c "from database_test import DatabaseManager; db = DatabaseManager(); print('Connected to:', db.get_schema_name())"
```

Expected output:
```
🔗 Connecting to NeonDB with Northwind schema...
✅ Successfully connected to NeonDB Northwind schema!
Connected to: northwind
```

## 5. Troubleshooting

### Common Issues:

**"Northwind schema not found"**
- Ensure your NeonDB has a `northwind` schema
- Check that tables exist in the `northwind` schema, not `public`

**"Connection failed"**
- Verify your connection string in `.env`
- Ensure your NeonDB allows connections from your IP
- Check that SSL is properly configured

**"No tables found"**
- Verify tables are in the `northwind` schema: `SELECT * FROM information_schema.tables WHERE table_schema='northwind';`

### Debug Mode
Enable debug logging by setting `DEBUG=true` in your `.env` file to see detailed connection information. 