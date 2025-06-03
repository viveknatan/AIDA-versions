#!/usr/bin/env python3
"""
Test script to verify NeonDB connection and Northwind schema setup
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def test_neondb_connection():
    """Test connection to NeonDB with Northwind schema"""
    print("🧪 Testing NeonDB Connection with Northwind Schema")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    db_url = os.getenv("NEON_DATABASE_URL")
    
    if not db_url:
        print("❌ NEON_DATABASE_URL not found in environment variables")
        print("💡 Please create a .env file with your NeonDB connection string")
        print("   Example: NEON_DATABASE_URL=postgresql://user:pass@host.neon.tech/dbname")
        return False
    
    print(f"🔗 Found database URL: {db_url[:30]}...{db_url[-20:]}")
    
    try:
        # Prepare connection URL (fix psycog -> psycopg issue)
        if db_url.startswith('postgresql://'):
            db_url = db_url.replace('postgresql://', 'postgresql+psycopg://')
        elif db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql+psycopg://')
        elif 'psycog' in db_url:
            # Fix common psycog typo in existing URLs
            db_url = db_url.replace('psycog', 'psycopg')
        
        # Add connection parameters
        if 'sslmode=' not in db_url:
            if '?' in db_url:
                db_url += '&sslmode=require'
            else:
                db_url += '?sslmode=require'
        
        if 'connect_timeout=' not in db_url:
            if '?' in db_url:
                db_url += '&connect_timeout=10'
            else:
                db_url += '?connect_timeout=10'
        
        print("🔌 Creating database connection...")
        engine = create_engine(db_url, echo=False)
        
        with engine.connect() as conn:
            print("✅ Successfully connected to NeonDB!")
            
            # Check if northwind schema exists
            print("\n🔍 Checking for Northwind schema...")
            result = conn.execute(text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name = 'northwind'
            """))
            
            if result.fetchone():
                print("✅ Northwind schema found!")
                
                # Check for tables in northwind schema
                print("📊 Checking for tables in Northwind schema...")
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'northwind'
                    ORDER BY table_name
                """))
                
                tables = [row[0] for row in result.fetchall()]
                
                if tables:
                    print(f"✅ Found {len(tables)} tables in Northwind schema:")
                    for table in tables:
                        print(f"   - {table}")
                    
                    # Test setting search path
                    print("\n🛠️  Setting search path to Northwind...")
                    conn.execute(text("SET search_path TO northwind, public"))
                    print("✅ Search path set successfully!")
                    
                    # Test sample query
                    print("\n🧪 Testing sample query...")
                    try:
                        result = conn.execute(text("SELECT COUNT(*) as count FROM customers"))
                        count = result.fetchone()[0]
                        print(f"✅ Sample query successful! Found {count} customers")
                        
                        print(f"\n🎉 NeonDB Northwind setup is PERFECT!")
                        print("✅ Your app will connect to the Northwind schema successfully")
                        return True
                        
                    except Exception as e:
                        print(f"⚠️  Sample query failed: {e}")
                        print("💡 Tables might exist but have different names or structure")
                        
                else:
                    print("❌ Northwind schema exists but no tables found")
                    print("💡 You need to import Northwind data into the schema")
                    
            else:
                print("❌ Northwind schema not found")
                print("\n🔍 Available schemas:")
                result = conn.execute(text("""
                    SELECT schema_name 
                    FROM information_schema.schemata 
                    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                    ORDER BY schema_name
                """))
                schemas = [row[0] for row in result.fetchall()]
                for schema in schemas:
                    print(f"   - {schema}")
                
                print("\n💡 You need to create the 'northwind' schema and import the data")
                print("   SQL: CREATE SCHEMA northwind;")
    
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your NEON_DATABASE_URL format")
        print("2. Ensure your NeonDB allows connections from your IP")
        print("3. Verify your credentials are correct")
        print("4. Check that SSL is properly configured")
        return False
    
    return False

if __name__ == "__main__":
    success = test_neondb_connection()
    
    if success:
        print(f"\n🚀 Ready to run: uv run streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️  Please fix the issues above before running the app")
        print("📖 See NEONDB_SETUP.md for detailed setup instructions") 