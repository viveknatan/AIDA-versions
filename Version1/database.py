from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from typing import Dict, List, Any
import pandas as pd
import sqlite3

class DatabaseManager:
    def __init__(self):
        try:
            from config import Config
            db_url = Config.NEON_DATABASE_URL
            
            if db_url:
                # Ensure proper PostgreSQL URL format for NeonDB (fix psycog -> psycopg issue)
                if db_url.startswith('postgresql://'):
                    db_url = db_url.replace('postgresql://', 'postgresql+psycopg://')
                elif db_url.startswith('postgres://'):
                    db_url = db_url.replace('postgres://', 'postgresql+psycopg://')
                elif 'psycog' in db_url:
                    # Fix common psycog typo in existing URLs
                    db_url = db_url.replace('psycog', 'psycopg')
                
                # Add connection parameters for better compatibility with NeonDB
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
                
                print(f"🔗 Connecting to NeonDB with Northwind schema...")
                self.engine = create_engine(db_url, echo=False)
                self.schema_name = "northwind"
                
                # Verify connection to Northwind schema
                with self.engine.connect() as conn:
                    # Verify we can access the northwind schema
                    result = conn.execute(text("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'northwind' 
                        LIMIT 1
                    """))
                    
                    if result.fetchone():
                        print(f"✅ Successfully connected to NeonDB Northwind schema!")
                        # Explicitly set search path to northwind
                        conn.execute(text("SET search_path TO northwind, public"))
                    else:
                        raise Exception("Northwind schema not found or empty")
                
                print(f"✅ Connected to PostgreSQL (schema: {self.schema_name})")
            else:
                raise Exception("No database URL")
                
        except Exception as e:
            print(f"⚠️ PostgreSQL failed: {e}")
            print("🔄 Using SQLite...")
            
            self._create_sample_db()
            self.engine = create_engine('sqlite:///sample.db')
            self.schema_name = "main"
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)
    
    def _create_sample_db(self):
        conn = sqlite3.connect('sample.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                company_name TEXT,
                city TEXT,
                country TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id TEXT,
                order_date DATE,
                amount REAL
            )
        ''')
        
        customers = [
            ('ALFKI', 'Alfreds Futterkiste', 'Berlin', 'Germany'),
            ('ANATR', 'Ana Trujillo', 'México D.F.', 'Mexico'),
            ('BERGS', 'Berglunds snabbköp', 'Luleå', 'Sweden')
        ]
        
        orders = [
            (1, 'ALFKI', '2024-01-15', 250.50),
            (2, 'ANATR', '2024-01-16', 180.75),
            (3, 'BERGS', '2024-01-17', 340.20)
        ]
        
        cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?)', customers)
        cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?)', orders)
        
        conn.commit()
        conn.close()
    
    def get_table_names(self):
        if self.schema_name != "main":
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = '{self.schema_name}'
                """))
                return [row[0] for row in result.fetchall()]
        else:
            return self.inspector.get_table_names()
    
    def get_schema_info(self):
        schema_info = {}
        table_names = self.get_table_names()
        
        for table_name in table_names:
            try:
                if self.schema_name != "main":
                    columns = self.inspector.get_columns(table_name, schema=self.schema_name)
                    foreign_keys = self.inspector.get_foreign_keys(table_name, schema=self.schema_name)
                else:
                    columns = self.inspector.get_columns(table_name)
                    foreign_keys = self.inspector.get_foreign_keys(table_name)
                
                schema_info[table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "primary_key": col.get("primary_key", False)
                        }
                        for col in columns
                    ],
                    "foreign_keys": foreign_keys
                }
            except Exception as e:
                print(f"Warning: {table_name}: {e}")
                continue
        
        return schema_info
    
    def execute_query(self, query: str):
        try:
            with self.engine.connect() as connection:
                if self.schema_name != "main" and self.schema_name != "public":
                    modified_query = self._add_schema_prefix(query)
                else:
                    modified_query = query
                
                result = connection.execute(text(modified_query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")
    
    def _add_schema_prefix(self, query):
        table_names = self.get_table_names()
        modified_query = query
        
        import re
        for table_name in table_names:
            pattern = r'\b' + re.escape(table_name) + r'\b'
            replacement = f"{self.schema_name}.{table_name}"
            modified_query = re.sub(pattern, replacement, modified_query, flags=re.IGNORECASE)
        
        return modified_query
    
    def get_sample_data(self, table_name: str, limit: int = 5):
        if self.schema_name != "main":
            query = f"SELECT * FROM {self.schema_name}.{table_name} LIMIT {limit}"
        else:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            raise Exception(f"Sample data query failed: {str(e)}")
    
    def get_schema_name(self):
        return self.schema_name