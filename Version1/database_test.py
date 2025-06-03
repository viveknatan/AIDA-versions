from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from typing import Dict, List, Any
import pandas as pd
import sqlite3
import os

class DatabaseManager:
    def __init__(self):
        # For testing, use SQLite if PostgreSQL connection fails
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
            
            # Test the connection and verify Northwind schema
            with self.engine.connect() as conn:
                # Verify we can access the northwind schema
                try:
                    # First check if northwind schema exists
                    result = conn.execute(text("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name = 'northwind'
                    """))
                    
                    if result.fetchone():
                        # Verify we can access tables in the northwind schema
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
                            print("⚠️ Northwind schema exists but no tables found")
                            raise Exception("Northwind schema is empty")
                    else:
                        raise Exception("Northwind schema not found")
                        
                except Exception as schema_error:
                    print(f"⚠️ Northwind schema issue: {schema_error}")
                    print("🔍 Checking available schemas...")
                    result = conn.execute(text("""
                        SELECT schema_name 
                        FROM information_schema.schemata 
                        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                        ORDER BY schema_name
                    """))
                    schemas = [row[0] for row in result.fetchall()]
                    print(f"Available schemas: {schemas}")
                    
                    if 'northwind' not in schemas:
                        raise Exception(f"Northwind schema not found in NeonDB. Available schemas: {schemas}")
                    else:
                        raise Exception("Cannot access Northwind schema tables")
            
        except Exception as e:
            print(f"⚠️ PostgreSQL connection failed: {e}")
            print("🔄 Falling back to SQLite with Northwind sample data...")
            
            # Create a test SQLite database with Northwind-style sample data
            self._create_northwind_sqlite_db()
            self.engine = create_engine('sqlite:///northwind_test.db')
            self.schema_name = "main"  # SQLite uses "main" as default schema
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.metadata = MetaData()
        self.inspector = inspect(self.engine)
    
    def _create_northwind_sqlite_db(self):
        """Create a test SQLite database with Northwind-style sample data"""
        conn = sqlite3.connect('northwind_test.db')
        cursor = conn.cursor()
        
        # Create Northwind-style tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                contact_name TEXT,
                contact_title TEXT,
                address TEXT,
                city TEXT,
                region TEXT,
                postal_code TEXT,
                country TEXT,
                phone TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                employee_id INTEGER PRIMARY KEY,
                last_name TEXT NOT NULL,
                first_name TEXT NOT NULL,
                title TEXT,
                title_of_courtesy TEXT,
                birth_date DATE,
                hire_date DATE,
                address TEXT,
                city TEXT,
                region TEXT,
                postal_code TEXT,
                country TEXT,
                home_phone TEXT,
                reports_to INTEGER,
                FOREIGN KEY (reports_to) REFERENCES employees (employee_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                supplier_id INTEGER,
                category_id INTEGER,
                quantity_per_unit TEXT,
                unit_price REAL,
                units_in_stock INTEGER,
                units_on_order INTEGER,
                reorder_level INTEGER,
                discontinued INTEGER,
                FOREIGN KEY (category_id) REFERENCES categories (category_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id TEXT,
                employee_id INTEGER,
                order_date DATE,
                required_date DATE,
                shipped_date DATE,
                ship_via INTEGER,
                freight REAL,
                ship_name TEXT,
                ship_address TEXT,
                ship_city TEXT,
                ship_region TEXT,
                ship_postal_code TEXT,
                ship_country TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_details (
                order_id INTEGER,
                product_id INTEGER,
                unit_price REAL,
                quantity INTEGER,
                discount REAL,
                PRIMARY KEY (order_id, product_id),
                FOREIGN KEY (order_id) REFERENCES orders (order_id),
                FOREIGN KEY (product_id) REFERENCES products (product_id)
            )
        ''')
        
        # Insert sample data
        customers_data = [
            ('ALFKI', 'Alfreds Futterkiste', 'Maria Anders', 'Sales Representative', 'Obere Str. 57', 'Berlin', None, '12209', 'Germany', '030-0074321'),
            ('ANATR', 'Ana Trujillo Emparedados y helados', 'Ana Trujillo', 'Owner', 'Avda. de la Constitución 2222', 'México D.F.', None, '05021', 'Mexico', '(5) 555-4729'),
            ('ANTON', 'Antonio Moreno Taquería', 'Antonio Moreno', 'Owner', 'Mataderos 2312', 'México D.F.', None, '05023', 'Mexico', '(5) 555-3932'),
            ('AROUT', 'Around the Horn', 'Thomas Hardy', 'Sales Representative', '120 Hanover Sq.', 'London', None, 'WA1 1DP', 'UK', '(171) 555-7788'),
            ('BERGS', 'Berglunds snabbköp', 'Christina Berglund', 'Order Administrator', 'Berguvsvägen 8', 'Luleå', None, 'S-958 22', 'Sweden', '0921-12 34 65')
        ]
        
        employees_data = [
            (1, 'Davolio', 'Nancy', 'Sales Representative', 'Ms.', '1948-12-08', '1992-05-01', '507 - 20th Ave. E.', 'Seattle', 'WA', '98122', 'USA', '(206) 555-9857', 2),
            (2, 'Fuller', 'Andrew', 'Vice President, Sales', 'Dr.', '1952-02-19', '1992-08-14', '908 W. Capital Way', 'Tacoma', 'WA', '98401', 'USA', '(206) 555-9482', None),
            (3, 'Leverling', 'Janet', 'Sales Representative', 'Ms.', '1963-08-30', '1992-04-01', '722 Moss Bay Blvd.', 'Kirkland', 'WA', '98033', 'USA', '(206) 555-3412', 2),
            (4, 'Peacock', 'Margaret', 'Sales Representative', 'Mrs.', '1937-09-19', '1993-05-03', '4110 Old Redmond Rd.', 'Redmond', 'WA', '98052', 'USA', '(206) 555-8122', 2),
            (5, 'Buchanan', 'Steven', 'Sales Manager', 'Mr.', '1955-03-04', '1993-10-17', '14 Garrett Hill', 'London', None, 'SW1 8JR', 'UK', '(71) 555-4848', 2)
        ]
        
        categories_data = [
            (1, 'Beverages', 'Soft drinks, coffees, teas, beers, and ales'),
            (2, 'Condiments', 'Sweet and savory sauces, relishes, spreads, and seasonings'),
            (3, 'Dairy Products', 'Cheeses'),
            (4, 'Grains/Cereals', 'Breads, crackers, pasta, and cereal'),
            (5, 'Meat/Poultry', 'Prepared meats'),
            (6, 'Produce', 'Dried fruit and bean curd'),
            (7, 'Seafood', 'Seaweed and fish')
        ]
        
        products_data = [
            (1, 'Chai', 1, 1, '10 boxes x 20 bags', 18.00, 39, 0, 10, 0),
            (2, 'Chang', 1, 1, '24 - 12 oz bottles', 19.00, 17, 40, 25, 0),
            (3, 'Aniseed Syrup', 1, 2, '12 - 550 ml bottles', 10.00, 13, 70, 25, 0),
            (4, 'Chef Antons Cajun Seasoning', 2, 2, '48 - 6 oz jars', 22.00, 53, 0, 0, 0),
            (5, 'Chef Antons Gumbo Mix', 2, 2, '36 boxes', 21.35, 0, 0, 0, 1)
        ]
        
        orders_data = [
            (10248, 'ALFKI', 5, '1996-07-04', '1996-08-01', '1996-07-16', 3, 32.38, 'Alfreds Futterkiste', 'Obere Str. 57', 'Berlin', None, '12209', 'Germany'),
            (10249, 'TOMSP', 6, '1996-07-05', '1996-08-16', '1996-07-10', 1, 11.61, 'Toms Spezialitäten', 'Luisenstr. 48', 'Münster', None, '44087', 'Germany'),
            (10250, 'HANAR', 4, '1996-07-08', '1996-08-05', '1996-07-12', 2, 65.83, 'Hanari Carnes', 'Rua do Paço, 67', 'Rio de Janeiro', 'RJ', '05454-876', 'Brazil')
        ]
        
        order_details_data = [
            (10248, 1, 14.00, 12, 0.0),
            (10248, 2, 9.80, 10, 0.0),
            (10248, 3, 34.80, 5, 0.0),
            (10249, 1, 14.00, 9, 0.0),
            (10249, 2, 9.80, 40, 0.0),
            (10250, 1, 7.70, 10, 0.0),
            (10250, 2, 35.10, 35, 0.15),
            (10250, 3, 25.89, 15, 0.15)
        ]
        
        cursor.executemany('INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', customers_data)
        cursor.executemany('INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', employees_data)
        cursor.executemany('INSERT OR REPLACE INTO categories VALUES (?, ?, ?)', categories_data)
        cursor.executemany('INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', products_data)
        cursor.executemany('INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', orders_data)
        cursor.executemany('INSERT OR REPLACE INTO order_details VALUES (?, ?, ?, ?, ?)', order_details_data)
        
        conn.commit()
        conn.close()
        print("📊 Created Northwind test database with sample data")
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information for the LLM"""
        schema_info = {}
        
        # Get table names using our helper method
        table_names = self.get_table_names()
        
        for table_name in table_names:
            try:
                if self.schema_name != "main":  # PostgreSQL
                    columns = self.inspector.get_columns(table_name, schema=self.schema_name)
                    foreign_keys = self.inspector.get_foreign_keys(table_name, schema=self.schema_name)
                else:  # SQLite
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
                print(f"Warning: Could not get info for table {table_name}: {e}")
                continue
        
        return schema_info
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            with self.engine.connect() as connection:
                # For PostgreSQL with schema (not SQLite), modify the query to include schema prefix
                if self.schema_name != "main" and self.schema_name != "public":
                    # Add schema prefix to table names in the query if not already present
                    modified_query = self._add_schema_prefix_to_query(query)
                else:
                    modified_query = query
                
                result = connection.execute(text(modified_query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
    
    def _add_schema_prefix_to_query(self, query: str) -> str:
        """Add schema prefix to table names in SQL query"""
        # Simple approach: if the query doesn't already have schema prefixes,
        # and we know our schema name, try to add it
        
        # Skip if query already has schema references or SET commands
        if (f"{self.schema_name}." in query.lower() or 
            "set " in query.lower() or 
            "information_schema" in query.lower()):
            return query
        
        # Get list of tables in our schema for replacement
        try:
            table_names = self.get_table_names()
            
            # Replace table names with schema-prefixed versions
            # This is a simple regex approach - could be made more sophisticated
            import re
            modified_query = query
            
            for table_name in table_names:
                # Look for table name as whole word (with word boundaries)
                pattern = r'\b' + re.escape(table_name) + r'\b'
                replacement = f"{self.schema_name}.{table_name}"
                modified_query = re.sub(pattern, replacement, modified_query, flags=re.IGNORECASE)
            
            return modified_query
            
        except Exception:
            # If anything fails, return original query
            return query
    
    def get_table_names(self) -> list:
        """Get list of table names in the current schema"""
        try:
            if self.schema_name != "main":  # PostgreSQL
                with self.engine.connect() as connection:
                    result = connection.execute(text(f"""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = '{self.schema_name}'
                    """))
                    return [row[0] for row in result.fetchall()]
            else:  # SQLite
                return self.inspector.get_table_names()
        except Exception:
            return []
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a table"""
        # Use explicit schema prefix for PostgreSQL
        if self.schema_name != "main":
            query = f"SELECT * FROM {self.schema_name}.{table_name} LIMIT {limit}"
        else:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
        
        # Use direct connection without setting search_path
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
    
    def get_schema_name(self) -> str:
        """Get the current schema name"""
        return self.schema_name