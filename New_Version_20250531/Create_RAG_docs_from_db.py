import pandas as pd
import psycopg2
from typing import List
from datetime import datetime

def create_comprehensive_northwind_business_documents(
    host: str,
    username: str,
    password: str,
    database: str = "neondb",
    port: int = 5432,
    schema: str = "northwind"
) -> List[str]:
    """
    Execute comprehensive SQL queries against PostgreSQL Northwind database hosted on Neon 
    and create detailed business-friendly documents for vector search and RAG applications.
    
    This expanded version includes deep analysis of all business aspects including:
    - Customer demographics and behavior patterns
    - Product performance and inventory management
    - Employee productivity and territory analysis
    - Supplier relationships and logistics
    - Financial performance and trends
    - Geographic distribution and shipping patterns
    """
    
    # Create connection string for Neon
    conn_string = f"postgresql://{username}:{password}@{host}:{port}/{database}?sslmode=require"
    
    documents = []
    
    try:
        print("Connecting to Northwind database and generating comprehensive business documents...")
        
        # 1. COMPREHENSIVE CUSTOMER ANALYSIS
        print("Generating customer analysis document...")
        
        # Customer demographics and distribution
        customer_demographics_query = f"""
        SELECT 
            country,
            city,
            COUNT(*) as customer_count,
            STRING_AGG(DISTINCT contact_title, ', ') as job_titles,
            STRING_AGG(company_name, '; ' ORDER BY company_name) as sample_companies
        FROM {schema}.customers 
        GROUP BY country, city
        ORDER BY customer_count DESC, country, city
        """
        
        df_demographics = pd.read_sql_query(customer_demographics_query, conn_string)
        
        customer_doc = "NORTHWIND COMPREHENSIVE CUSTOMER ANALYSIS:\n\n"
        customer_doc += f"CUSTOMER BASE OVERVIEW:\n"
        customer_doc += f"Northwind serves {df_demographics['customer_count'].sum()} customers across {len(df_demographics['country'].unique())} countries and {len(df_demographics)} cities.\n\n"
        
        # Country analysis
        country_summary = df_demographics.groupby('country').agg({
            'customer_count': 'sum',
            'city': 'count'
        }).sort_values('customer_count', ascending=False)
        
        customer_doc += "CUSTOMER DISTRIBUTION BY COUNTRY:\n"
        for country, row in country_summary.head(15).iterrows():
            customer_doc += f"- {country}: {row['customer_count']} customers across {row['city']} cities\n"
        
        # Detailed customer profiles with contact information
        detailed_customers_query = f"""
        SELECT 
            customer_id,
            company_name,
            contact_name,
            contact_title,
            city,
            region,
            country,
            phone,
            fax
        FROM {schema}.customers
        ORDER BY country, city, company_name
        """
        
        df_detailed = pd.read_sql_query(detailed_customers_query, conn_string)
        
        customer_doc += "\n\nDETAILED CUSTOMER DIRECTORY:\n"
        
        # Group by country for better organization
        for country in df_detailed['country'].unique()[:10]:  # Top 10 countries
            country_customers = df_detailed[df_detailed['country'] == country]
            customer_doc += f"\n{country.upper()} ({len(country_customers)} customers):\n"
            
            for _, customer in country_customers.head(8).iterrows():  # Top 8 per country
                region_info = f", {customer['region']}" if pd.notna(customer['region']) else ""
                fax_info = f", Fax: {customer['fax']}" if pd.notna(customer['fax']) else ""
                customer_doc += f"  • {customer['company_name']} - {customer['contact_name']} ({customer['contact_title']})\n"
                customer_doc += f"    Location: {customer['city']}{region_info}, Phone: {customer['phone']}{fax_info}\n"
        
        documents.append(customer_doc)
        
        # 2. CUSTOMER PURCHASING BEHAVIOR AND TOP PERFORMERS
        print("Generating customer purchasing behavior analysis...")
        
        customer_behavior_query = f"""
        WITH customer_metrics AS (
            SELECT 
                c.customer_id,
                c.company_name,
                c.contact_name,
                c.city,
                c.region,
                c.country,
                COUNT(DISTINCT o.order_id) as total_orders,
                COUNT(DISTINCT DATE_PART('year', o.order_date)) as years_active,
                MIN(o.order_date) as first_order_date,
                MAX(o.order_date) as last_order_date,
                ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) as total_revenue,
                ROUND(AVG(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) as avg_order_value,
                SUM(od.quantity) as total_items_purchased,
                ROUND(AVG(o.freight)::numeric, 2) as avg_shipping_cost,
                COUNT(DISTINCT od.product_id) as unique_products_bought
            FROM {schema}.customers c
            JOIN {schema}.orders o ON c.customer_id = o.customer_id
            JOIN {schema}.order_details od ON o.order_id = od.order_id
            GROUP BY c.customer_id, c.company_name, c.contact_name, c.city, c.region, c.country
        )
        SELECT * FROM customer_metrics
        ORDER BY total_revenue DESC
        """
        
        df_behavior = pd.read_sql_query(customer_behavior_query, conn_string)
        
        behavior_doc = "NORTHWIND CUSTOMER PURCHASING BEHAVIOR ANALYSIS:\n\n"
        
        # Top customers by revenue
        behavior_doc += "TOP 20 CUSTOMERS BY TOTAL REVENUE:\n"
        for i, row in df_behavior.head(20).iterrows():
            region = f", {row['region']}" if pd.notna(row['region']) else ""
            years_span = f"{row['first_order_date']:.10}" + " to " + f"{row['last_order_date']:.10}"
            
            behavior_doc += f"{i+1}. {row['company_name']} ({row['country']})\n"
            behavior_doc += f"   • Total Revenue: ${row['total_revenue']:,.2f} across {row['total_orders']} orders\n"
            behavior_doc += f"   • Average Order Value: ${row['avg_order_value']:,.2f}\n"
            behavior_doc += f"   • Active Period: {years_span} ({row['years_active']} years)\n"
            behavior_doc += f"   • Location: {row['city']}{region}\n"
            behavior_doc += f"   • Products Diversity: {row['unique_products_bought']} different products, {row['total_items_purchased']} total items\n\n"
        
        # Customer segmentation analysis
        behavior_doc += "CUSTOMER SEGMENTATION ANALYSIS:\n"
        revenue_percentiles = df_behavior['total_revenue'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
        
        behavior_doc += f"• Premium Customers (Top 5%): ${revenue_percentiles[0.95]:,.2f}+ revenue ({len(df_behavior[df_behavior['total_revenue'] >= revenue_percentiles[0.95]])} customers)\n"
        behavior_doc += f"• High-Value Customers (Top 10%): ${revenue_percentiles[0.9]:,.2f}+ revenue ({len(df_behavior[df_behavior['total_revenue'] >= revenue_percentiles[0.9]])} customers)\n"
        behavior_doc += f"• Regular Customers (Median): ${revenue_percentiles[0.5]:,.2f} revenue\n"
        behavior_doc += f"• Average Order Value Range: ${df_behavior['avg_order_value'].min():,.2f} - ${df_behavior['avg_order_value'].max():,.2f}\n"
        
        # Geographic revenue distribution
        geographic_revenue = df_behavior.groupby('country').agg({
            'total_revenue': 'sum',
            'total_orders': 'sum',
            'company_name': 'count'
        }).sort_values('total_revenue', ascending=False)
        
        behavior_doc += "\nREVENUE BY COUNTRY:\n"
        for country, row in geographic_revenue.head(10).iterrows():
            avg_revenue_per_customer = row['total_revenue'] / row['company_name']
            behavior_doc += f"• {country}: ${row['total_revenue']:,.2f} total (${avg_revenue_per_customer:,.2f} avg per customer)\n"
        
        documents.append(behavior_doc)
        
        # 3. COMPREHENSIVE PRODUCT CATALOG AND PERFORMANCE
        print("Generating comprehensive product analysis...")
        
        product_analysis_query = f"""
        WITH product_performance AS (
            SELECT 
                p.product_id,
                p.product_name,
                c.category_name,
                s.company_name as supplier_name,
                s.country as supplier_country,
                p.quantity_per_unit,
                p.unit_price,
                p.units_in_stock,
                p.units_on_order,
                p.reorder_level,
                p.discontinued,
                COALESCE(SUM(od.quantity), 0) as total_quantity_sold,
                COALESCE(ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2), 0) as total_revenue,
                COALESCE(COUNT(DISTINCT od.order_id), 0) as orders_count,
                COALESCE(ROUND(AVG(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2), 0) as avg_order_line_value
            FROM {schema}.products p
            JOIN {schema}.categories c ON p.category_id = c.category_id
            JOIN {schema}.suppliers s ON p.supplier_id = s.supplier_id
            LEFT JOIN {schema}.order_details od ON p.product_id = od.product_id
            GROUP BY p.product_id, p.product_name, c.category_name, s.company_name, s.country,
                     p.quantity_per_unit, p.unit_price, p.units_in_stock, p.units_on_order, 
                     p.reorder_level, p.discontinued
        )
        SELECT * FROM product_performance
        ORDER BY total_revenue DESC
        """
        
        df_products = pd.read_sql_query(product_analysis_query, conn_string)
        
        product_doc = "NORTHWIND COMPREHENSIVE PRODUCT CATALOG AND PERFORMANCE:\n\n"
        
        # Category overview
        category_performance = df_products.groupby('category_name').agg({
            'product_id': 'count',
            'total_revenue': 'sum',
            'total_quantity_sold': 'sum',
            'unit_price': 'mean',
            'units_in_stock': 'sum'
        }).sort_values('total_revenue', ascending=False)
        
        product_doc += f"PRODUCT PORTFOLIO OVERVIEW:\n"
        product_doc += f"Total Products: {len(df_products)} across {len(category_performance)} categories\n"
        product_doc += f"Active Products: {len(df_products[df_products['discontinued'] == 0])}\n"
        product_doc += f"Discontinued Products: {len(df_products[df_products['discontinued'] == 1])}\n\n"
        
        product_doc += "CATEGORY PERFORMANCE ANALYSIS:\n"
        for category, row in category_performance.iterrows():
            product_doc += f"• {category}: {row['product_id']} products, ${row['total_revenue']:,.2f} revenue\n"
            product_doc += f"  - {row['total_quantity_sold']:,.0f} units sold, avg price: ${row['unit_price']:,.2f}\n"
            product_doc += f"  - Current inventory: {row['units_in_stock']:,.0f} units\n"
        
        # Top performing products
        product_doc += "\nTOP 25 PRODUCTS BY REVENUE:\n"
        for i, row in df_products.head(25).iterrows():
            discontinued_status = " [DISCONTINUED]" if row['discontinued'] == 1 else ""
            stock_status = "⚠️ LOW STOCK" if row['units_in_stock'] <= row['reorder_level'] else "✅ IN STOCK"
            
            product_doc += f"{i+1}. {row['product_name']} ({row['category_name']}){discontinued_status}\n"
            product_doc += f"   • Revenue: ${row['total_revenue']:,.2f} from {row['total_quantity_sold']} units sold\n"
            product_doc += f"   • Price: ${row['unit_price']:.2f} per {row['quantity_per_unit']}\n"
            product_doc += f"   • Supplier: {row['supplier_name']} ({row['supplier_country']})\n"
            product_doc += f"   • Inventory: {row['units_in_stock']} in stock, {row['units_on_order']} on order ({stock_status})\n\n"
        
        # Inventory management insights
        low_stock_products = df_products[df_products['units_in_stock'] <= df_products['reorder_level']]
        high_revenue_low_stock = low_stock_products[low_stock_products['total_revenue'] > df_products['total_revenue'].median()]
        
        product_doc += f"INVENTORY MANAGEMENT ALERTS:\n"
        product_doc += f"• Products below reorder level: {len(low_stock_products)}\n"
        product_doc += f"• High-revenue products with low stock: {len(high_revenue_low_stock)}\n"
        
        if len(high_revenue_low_stock) > 0:
            product_doc += "  Critical reorder needed for:\n"
            for _, product in high_revenue_low_stock.head(5).iterrows():
                product_doc += f"    - {product['product_name']}: {product['units_in_stock']} units (${product['total_revenue']:,.0f} revenue)\n"
        
        documents.append(product_doc)
        
        # 4. SUPPLIER RELATIONSHIPS AND LOGISTICS
        print("Generating supplier analysis...")
        
        supplier_analysis_query = f"""
        WITH supplier_metrics AS (
            SELECT 
                s.supplier_id,
                s.company_name,
                s.contact_name,
                s.contact_title,
                s.city,
                s.region,
                s.country,
                s.phone,
                s.fax,
                COUNT(p.product_id) as products_supplied,
                COUNT(CASE WHEN p.discontinued = 0 THEN 1 END) as active_products,
                ROUND(AVG(p.unit_price)::numeric, 2) as avg_product_price,
                SUM(p.units_in_stock) as total_inventory_units,
                COALESCE(SUM(od.quantity), 0) as total_units_sold,
                COALESCE(ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2), 0) as total_revenue_generated
            FROM {schema}.suppliers s
            JOIN {schema}.products p ON s.supplier_id = p.supplier_id
            LEFT JOIN {schema}.order_details od ON p.product_id = od.product_id
            GROUP BY s.supplier_id, s.company_name, s.contact_name, s.contact_title,
                     s.city, s.region, s.country, s.phone, s.fax
        )
        SELECT * FROM supplier_metrics
        ORDER BY total_revenue_generated DESC
        """
        
        df_suppliers = pd.read_sql_query(supplier_analysis_query, conn_string)
        
        supplier_doc = "NORTHWIND SUPPLIER RELATIONSHIP AND LOGISTICS ANALYSIS:\n\n"
        
        # Supplier overview
        supplier_doc += f"SUPPLIER NETWORK OVERVIEW:\n"
        supplier_doc += f"Total Suppliers: {len(df_suppliers)}\n"
        supplier_doc += f"Geographic Distribution: {len(df_suppliers['country'].unique())} countries\n"
        supplier_doc += f"Total Products Supplied: {df_suppliers['products_supplied'].sum()}\n"
        supplier_doc += f"Active Products: {df_suppliers['active_products'].sum()}\n\n"
        
        # Supplier performance ranking
        supplier_doc += "TOP SUPPLIERS BY REVENUE GENERATION:\n"
        for i, row in df_suppliers.head(15).iterrows():
            region_info = f", {row['region']}" if pd.notna(row['region']) else ""
            fax_info = f", Fax: {row['fax']}" if pd.notna(row['fax']) else ""
            
            supplier_doc += f"{i+1}. {row['company_name']} ({row['country']})\n"
            supplier_doc += f"   • Contact: {row['contact_name']} ({row['contact_title']})\n"
            supplier_doc += f"   • Location: {row['city']}{region_info}\n"
            supplier_doc += f"   • Phone: {row['phone']}{fax_info}\n"
            supplier_doc += f"   • Products: {row['products_supplied']} total ({row['active_products']} active)\n"
            supplier_doc += f"   • Revenue Generated: ${row['total_revenue_generated']:,.2f}\n"
            supplier_doc += f"   • Units Sold: {row['total_units_sold']:,.0f}, Avg Product Price: ${row['avg_product_price']:,.2f}\n\n"
        
        # Geographic supplier distribution
        supplier_by_country = df_suppliers.groupby('country').agg({
            'supplier_id': 'count',
            'products_supplied': 'sum',
            'total_revenue_generated': 'sum'
        }).sort_values('total_revenue_generated', ascending=False)
        
        supplier_doc += "SUPPLIER GEOGRAPHIC DISTRIBUTION:\n"
        for country, row in supplier_by_country.iterrows():
            avg_revenue_per_supplier = row['total_revenue_generated'] / row['supplier_id']
            supplier_doc += f"• {country}: {row['supplier_id']} suppliers, {row['products_supplied']} products, ${row['total_revenue_generated']:,.2f} revenue\n"
            supplier_doc += f"  Average revenue per supplier: ${avg_revenue_per_supplier:,.2f}\n"
        
        documents.append(supplier_doc)
        
        # 5. EMPLOYEE PERFORMANCE AND TERRITORY ANALYSIS
        print("Generating employee and territory analysis...")
        
        employee_analysis_query = f"""
        WITH employee_performance AS (
            SELECT 
                e.employee_id,
                e.first_name || ' ' || e.last_name as full_name,
                e.title,
                e.title_of_courtesy,
                e.birth_date,
                e.hire_date,
                e.city,
                e.region,
                e.country,
                e.home_phone,
                e.reports_to,
                mgr.first_name || ' ' || mgr.last_name as manager_name,
                COUNT(DISTINCT o.order_id) as orders_handled,
                COUNT(DISTINCT o.customer_id) as customers_served,
                COALESCE(ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2), 0) as total_sales,
                COALESCE(ROUND(AVG(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2), 0) as avg_order_value,
                COALESCE(SUM(od.quantity), 0) as total_units_sold,
                COUNT(DISTINCT DATE_PART('year', o.order_date)) as years_active,
                MIN(o.order_date) as first_sale_date,
                MAX(o.order_date) as last_sale_date
            FROM {schema}.employees e
            LEFT JOIN {schema}.employees mgr ON e.reports_to = mgr.employee_id
            LEFT JOIN {schema}.orders o ON e.employee_id = o.employee_id
            LEFT JOIN {schema}.order_details od ON o.order_id = od.order_id
            GROUP BY e.employee_id, e.first_name, e.last_name, e.title, e.title_of_courtesy,
                     e.birth_date, e.hire_date, e.city, e.region, e.country, e.home_phone,
                     e.reports_to, mgr.first_name, mgr.last_name
        )
        SELECT * FROM employee_performance
        ORDER BY total_sales DESC
        """
        
        df_employees = pd.read_sql_query(employee_analysis_query, conn_string)
        
        employee_doc = "NORTHWIND EMPLOYEE PERFORMANCE AND ORGANIZATIONAL ANALYSIS:\n\n"
        
        # Organizational overview
        employee_doc += f"ORGANIZATIONAL STRUCTURE:\n"
        employee_doc += f"Total Employees: {len(df_employees)}\n"
        
        # Management hierarchy
        managers = df_employees[df_employees['reports_to'].isna()]
        subordinates = df_employees[df_employees['reports_to'].notna()]
        
        employee_doc += f"Management Level: {len(managers)} managers\n"
        employee_doc += f"Staff Level: {len(subordinates)} employees\n\n"
        
        # Employee performance ranking
        employee_doc += "EMPLOYEE SALES PERFORMANCE RANKING:\n"
        for i, row in df_employees.iterrows():
            manager_info = f" (Reports to: {row['manager_name']})" if pd.notna(row['manager_name']) else " (Senior Management)"
            years_service = datetime.now().year - pd.to_datetime(row['hire_date']).year if pd.notna(row['hire_date']) else 0
            age = datetime.now().year - pd.to_datetime(row['birth_date']).year if pd.notna(row['birth_date']) else 0
            
            employee_doc += f"{i+1}. {row['full_name']} - {row['title']}\n"
            employee_doc += f"   • Total Sales: ${row['total_sales']:,.2f} across {row['orders_handled']} orders\n"
            employee_doc += f"   • Customers Served: {row['customers_served']}, Avg Order Value: ${row['avg_order_value']:,.2f}\n"
            employee_doc += f"   • Service Period: {years_service} years (hired {str(row['hire_date'])[:10]})\n"
            employee_doc += f"   • Age: {age}, Location: {row['city']}, {row['country']}\n"
            employee_doc += f"   • Contact: {row['home_phone']}{manager_info}\n\n"
        
        # Performance metrics analysis
        total_company_sales = df_employees['total_sales'].sum()
        top_performer = df_employees.iloc[0]
        
        employee_doc += "PERFORMANCE INSIGHTS:\n"
        employee_doc += f"• Top Performer: {top_performer['full_name']} (${top_performer['total_sales']:,.2f} - {(top_performer['total_sales']/total_company_sales*100):.1f}% of total sales)\n"
        employee_doc += f"• Average Sales per Employee: ${df_employees['total_sales'].mean():,.2f}\n"
        employee_doc += f"• Sales Performance Range: ${df_employees['total_sales'].min():,.2f} - ${df_employees['total_sales'].max():,.2f}\n"
        employee_doc += f"• Average Customer Base per Employee: {df_employees['customers_served'].mean():.1f} customers\n"
        
        documents.append(employee_doc)
        
        # 6. SHIPPING AND LOGISTICS ANALYSIS
        print("Generating shipping and logistics analysis...")
        
        shipping_analysis_query = f"""
        WITH shipping_metrics AS (
    SELECT 
        sh.shipper_id,
        sh.company_name as shipper_name,
        sh.phone as shipper_phone,
        COUNT(o.order_id) as total_shipments,
        COUNT(DISTINCT o.customer_id) as customers_served,
        COUNT(DISTINCT o.ship_country) as countries_served,
        ROUND(AVG(o.freight)::numeric, 2) as avg_freight_cost,
        ROUND(SUM(o.freight)::numeric, 2) as total_freight_revenue,
        ROUND(AVG((o.shipped_date::date - o.order_date::date))::numeric, 1) as avg_delivery_days,
        COUNT(CASE WHEN o.shipped_date::date > o.required_date::date THEN 1 END) as late_deliveries,
        ROUND((COUNT(CASE WHEN o.shipped_date::date > o.required_date::date THEN 1 END) * 100.0 / COUNT(o.order_id))::numeric, 2) as late_delivery_rate
    FROM {schema}.shippers sh
    LEFT JOIN {schema}.orders o ON sh.shipper_id = o.ship_via
    WHERE o.shipped_date IS NOT NULL AND o.order_date IS NOT NULL
    GROUP BY sh.shipper_id, sh.company_name, sh.phone
),
route_analysis AS (
    SELECT 
        o.ship_country,
        o.ship_city,
        COUNT(o.order_id) as shipment_count,
        ROUND(AVG(o.freight)::numeric, 2) as avg_freight_cost,
        COUNT(DISTINCT o.customer_id) as customers_in_location
    FROM {schema}.orders o
    WHERE o.shipped_date IS NOT NULL
    GROUP BY o.ship_country, o.ship_city
)
SELECT 
    sm.*,
    (SELECT COUNT(*) FROM route_analysis) as total_shipping_locations
FROM shipping_metrics sm
ORDER BY sm.total_freight_revenue DESC;
        """
        
        df_shipping = pd.read_sql_query(shipping_analysis_query, conn_string)
        
        # Route analysis
        route_query = f"""
        SELECT 
            ship_country,
            ship_city,
            COUNT(order_id) as shipment_count,
            ROUND(AVG(freight)::numeric, 2) as avg_freight_cost,
            COUNT(DISTINCT customer_id) as customers_in_location
        FROM {schema}.orders
        WHERE shipped_date IS NOT NULL
        GROUP BY ship_country, ship_city
        ORDER BY shipment_count DESC
        """
        
        df_routes = pd.read_sql_query(route_query, conn_string)
        
        shipping_doc = "NORTHWIND SHIPPING AND LOGISTICS PERFORMANCE ANALYSIS:\n\n"
        
        # Shipping company performance
        shipping_doc += "SHIPPING PARTNER PERFORMANCE:\n"
        for _, row in df_shipping.iterrows():
            on_time_rate = 100 - row['late_delivery_rate']
            
            shipping_doc += f"• {row['shipper_name']} (Phone: {row['shipper_phone']})\n"
            shipping_doc += f"  - Total Shipments: {row['total_shipments']:,} to {row['customers_served']} customers\n"
            shipping_doc += f"  - Coverage: {row['countries_served']} countries\n"
            shipping_doc += f"  - Freight Revenue: ${row['total_freight_revenue']:,.2f} (Avg: ${row['avg_freight_cost']:.2f} per shipment)\n"
            shipping_doc += f"  - Delivery Performance: {row['avg_delivery_days']:.1f} days average, {on_time_rate:.1f}% on-time rate\n"
            shipping_doc += f"  - Late Deliveries: {row['late_deliveries']} ({row['late_delivery_rate']:.1f}%)\n\n"
        
        # Geographic shipping analysis
        shipping_doc += "TOP SHIPPING DESTINATIONS:\n"
        country_routes = df_routes.groupby('ship_country').agg({
            'shipment_count': 'sum',
            'avg_freight_cost': 'mean',
            'customers_in_location': 'sum',
            'ship_city': 'count'
        }).sort_values('shipment_count', ascending=False)
        
        for country, row in country_routes.head(15).iterrows():
            shipping_doc += f"• {country}: {row['shipment_count']} shipments to {row['ship_city']} cities\n"
            shipping_doc += f"  - {row['customers_in_location']} customers, avg freight: ${row['avg_freight_cost']:.2f}\n"
        
        shipping_doc += "\nTOP CITY DESTINATIONS:\n"
        for _, row in df_routes.head(20).iterrows():
            shipping_doc += f"• {row['ship_city']}, {row['ship_country']}: {row['shipment_count']} shipments\n"
            shipping_doc += f"  - {row['customers_in_location']} customers, avg freight: ${row['avg_freight_cost']:.2f}\n"
        
        documents.append(shipping_doc)
        
        # 7. FINANCIAL PERFORMANCE AND TRENDS ANALYSIS
        print("Generating comprehensive financial analysis...")
        
        financial_analysis_query = f"""
        WITH monthly_performance AS (
            SELECT 
                EXTRACT(year FROM o.order_date) as year,
                EXTRACT(month FROM o.order_date) as month,
                COUNT(DISTINCT o.order_id) as order_count,
                COUNT(DISTINCT o.customer_id) as active_customers,
                COUNT(DISTINCT od.product_id) as products_sold,
                SUM(od.quantity) as units_sold,
                ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) as revenue,
                ROUND(AVG(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) as avg_order_line_value,
                ROUND(SUM(o.freight)::numeric, 2) as total_freight,
                ROUND(SUM(od.unit_price * od.quantity * od.discount)::numeric, 2) as total_discounts,
                ROUND(AVG(od.discount * 100)::numeric, 2) as avg_discount_percentage
            FROM {schema}.orders o
            JOIN {schema}.order_details od ON o.order_id = od.order_id
            WHERE o.order_date IS NOT NULL
            GROUP BY EXTRACT(year FROM o.order_date), EXTRACT(month FROM o.order_date)
        ),
        quarterly_performance AS (
            SELECT 
                year,
                CASE 
                    WHEN month IN (1,2,3) THEN 'Q1'
                    WHEN month IN (4,5,6) THEN 'Q2'
                    WHEN month IN (7,8,9) THEN 'Q3'
                    ELSE 'Q4'
                END as quarter,
                SUM(order_count) as orders,
                SUM(revenue) as quarterly_revenue,
                AVG(active_customers) as avg_monthly_customers,
                SUM(units_sold) as total_units,
                SUM(total_freight) as freight_revenue
            FROM monthly_performance
            GROUP BY year, CASE 
                WHEN month IN (1,2,3) THEN 'Q1'
                WHEN month IN (4,5,6) THEN 'Q2'
                WHEN month IN (7,8,9) THEN 'Q3'
                ELSE 'Q4'
            END
        )
        SELECT * FROM monthly_performance
        ORDER BY year, month
        """
        
        df_financial = pd.read_sql_query(financial_analysis_query, conn_string)
        
        financial_doc = "NORTHWIND COMPREHENSIVE FINANCIAL PERFORMANCE ANALYSIS:\n\n"
        
        # Overall financial summary
        total_revenue = df_financial['revenue'].sum()
        total_orders = df_financial['order_count'].sum()
        total_discounts = df_financial['total_discounts'].sum()
        
        financial_doc += f"FINANCIAL OVERVIEW:\n"
        financial_doc += f"• Total Revenue: ${total_revenue:,.2f}\n"
        financial_doc += f"• Total Orders: {total_orders:,}\n"
        financial_doc += f"• Total Discounts Given: ${total_discounts:,.2f} ({(total_discounts/total_revenue*100):.1f}% of revenue)\n"
        financial_doc += f"• Average Order Value: ${(total_revenue/total_orders):,.2f}\n"
        financial_doc += f"• Total Units Sold: {df_financial['units_sold'].sum():,}\n\n"
        
        # Yearly performance
        yearly_summary = df_financial.groupby('year').agg({
            'revenue': 'sum',
            'order_count': 'sum',
            'active_customers': 'mean',
            'units_sold': 'sum',
            'total_freight': 'sum',
            'total_discounts': 'sum'
        }).round(2)
        
        financial_doc += "ANNUAL PERFORMANCE BREAKDOWN:\n"
        for year, row in yearly_summary.iterrows():
            year_growth = ""
            if year > yearly_summary.index.min():
                prev_year_revenue = yearly_summary.loc[year-1, 'revenue']
                growth_rate = ((row['revenue'] - prev_year_revenue) / prev_year_revenue * 100)
                year_growth = f" ({growth_rate:+.1f}% vs prior year)"
            
            financial_doc += f"• {int(year)}: ${row['revenue']:,.2f} revenue{year_growth}\n"
            financial_doc += f"  - {int(row['order_count']):,} orders from {row['active_customers']:.0f} avg monthly customers\n"
            financial_doc += f"  - {int(row['units_sold']):,} units sold, ${row['total_freight']:,.2f} freight revenue\n"
            financial_doc += f"  - ${row['total_discounts']:,.2f} in discounts ({(row['total_discounts']/row['revenue']*100):.1f}%)\n"
        
        # Monthly trends analysis
        financial_doc += "\nMONTHLY PERFORMANCE TRENDS:\n"
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_avg = df_financial.groupby('month').agg({
            'revenue': 'mean',
            'order_count': 'mean',
            'active_customers': 'mean'
        }).round(2)
        
        # Find best and worst performing months
        best_month = monthly_avg['revenue'].idxmax()
        worst_month = monthly_avg['revenue'].idxmin()
        
        financial_doc += f"• Best Month: {month_names[int(best_month)-1]} (${monthly_avg.loc[best_month, 'revenue']:,.2f} avg revenue)\n"
        financial_doc += f"• Weakest Month: {month_names[int(worst_month)-1]} (${monthly_avg.loc[worst_month, 'revenue']:,.2f} avg revenue)\n"
        financial_doc += f"• Seasonal Variance: {((monthly_avg['revenue'].max() - monthly_avg['revenue'].min()) / monthly_avg['revenue'].mean() * 100):.1f}%\n\n"
        
        financial_doc += "MONTHLY AVERAGE PERFORMANCE:\n"
        for month, row in monthly_avg.sort_values('revenue', ascending=False).iterrows():
            month_name = month_names[int(month)-1]
            financial_doc += f"• {month_name}: ${row['revenue']:,.2f} revenue, {row['order_count']:.0f} orders, {row['active_customers']:.0f} customers\n"
        
        # Discount analysis
        financial_doc += f"\nDISCOUNT STRATEGY ANALYSIS:\n"
        financial_doc += f"• Average Discount Rate: {df_financial['avg_discount_percentage'].mean():.1f}%\n"
        financial_doc += f"• Total Discount Impact: ${total_discounts:,.2f} ({(total_discounts/(total_revenue+total_discounts)*100):.1f}% of gross revenue)\n"
        financial_doc += f"• Revenue Recovery Ratio: {((total_revenue/total_discounts) if total_discounts > 0 else 0):.1f}:1\n"
        
        documents.append(financial_doc)
        
        # 8. ADVANCED BUSINESS INTELLIGENCE AND INSIGHTS
        print("Generating advanced business intelligence insights...")
        
        # Customer loyalty and retention analysis
        loyalty_analysis_query = f"""
WITH customer_behavior AS (
    SELECT 
        c.customer_id,
        c.company_name,
        c.country,
        COUNT(DISTINCT o.order_id) as total_orders,
        COUNT(DISTINCT EXTRACT(year FROM o.order_date::date)) as years_active,
        COUNT(DISTINCT EXTRACT(month FROM o.order_date::date)) as months_active,
        MIN(o.order_date::date) as first_order,
        MAX(o.order_date::date) as last_order,
        SUM(od.quantity) as total_items,
        COUNT(DISTINCT od.product_id) as product_variety,
        ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) as total_spent,
        -- Calculate customer lifetime in days instead
        MAX(o.order_date::date) - MIN(o.order_date::date) as customer_lifetime_days
    FROM northwind.customers c
    JOIN northwind.orders o ON c.customer_id = o.customer_id
    JOIN northwind.order_details od ON o.order_id = od.order_id
    WHERE o.order_date IS NOT NULL
    GROUP BY c.customer_id, c.company_name, c.country
)
SELECT *,
    CASE 
        WHEN total_orders >= 10 AND years_active >= 2 THEN 'Loyal'
        WHEN total_orders >= 5 AND years_active >= 1 THEN 'Regular'
        WHEN total_orders >= 3 THEN 'Developing'
        ELSE 'New'
    END as customer_segment,
    -- Calculate average days between orders as total lifetime / (orders - 1)
    CASE 
        WHEN total_orders > 1 THEN ROUND((customer_lifetime_days::numeric / (total_orders - 1)), 1)
        ELSE NULL
    END as avg_days_between_orders
FROM customer_behavior
ORDER BY total_spent DESC;
        """
        
        df_loyalty = pd.read_sql_query(loyalty_analysis_query, conn_string)
        
        # Product affinity analysis
        affinity_query = f"""
        WITH product_pairs AS (
            SELECT 
                od1.product_id as product_a,
                od2.product_id as product_b,
                COUNT(*) as co_occurrence
            FROM {schema}.order_details od1
            JOIN {schema}.order_details od2 ON od1.order_id = od2.order_id
            WHERE od1.product_id < od2.product_id
            GROUP BY od1.product_id, od2.product_id
            HAVING COUNT(*) >= 3
        )
        SELECT 
            pp.product_a,
            p1.product_name as product_a_name,
            pp.product_b,
            p2.product_name as product_b_name,
            pp.co_occurrence
        FROM product_pairs pp
        JOIN {schema}.products p1 ON pp.product_a = p1.product_id
        JOIN {schema}.products p2 ON pp.product_b = p2.product_id
        ORDER BY pp.co_occurrence DESC
        LIMIT 20
        """
        
        df_affinity = pd.read_sql_query(affinity_query, conn_string)
        
        insights_doc = "NORTHWIND ADVANCED BUSINESS INTELLIGENCE AND STRATEGIC INSIGHTS:\n\n"
        
        # Customer segmentation analysis
        segment_analysis = df_loyalty['customer_segment'].value_counts()
        
        insights_doc += "CUSTOMER LOYALTY SEGMENTATION:\n"
        for segment, count in segment_analysis.items():
            segment_customers = df_loyalty[df_loyalty['customer_segment'] == segment]
            avg_spend = segment_customers['total_spent'].mean()
            avg_orders = segment_customers['total_orders'].mean()
            
            insights_doc += f"• {segment} Customers: {count} ({(count/len(df_loyalty)*100):.1f}%)\n"
            insights_doc += f"  - Average Spend: ${avg_spend:,.2f}\n"
            insights_doc += f"  - Average Orders: {avg_orders:.1f}\n"
            insights_doc += f"  - Example: {segment_customers.iloc[0]['company_name']}\n"
        
        # High-value customer analysis
        insights_doc += "\nHIGH-VALUE CUSTOMER PROFILE:\n"
        top_customers = df_loyalty.head(10)
        insights_doc += f"• Top 10 customers represent ${top_customers['total_spent'].sum():,.2f} ({(top_customers['total_spent'].sum()/df_loyalty['total_spent'].sum()*100):.1f}% of total revenue)\n"
        insights_doc += f"• Average order frequency: {top_customers['avg_days_between_orders'].mean():.0f} days between orders\n"
        insights_doc += f"• Product diversity: {top_customers['product_variety'].mean():.0f} different products per customer\n"
        
        insights_doc += "\nTOP 10 MOST VALUABLE CUSTOMERS:\n"
        for i, row in top_customers.iterrows():
            customer_tenure = f"{row['first_order']:.10} to {row['last_order']:.10}"
            insights_doc += f"{i+1}. {row['company_name']} ({row['country']})\n"
            insights_doc += f"   • Total Value: ${row['total_spent']:,.2f} over {row['total_orders']} orders\n"
            insights_doc += f"   • Tenure: {customer_tenure} ({row['years_active']} years)\n"
            insights_doc += f"   • Behavior: {row['product_variety']} different products, avg {row['avg_days_between_orders']:.0f} days between orders\n"
            insights_doc += f"   • Segment: {row['customer_segment']}\n\n"
        
        # Product affinity insights
        insights_doc += "PRODUCT AFFINITY ANALYSIS (Frequently Bought Together):\n"
        insights_doc += "Products commonly purchased together can inform cross-selling strategies:\n\n"
        
        for i, row in df_affinity.head(15).iterrows():
            insights_doc += f"• {row['product_a_name']} + {row['product_b_name']}\n"
            insights_doc += f"  Purchased together in {row['co_occurrence']} orders\n"
        
        # Business recommendations
        insights_doc += "\nSTRATEGIC BUSINESS RECOMMENDATIONS:\n"
        
        # Revenue concentration analysis
        top_20_pct_customers = len(df_loyalty) // 5
        top_20_revenue = df_loyalty.head(top_20_pct_customers)['total_spent'].sum()
        total_revenue_check = df_loyalty['total_spent'].sum()
        
        insights_doc += f"• Revenue Concentration: Top 20% of customers generate ${top_20_revenue:,.2f} ({(top_20_revenue/total_revenue_check*100):.1f}% of revenue)\n"
        insights_doc += f"• Customer Retention: Focus on {segment_analysis['Loyal']} loyal customers who drive consistent revenue\n"
        insights_doc += f"• Growth Opportunity: {segment_analysis['Developing']} developing customers show potential for increased engagement\n"
        
        # Seasonal insights
        peak_months = df_financial.groupby('month')['revenue'].mean().nlargest(3)
        insights_doc += f"• Seasonal Strategy: Peak sales months are {', '.join([month_names[int(m)-1] for m in peak_months.index])}\n"
        
        # Geographic insights
        country_performance = df_loyalty.groupby('country').agg({
            'total_spent': 'sum',
            'customer_id': 'count'
        }).sort_values('total_spent', ascending=False)
        
        top_country = country_performance.index[0]
        insights_doc += f"• Geographic Focus: {top_country} is the top market with ${country_performance.loc[top_country, 'total_spent']:,.2f} from {country_performance.loc[top_country, 'customer_id']} customers\n"
        
        documents.append(insights_doc)
# /*        
#         # 9. OPERATIONAL EFFICIENCY AND INVENTORY INSIGHTS
#         print("Generating operational efficiency analysis...")
        
#         operational_query = f"""
#         WITH order_processing AS (
#         SELECT 
#             o.order_id,
#             o.customer_id,
#             o.employee_id,
#             o.order_date::date,
#             o.required_date::date,
#             o.shipped_date::date,
#             (o.shipped_date::date - o.order_date::date) as processing_days,
#             (o.required_date::date - o.order_date::date) as promised_delivery_days,
#             CASE WHEN o.shipped_date::date > o.required_date::date THEN 1 ELSE 0 END as late_delivery,
#             o.freight,
#             o.ship_country,
#             COUNT(od.product_id) as items_in_order,
#             SUM(od.quantity) as total_quantity,
#             ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) as order_value
#         FROM {schema}.orders o
#         JOIN {schema}.order_details od ON o.order_id = od.order_id
#         WHERE o.shipped_date IS NOT NULL AND o.order_date IS NOT NULL
#         GROUP BY o.order_id, o.customer_id, o.employee_id, o.order_date, o.required_date, o.shipped_date, o.freight, o.ship_country
#         ),
#         inventory_turnover AS (
#         SELECT 
#             p.product_id,
#             p.product_name,
#             c.category_name,
#             p.units_in_stock,
#             p.units_on_order,
#             p.reorder_level,
#             COALESCE(SUM(od.quantity), 0) as total_sold,
#             CASE 
#             WHEN p.units_in_stock > 0 THEN ROUND((COALESCE(SUM(od.quantity), 0)::numeric / p.units_in_stock), 2)
#             ELSE 0 
#             END as turnover_ratio
#         FROM {schema}.products p
#         JOIN {schema}.categories c ON p.category_id = c.category_id
#         LEFT JOIN {schema}.order_details od ON p.product_id = od.product_id
#         WHERE p.discontinued = 0
#         GROUP BY p.product_id, p.product_name, c.category_name, p.units_in_stock, p.units_on_order, p.reorder_level
#         )
#         SELECT 
#         'processing' as analysis_type,
#         ROUND(AVG(processing_days)::numeric, 1) as avg_processing_days,
#         ROUND(AVG(promised_delivery_days)::numeric, 1) as avg_promised_days,
#         SUM(late_delivery) as total_late_deliveries,
#         COUNT(*) as total_orders,
#         ROUND(AVG(freight)::numeric, 2) as avg_freight_cost
#         FROM order_processing

#         UNION ALL

#         SELECT 
#         'inventory' as analysis_type,
#         ROUND(AVG(turnover_ratio)::numeric, 2) as avg_turnover,
#         COUNT(CASE WHEN units_in_stock <= reorder_level THEN 1 END)::numeric as low_stock_items,
#         COUNT(*)::numeric as total_active_products,
#         SUM(units_in_stock)::numeric as total_inventory_units,
#         ROUND(AVG(units_in_stock)::numeric, 1) as avg_stock_per_product
#         FROM inventory_turnover; """

#         df_operational = pd.read_sql_query(operational_query, conn_string)
        
        
#         # Detailed inventory analysis
#         inventory_detail_query = f"""
#         SELECT 
#             p.product_name,
#             c.category_name,
#             p.units_in_stock,
#             p.reorder_level,
#             p.units_on_order,
#             COALESCE(SUM(od.quantity), 0) as units_sold,
#             CASE 
#                 WHEN p.units_in_stock > 0 AND COALESCE(SUM(od.quantity), 0) > 0
#                 THEN ROUND((COALESCE(SUM(od.quantity), 0) / p.units_in_stock)::numeric, 2)
#                 ELSE 0 
#             END as turnover_ratio,
#             CASE 
#                 WHEN p.units_in_stock <= p.reorder_level THEN 'LOW STOCK'
#                 WHEN p.units_in_stock = 0 THEN 'OUT OF STOCK'
#                 WHEN p.units_in_stock > p.reorder_level * 3 THEN 'OVERSTOCK'
#                 ELSE 'NORMAL'
#             END as stock_status
#         FROM {schema}.products p
#         JOIN {schema}.categories c ON p.category_id = c.category_id
#         LEFT JOIN {schema}.order_details od ON p.product_id = od.product_id
#         WHERE p.discontinued = 0
#         GROUP BY p.product_id, p.product_name, c.category_name, 
#                  p.units_in_stock, p.reorder_level, p.units_on_order
#         ORDER BY turnover_ratio DESC
#         """
        
#         df_inventory_detail = pd.read_sql_query(inventory_detail_query, conn_string)
        
#         operational_doc = "NORTHWIND OPERATIONAL EFFICIENCY AND INVENTORY MANAGEMENT ANALYSIS:\n\n"
        
#         # Extract operational metrics
#         processing_metrics = df_operational[df_operational['analysis_type'] == 'processing'].iloc[0]
#         inventory_metrics = df_operational[df_operational['analysis_type'] == 'inventory'].iloc[0]

#         operational_doc += "ORDER PROCESSING EFFICIENCY:\n"
#         operational_doc += f"• Average Processing Time: {processing_metrics['avg_processing_days']:.1f} days\n"
#         operational_doc += f"• Average Promised Delivery: {processing_metrics['avg_promised_days']:.1f} days\n"
#         operational_doc += f"• Late Deliveries: {processing_metrics['total_late_deliveries']:.0f} out of {processing_metrics['total_orders']:.0f} orders\n"
#         operational_doc += f"• Average Freight Cost: ${processing_metrics['avg_freight_cost']:.2f} per shipment\n\n"

#         operational_doc += "INVENTORY MANAGEMENT PERFORMANCE:\n"
#         operational_doc += f"• Average Inventory Turnover Ratio: {inventory_metrics['avg_turnover']:.2f}\n"
#         operational_doc += f"• Products Below Reorder Level: {inventory_metrics['low_stock_items']:.0f} out of {inventory_metrics['total_active_products']:.0f}\n"
#         operational_doc += f"• Total Inventory Units: {inventory_metrics['total_inventory_units']:.0f}\n"
#         operational_doc += f"• Average Stock per Product: {inventory_metrics['avg_stock_per_product']:.0f} units\n\n"        
#         # Stock status analysis
#         stock_status_summary = df_inventory_detail['stock_status'].value_counts()
        
#         operational_doc += "INVENTORY STATUS BREAKDOWN:\n"
#         for status, count in stock_status_summary.items():
#             operational_doc += f"• {status}: {count} products ({(count/len(df_inventory_detail)*100):.1f}%)\n"
        
#         # Critical inventory alerts
#         critical_items = df_inventory_detail[df_inventory_detail['stock_status'].isin(['LOW STOCK', 'OUT OF STOCK'])]
#         high_turnover_critical = critical_items[critical_items['turnover_ratio'] > inventory_metrics['avg_turnover']]
        
#         operational_doc += f"\nCRITICAL INVENTORY ALERTS:\n"
#         operational_doc += f"• Items Needing Immediate Attention: {len(critical_items)}\n"
#         operational_doc += f"• High-Demand Items with Low Stock: {len(high_turnover_critical)}\n"
        
#         if len(high_turnover_critical) > 0:
#             operational_doc += "\nURGENT REORDER RECOMMENDATIONS:\n"
#             for _, item in high_turnover_critical.head(10).iterrows():
#                 operational_doc += f"• {item['product_name']} ({item['category_name']})\n"
#                 operational_doc += f"  - Current Stock: {item['units_in_stock']}, Reorder Level: {item['reorder_level']}\n"
#                 operational_doc += f"  - Turnover Ratio: {item['turnover_ratio']:.2f}, On Order: {item['units_on_order']}\n"
        
#         # Best performing inventory
#         operational_doc += "\nTOP PERFORMING PRODUCTS (by turnover):\n"
#         top_performers = df_inventory_detail[df_inventory_detail['turnover_ratio'] > 0].head(15)
#         for _, item in top_performers.iterrows():
#             operational_doc += f"• {item['product_name']}: {item['turnover_ratio']:.2f} turnover ratio\n"
#             operational_doc += f"  - Stock: {item['units_in_stock']}, Sold: {item['units_sold']}, Status: {item['stock_status']}\n"
        
#         documents.append(operational_doc)
        
        print(f"Successfully created {len(documents)} comprehensive business documents from Northwind PostgreSQL database")
        print(f"Total document length: {sum(len(doc) for doc in documents):,} characters")
        
        return documents
        
    except Exception as e:
        print(f"Error creating comprehensive documents: {e}")
        import traceback
        traceback.print_exc()
        return []

# # Usage example:
# if __name__ == "__main__":
#     # Neon PostgreSQL connection parameters
#     business_docs = create_comprehensive_northwind_business_documents(
#         host="ep-xxx-xxx.us-east-1.aws.neon.tech",  # Your Neon host
#         username="your_username",
#         password="your_password",
#         database="neondb",
#         port=5432,
#         schema="northwind"
#     )
    
#     # Preview the documents
#     for i, doc in enumerate(business_docs):
#         print(f"\n{'='*80}")
#         print(f"DOCUMENT {i+1}: {['CUSTOMER ANALYSIS', 'CUSTOMER BEHAVIOR', 'PRODUCT CATALOG', 'SUPPLIER ANALYSIS', 'EMPLOYEE PERFORMANCE', 'SHIPPING LOGISTICS', 'FINANCIAL PERFORMANCE', 'BUSINESS INTELLIGENCE', 'OPERATIONAL EFFICIENCY'][i]}")
#         print('='*80)
#         print(doc[:1000] + "..." if len(doc) > 1000 else doc)
    
    # Integration with vector store
    """
    from langchain.schema import Document
    
    # Convert to LangChain documents with detailed metadata
    langchain_docs = []
    doc_types = [
        "customer_analysis", "customer_behavior", "product_catalog", 
        "supplier_analysis", "employee_performance", "shipping_logistics",
        "financial_performance", "business_intelligence", "operational_efficiency"
    ]
    
    for i, doc in enumerate(business_docs):
        langchain_docs.append(Document(
            page_content=doc,
            metadata={
                "source": f"northwind_comprehensive_{doc_types[i]}",
                "type": "business_analysis",
                "document_id": i,
                "comprehensive": True,
                "data_source": "postgresql_neon"
            }
        ))
    
    # Enhanced text splitting for comprehensive documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Larger chunks for comprehensive content
        chunk_overlap=100,  # More overlap for context preservation
        length_function=tiktoken_len,
    )
    
    split_chunks = text_splitter.split_documents(langchain_docs)
    
    # Create vector store
    qdrant_vectorstore = Qdrant.from_documents(
        split_chunks,
        embedding_model,
        location=":memory:",
        collection_name="northwind_comprehensive_business_data",
    )
    """