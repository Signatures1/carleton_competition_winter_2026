"""
Labeled training dataset for the SQL Query Writer Agent.

Contains (question, expected_sql) pairs covering the full range of query
types that may appear in the competition evaluation dataset:
  - Simple SELECT
  - COUNT / aggregation
  - ORDER BY + LIMIT  (top-N)
  - GROUP BY
  - JOIN (2-table and 3-table)
  - WHERE filters
  - DISTINCT
  - Complex / subquery

Database schema (bike_store DuckDB):
  brands(brand_id, brand_name)
  categories(category_id, category_name)
  customers(customer_id, first_name, last_name, phone, email, street, city, state, zip_code)
  order_items(order_id, item_id, product_id, quantity, list_price, discount)
  orders(order_id, customer_id, order_status, order_date, required_date, shipped_date, store_id, staff_id)
  products(product_id, product_name, brand_id, category_id, model_year, list_price)
  staffs(staff_id, first_name, last_name, email, phone, active, store_id, manager_id)
  stocks(store_id, product_id, quantity)
  stores(store_id, store_name, phone, email, street, city, state, zip_code)
"""

LABELED_EXAMPLES = [
    # ---------------------------------------------------------------
    # Simple COUNT queries
    # ---------------------------------------------------------------
    {
        "question": "How many customers are there?",
        "sql": "SELECT COUNT(*) AS total_customers FROM customers",
        "type": "count",
    },
    {
        "question": "What is the total number of orders?",
        "sql": "SELECT COUNT(*) AS total_orders FROM orders",
        "type": "count",
    },
    {
        "question": "How many products does the store sell?",
        "sql": "SELECT COUNT(*) AS total_products FROM products",
        "type": "count",
    },
    {
        "question": "How many staff members are there?",
        "sql": "SELECT COUNT(*) AS total_staff FROM staffs",
        "type": "count",
    },
    {
        "question": "How many stores are there?",
        "sql": "SELECT COUNT(*) AS total_stores FROM stores",
        "type": "count",
    },
    {
        "question": "How many brands are available?",
        "sql": "SELECT COUNT(*) AS total_brands FROM brands",
        "type": "count",
    },
    {
        "question": "How many product categories are there?",
        "sql": "SELECT COUNT(*) AS total_categories FROM categories",
        "type": "count",
    },
    {
        "question": "What is the total number of products?",
        "sql": "SELECT COUNT(*) AS total_products FROM products",
        "type": "count",
    },
    {
        "question": "How many orders have been placed in total?",
        "sql": "SELECT COUNT(*) AS total_orders FROM orders",
        "type": "count",
    },
    {
        "question": "How many items are in the order_items table?",
        "sql": "SELECT COUNT(*) AS total_items FROM order_items",
        "type": "count",
    },

    # ---------------------------------------------------------------
    # Simple SELECT queries
    # ---------------------------------------------------------------
    {
        "question": "List all brands",
        "sql": "SELECT brand_name FROM brands ORDER BY brand_name",
        "type": "select",
    },
    {
        "question": "Show all product categories",
        "sql": "SELECT category_name FROM categories ORDER BY category_name",
        "type": "select",
    },
    {
        "question": "List all store names and their cities",
        "sql": "SELECT store_name, city FROM stores ORDER BY store_name",
        "type": "select",
    },
    {
        "question": "Show all customer names",
        "sql": "SELECT first_name, last_name FROM customers ORDER BY last_name, first_name",
        "type": "select",
    },
    {
        "question": "List all staff member names",
        "sql": "SELECT first_name, last_name FROM staffs ORDER BY last_name, first_name",
        "type": "select",
    },
    {
        "question": "Show all product names and their prices",
        "sql": "SELECT product_name, list_price FROM products ORDER BY product_name",
        "type": "select",
    },
    {
        "question": "List all stores with their contact information",
        "sql": "SELECT store_name, phone, email, street, city, state FROM stores ORDER BY store_name",
        "type": "select",
    },

    # ---------------------------------------------------------------
    # ORDER BY + LIMIT  (top-N)
    # ---------------------------------------------------------------
    {
        "question": "What are the top 5 most expensive products?",
        "sql": "SELECT product_name, list_price FROM products ORDER BY list_price DESC LIMIT 5",
        "type": "top_n",
    },
    {
        "question": "What are the 10 cheapest products?",
        "sql": "SELECT product_name, list_price FROM products ORDER BY list_price ASC LIMIT 10",
        "type": "top_n",
    },
    {
        "question": "Show the 3 most recent orders",
        "sql": "SELECT order_id, order_date, customer_id FROM orders ORDER BY order_date DESC LIMIT 3",
        "type": "top_n",
    },
    {
        "question": "What are the top 5 most expensive products with their brand names?",
        "sql": (
            "SELECT p.product_name, b.brand_name, p.list_price "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "ORDER BY p.list_price DESC LIMIT 5"
        ),
        "type": "top_n",
    },
    {
        "question": "Show the 5 most recently placed orders with customer names",
        "sql": (
            "SELECT o.order_id, c.first_name, c.last_name, o.order_date "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "ORDER BY o.order_date DESC LIMIT 5"
        ),
        "type": "top_n",
    },
    {
        "question": "What are the 5 least expensive products?",
        "sql": "SELECT product_name, list_price FROM products ORDER BY list_price ASC LIMIT 5",
        "type": "top_n",
    },
    {
        "question": "Show the 3 oldest orders",
        "sql": "SELECT order_id, order_date, customer_id FROM orders ORDER BY order_date ASC LIMIT 3",
        "type": "top_n",
    },
    {
        "question": "What are the top 3 brands by number of products?",
        "sql": (
            "SELECT b.brand_name, COUNT(p.product_id) AS product_count "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "GROUP BY b.brand_name ORDER BY product_count DESC LIMIT 3"
        ),
        "type": "top_n",
    },
    {
        "question": "Which 5 customers made the most orders?",
        "sql": (
            "SELECT c.first_name, c.last_name, COUNT(o.order_id) AS order_count "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.customer_id, c.first_name, c.last_name "
            "ORDER BY order_count DESC LIMIT 5"
        ),
        "type": "top_n",
    },

    # ---------------------------------------------------------------
    # Aggregation (no GROUP BY)
    # ---------------------------------------------------------------
    {
        "question": "What is the average product price?",
        "sql": "SELECT AVG(list_price) AS avg_price FROM products",
        "type": "aggregation",
    },
    {
        "question": "What is the total revenue from all order items?",
        "sql": "SELECT SUM(quantity * list_price * (1 - discount)) AS total_revenue FROM order_items",
        "type": "aggregation",
    },
    {
        "question": "What is the most expensive product?",
        "sql": "SELECT product_name, list_price FROM products ORDER BY list_price DESC LIMIT 1",
        "type": "aggregation",
    },
    {
        "question": "What is the cheapest product?",
        "sql": "SELECT product_name, list_price FROM products ORDER BY list_price ASC LIMIT 1",
        "type": "aggregation",
    },
    {
        "question": "What is the maximum product price?",
        "sql": "SELECT MAX(list_price) AS max_price FROM products",
        "type": "aggregation",
    },
    {
        "question": "What is the minimum product price?",
        "sql": "SELECT MIN(list_price) AS min_price FROM products",
        "type": "aggregation",
    },
    {
        "question": "What store has the most inventory?",
        "sql": (
            "SELECT s.store_name, SUM(sk.quantity) AS total_inventory "
            "FROM stocks sk JOIN stores s ON sk.store_id = s.store_id "
            "GROUP BY s.store_name ORDER BY total_inventory DESC LIMIT 1"
        ),
        "type": "aggregation",
    },
    {
        "question": "Which product has the highest total stock across all stores?",
        "sql": (
            "SELECT p.product_name, SUM(sk.quantity) AS total_stock "
            "FROM stocks sk JOIN products p ON sk.product_id = p.product_id "
            "GROUP BY p.product_name ORDER BY total_stock DESC LIMIT 1"
        ),
        "type": "aggregation",
    },

    # ---------------------------------------------------------------
    # GROUP BY
    # ---------------------------------------------------------------
    {
        "question": "How many products are in each category?",
        "sql": (
            "SELECT c.category_name, COUNT(p.product_id) AS product_count "
            "FROM products p JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.category_name ORDER BY product_count DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "How many products does each brand have?",
        "sql": (
            "SELECT b.brand_name, COUNT(p.product_id) AS product_count "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "GROUP BY b.brand_name ORDER BY product_count DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "What is the total number of orders per store?",
        "sql": (
            "SELECT s.store_name, COUNT(o.order_id) AS order_count "
            "FROM orders o JOIN stores s ON o.store_id = s.store_id "
            "GROUP BY s.store_name ORDER BY order_count DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "What is the average price of products by brand?",
        "sql": (
            "SELECT b.brand_name, AVG(p.list_price) AS avg_price "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "GROUP BY b.brand_name ORDER BY avg_price DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "What is the total revenue by store?",
        "sql": (
            "SELECT s.store_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_revenue "
            "FROM orders o "
            "JOIN order_items oi ON o.order_id = oi.order_id "
            "JOIN stores s ON o.store_id = s.store_id "
            "GROUP BY s.store_name ORDER BY total_revenue DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "How many orders are there per year?",
        "sql": (
            "SELECT EXTRACT(YEAR FROM order_date) AS year, COUNT(*) AS order_count "
            "FROM orders GROUP BY year ORDER BY year"
        ),
        "type": "group_by",
    },
    {
        "question": "What is the total revenue by brand?",
        "sql": (
            "SELECT b.brand_name, SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_revenue "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN brands b ON p.brand_id = b.brand_id "
            "GROUP BY b.brand_name ORDER BY total_revenue DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "How many orders were placed each month in 2018?",
        "sql": (
            "SELECT EXTRACT(MONTH FROM order_date) AS month, COUNT(*) AS order_count "
            "FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2018 "
            "GROUP BY month ORDER BY month"
        ),
        "type": "group_by",
    },
    {
        "question": "What is the average product price by category?",
        "sql": (
            "SELECT c.category_name, AVG(p.list_price) AS avg_price "
            "FROM products p JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.category_name ORDER BY avg_price DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "How many orders are there per store?",
        "sql": (
            "SELECT s.store_name, COUNT(o.order_id) AS order_count "
            "FROM orders o JOIN stores s ON o.store_id = s.store_id "
            "GROUP BY s.store_name ORDER BY order_count DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "What is the total revenue per category?",
        "sql": (
            "SELECT c.category_name, "
            "SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_revenue "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.category_name ORDER BY total_revenue DESC"
        ),
        "type": "group_by",
    },
    {
        "question": "How many products are available per brand?",
        "sql": (
            "SELECT b.brand_name, COUNT(p.product_id) AS product_count "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "GROUP BY b.brand_name ORDER BY product_count DESC"
        ),
        "type": "group_by",
    },

    # ---------------------------------------------------------------
    # JOIN queries (2-table)
    # ---------------------------------------------------------------
    {
        "question": "Show all products with their brand names",
        "sql": (
            "SELECT p.product_name, b.brand_name, p.list_price "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "ORDER BY p.product_name"
        ),
        "type": "join",
    },
    {
        "question": "List products with brand name and price",
        "sql": (
            "SELECT p.product_name, b.brand_name, p.list_price "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "ORDER BY p.product_name"
        ),
        "type": "join",
    },
    {
        "question": "Show all products with their category names",
        "sql": (
            "SELECT p.product_name, c.category_name, p.list_price "
            "FROM products p JOIN categories c ON p.category_id = c.category_id "
            "ORDER BY p.product_name"
        ),
        "type": "join",
    },
    {
        "question": "Show products with their brand and category and price",
        "sql": (
            "SELECT p.product_name, b.brand_name, c.category_name, p.list_price "
            "FROM products p "
            "JOIN brands b ON p.brand_id = b.brand_id "
            "JOIN categories c ON p.category_id = c.category_id "
            "ORDER BY p.product_name"
        ),
        "type": "join",
    },
    {
        "question": "Show all products with their brand and price",
        "sql": (
            "SELECT p.product_name, b.brand_name, p.list_price "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "ORDER BY p.product_name"
        ),
        "type": "join",
    },
    {
        "question": "List orders with customer names",
        "sql": (
            "SELECT o.order_id, c.first_name, c.last_name, o.order_date "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "ORDER BY o.order_date DESC"
        ),
        "type": "join",
    },
    {
        "question": "List all staff members and their stores",
        "sql": (
            "SELECT s.first_name, s.last_name, st.store_name "
            "FROM staffs s JOIN stores st ON s.store_id = st.store_id "
            "ORDER BY st.store_name, s.last_name"
        ),
        "type": "join",
    },
    {
        "question": "Show inventory levels for each product at each store",
        "sql": (
            "SELECT st.store_name, p.product_name, sk.quantity "
            "FROM stocks sk "
            "JOIN stores st ON sk.store_id = st.store_id "
            "JOIN products p ON sk.product_id = p.product_id "
            "ORDER BY st.store_name, p.product_name"
        ),
        "type": "join",
    },

    # ---------------------------------------------------------------
    # JOIN queries (3-table)
    # ---------------------------------------------------------------
    {
        "question": "Show products with their brand and category",
        "sql": (
            "SELECT p.product_name, b.brand_name, c.category_name, p.list_price "
            "FROM products p "
            "JOIN brands b ON p.brand_id = b.brand_id "
            "JOIN categories c ON p.category_id = c.category_id "
            "ORDER BY p.product_name"
        ),
        "type": "join",
    },
    {
        "question": "Show order items with product names and order dates",
        "sql": (
            "SELECT o.order_id, o.order_date, p.product_name, oi.quantity, oi.list_price "
            "FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.order_id "
            "JOIN products p ON oi.product_id = p.product_id "
            "ORDER BY o.order_date DESC"
        ),
        "type": "join",
    },
    {
        "question": "List order details with product names and quantities",
        "sql": (
            "SELECT o.order_id, o.order_date, p.product_name, oi.quantity, oi.list_price "
            "FROM order_items oi "
            "JOIN orders o ON oi.order_id = o.order_id "
            "JOIN products p ON oi.product_id = p.product_id "
            "ORDER BY o.order_id"
        ),
        "type": "join",
    },

    # ---------------------------------------------------------------
    # WHERE filter queries
    # ---------------------------------------------------------------
    {
        "question": "Show all orders from 2018",
        "sql": (
            "SELECT order_id, customer_id, order_date, order_status "
            "FROM orders WHERE EXTRACT(YEAR FROM order_date) = 2018"
        ),
        "type": "filter",
    },
    {
        "question": "List products under $500",
        "sql": (
            "SELECT product_name, list_price FROM products "
            "WHERE list_price < 500 ORDER BY list_price"
        ),
        "type": "filter",
    },
    {
        "question": "Show products priced over $1000",
        "sql": (
            "SELECT product_name, list_price FROM products "
            "WHERE list_price > 1000 ORDER BY list_price DESC"
        ),
        "type": "filter",
    },
    {
        "question": "Find customers in New York",
        "sql": (
            "SELECT first_name, last_name, city FROM customers "
            "WHERE state = 'NY' ORDER BY last_name"
        ),
        "type": "filter",
    },
    {
        "question": "Show active staff members",
        "sql": (
            "SELECT first_name, last_name, email FROM staffs "
            "WHERE active = 1 ORDER BY last_name"
        ),
        "type": "filter",
    },
    {
        "question": "List all inactive staff members",
        "sql": (
            "SELECT first_name, last_name, email FROM staffs "
            "WHERE active = 0 ORDER BY last_name"
        ),
        "type": "filter",
    },
    {
        "question": "List products from model year 2018",
        "sql": (
            "SELECT product_name, list_price FROM products "
            "WHERE model_year = 2018 ORDER BY product_name"
        ),
        "type": "filter",
    },
    {
        "question": "Show completed orders",
        "sql": (
            "SELECT order_id, customer_id, order_date FROM orders "
            "WHERE order_status = 4 ORDER BY order_date DESC"
        ),
        "type": "filter",
    },
    {
        "question": "Find all Trek products",
        "sql": (
            "SELECT p.product_name, p.list_price, p.model_year "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "WHERE b.brand_name = 'Trek' ORDER BY p.product_name"
        ),
        "type": "filter",
    },
    {
        "question": "Show orders placed in 2019",
        "sql": (
            "SELECT order_id, customer_id, order_date FROM orders "
            "WHERE EXTRACT(YEAR FROM order_date) = 2019 ORDER BY order_date"
        ),
        "type": "filter",
    },
    {
        "question": "Find all orders placed in 2016",
        "sql": (
            "SELECT order_id, customer_id, order_date, order_status FROM orders "
            "WHERE EXTRACT(YEAR FROM order_date) = 2016 ORDER BY order_date"
        ),
        "type": "filter",
    },
    {
        "question": "Show orders shipped in January 2018",
        "sql": (
            "SELECT order_id, customer_id, shipped_date FROM orders "
            "WHERE EXTRACT(YEAR FROM shipped_date) = 2018 "
            "AND EXTRACT(MONTH FROM shipped_date) = 1 ORDER BY shipped_date"
        ),
        "type": "filter",
    },
    {
        "question": "List customers from California",
        "sql": (
            "SELECT first_name, last_name, city FROM customers "
            "WHERE state = 'CA' ORDER BY last_name"
        ),
        "type": "filter",
    },
    {
        "question": "Show customers who live in Texas",
        "sql": (
            "SELECT first_name, last_name, city FROM customers "
            "WHERE state = 'TX' ORDER BY last_name"
        ),
        "type": "filter",
    },

    # ---------------------------------------------------------------
    # DISTINCT
    # ---------------------------------------------------------------
    {
        "question": "What states do customers come from?",
        "sql": "SELECT DISTINCT state FROM customers ORDER BY state",
        "type": "distinct",
    },
    {
        "question": "What model years are available?",
        "sql": "SELECT DISTINCT model_year FROM products ORDER BY model_year DESC",
        "type": "distinct",
    },
    {
        "question": "What cities are stores located in?",
        "sql": "SELECT DISTINCT city, state FROM stores ORDER BY state, city",
        "type": "distinct",
    },
    {
        "question": "What city and state is each store located in?",
        "sql": "SELECT DISTINCT city, state FROM stores ORDER BY state, city",
        "type": "distinct",
    },
    {
        "question": "What unique cities do customers live in?",
        "sql": "SELECT DISTINCT city FROM customers ORDER BY city",
        "type": "distinct",
    },
    {
        "question": "List all distinct order statuses",
        "sql": "SELECT DISTINCT order_status FROM orders ORDER BY order_status",
        "type": "distinct",
    },
    {
        "question": "What brands are available in the store?",
        "sql": "SELECT DISTINCT brand_name FROM brands ORDER BY brand_name",
        "type": "distinct",
    },
    {
        "question": "What zip codes do customers come from?",
        "sql": "SELECT DISTINCT zip_code FROM customers ORDER BY zip_code",
        "type": "distinct",
    },

    # ---------------------------------------------------------------
    # Complex / subquery
    # ---------------------------------------------------------------
    {
        "question": "Which customers have placed more than 2 orders?",
        "sql": (
            "SELECT c.first_name, c.last_name, COUNT(o.order_id) AS order_count "
            "FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.customer_id, c.first_name, c.last_name "
            "HAVING COUNT(o.order_id) > 2 ORDER BY order_count DESC"
        ),
        "type": "complex",
    },
    {
        "question": "What products have never been ordered?",
        "sql": (
            "SELECT p.product_name FROM products p "
            "WHERE p.product_id NOT IN (SELECT DISTINCT product_id FROM order_items)"
        ),
        "type": "complex",
    },
    {
        "question": "Which customer has spent the most money?",
        "sql": (
            "SELECT c.first_name, c.last_name, "
            "SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_spent "
            "FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "JOIN order_items oi ON o.order_id = oi.order_id "
            "GROUP BY c.customer_id, c.first_name, c.last_name "
            "ORDER BY total_spent DESC LIMIT 1"
        ),
        "type": "complex",
    },
    {
        "question": "Which products are out of stock?",
        "sql": (
            "SELECT p.product_name "
            "FROM products p JOIN stocks sk ON p.product_id = sk.product_id "
            "GROUP BY p.product_id, p.product_name HAVING SUM(sk.quantity) = 0"
        ),
        "type": "complex",
    },
    {
        "question": "Show the total sales amount by category",
        "sql": (
            "SELECT c.category_name, "
            "SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_sales "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.category_name ORDER BY total_sales DESC"
        ),
        "type": "complex",
    },
    {
        "question": "What is the average discount per category?",
        "sql": (
            "SELECT c.category_name, AVG(oi.discount) AS avg_discount "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.product_id "
            "JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.category_name ORDER BY avg_discount DESC"
        ),
        "type": "complex",
    },
    {
        "question": "Which staff member has handled the most orders?",
        "sql": (
            "SELECT s.first_name, s.last_name, COUNT(o.order_id) AS order_count "
            "FROM staffs s JOIN orders o ON s.staff_id = o.staff_id "
            "GROUP BY s.staff_id, s.first_name, s.last_name "
            "ORDER BY order_count DESC LIMIT 1"
        ),
        "type": "complex",
    },
    {
        "question": "Show customer orders with order totals",
        "sql": (
            "SELECT c.first_name, c.last_name, o.order_id, "
            "SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS order_total "
            "FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "JOIN order_items oi ON o.order_id = oi.order_id "
            "GROUP BY c.customer_id, c.first_name, c.last_name, o.order_id "
            "ORDER BY order_total DESC"
        ),
        "type": "complex",
    },
    {
        "question": "How many orders has each customer placed?",
        "sql": (
            "SELECT c.first_name, c.last_name, COUNT(o.order_id) AS order_count "
            "FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.customer_id, c.first_name, c.last_name "
            "ORDER BY order_count DESC"
        ),
        "type": "complex",
    },
    {
        "question": "What is the best-selling product by quantity?",
        "sql": (
            "SELECT p.product_name, SUM(oi.quantity) AS total_quantity "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_id, p.product_name "
            "ORDER BY total_quantity DESC LIMIT 1"
        ),
        "type": "complex",
    },
    {
        "question": "Show the top 5 best-selling products by revenue",
        "sql": (
            "SELECT p.product_name, "
            "SUM(oi.quantity * oi.list_price * (1 - oi.discount)) AS total_revenue "
            "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
            "GROUP BY p.product_id, p.product_name "
            "ORDER BY total_revenue DESC LIMIT 5"
        ),
        "type": "complex",
    },
    {
        "question": "Show products and their in-stock quantities across all stores",
        "sql": (
            "SELECT p.product_name, SUM(sk.quantity) AS total_stock "
            "FROM products p JOIN stocks sk ON p.product_id = sk.product_id "
            "GROUP BY p.product_id, p.product_name "
            "ORDER BY total_stock DESC"
        ),
        "type": "complex",
    },
    {
        "question": "Which brands have the highest average product price?",
        "sql": (
            "SELECT b.brand_name, AVG(p.list_price) AS avg_price "
            "FROM products p JOIN brands b ON p.brand_id = b.brand_id "
            "GROUP BY b.brand_name ORDER BY avg_price DESC LIMIT 5"
        ),
        "type": "complex",
    },
    {
        "question": "Show stores and the number of staff members at each",
        "sql": (
            "SELECT st.store_name, COUNT(s.staff_id) AS staff_count "
            "FROM stores st LEFT JOIN staffs s ON st.store_id = s.store_id "
            "GROUP BY st.store_name ORDER BY staff_count DESC"
        ),
        "type": "complex",
    },
]


def get_train_val_test_split(seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Split LABELED_EXAMPLES into stratified train / validation / test sets.

    Args:
        seed:        Random seed for reproducibility.
        train_ratio: Fraction of examples to use for training (default 0.70).
        val_ratio:   Fraction for validation (default 0.15); remainder goes to test.

    Returns:
        tuple[list, list, list]: (train_examples, val_examples, test_examples)
    """
    import random
    from collections import defaultdict

    # Group by query type for stratified split
    by_type = defaultdict(list)
    for ex in LABELED_EXAMPLES:
        by_type[ex["type"]].append(ex)

    train, val, test = [], [], []
    rng = random.Random(seed)

    for examples in by_type.values():
        shuffled = examples.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        # Make sure val and test both get at least 1 example when n >= 3
        if n >= 3 and n_val == 0:
            n_val = 1
        n_train = max(1, n - n_val - max(1 if n >= 3 else 0, 0))
        n_test = n - n_train - n_val

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return train, val, test
