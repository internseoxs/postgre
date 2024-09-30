from flask import Flask, request, jsonify, render_template
import openai
import psycopg2
import json
import sqlparse
import re
from decimal import Decimal
from datetime import datetime, date
import logging

app = Flask(__name__)
response_history = {}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from config.json file
try:
    with open('config.json') as config_file:
        config = json.load(config_file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.error("Error loading config file: %s", e)
    raise

openai.api_key = config.get('openai_api_key')
if not openai.api_key:
    logging.error('OpenAI API key is missing in the config file')
    raise ValueError('OpenAI API key is missing in the config file')

# Database connection parameters from config file
DB_HOST = config['database']['host']
DB_PORT = config['database']['port']
DB_NAME = config['database']['dbname']
DB_USER = config['database']['user']
DB_PASSWORD = config['database']['password']


def get_database_schema():
    """Retrieve table and column names from the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
        """)
        schema = cursor.fetchall()
        cursor.close()
        conn.close()

        schema_dict = {}
        for table, column in schema:
            if table not in schema_dict:
                schema_dict[table] = []
            schema_dict[table].append(column)

        return schema_dict

    except Exception as e:
        logging.error("Error fetching database schema: %s", e)
        return {}


def format_schema_for_gpt(schema):
    """Format the schema in a way that can be included in the prompt to GPT."""
    formatted_schema = ""
    for table, columns in schema.items():
        formatted_schema += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
    return formatted_schema


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/submit_query", methods=["POST"])
def submit_query():
    prompt = request.form.get('prompt')
    if not prompt:
        return render_template("contact.html", query=None, results=None, error="No prompt provided.")
    return render_template("about.html", prompt=prompt)


@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']
    schema = get_database_schema()
    if not schema:
        return jsonify({'error': 'Failed to retrieve database schema'}), 500

    formatted_schema = format_schema_for_gpt(schema)

    # Determine if new data is needed based on the current prompt
    if requires_more_data(prompt):
        sql_query = generate_sql_query(prompt, formatted_schema)
        if not sql_query:
            return jsonify({'error': 'Failed to generate SQL query'}), 500

        # Log the generated SQL query
        logging.info("Generated SQL query: %s", sql_query)

        new_data = execute_sql_query(sql_query)
        if new_data is None:
            new_data = []  # Proceed with empty data if SQL execution failed

        # Store the new data in response_history
        response_history['previous_response'] = new_data
    else:
        # Use previously retrieved data
        new_data = response_history.get('previous_response', [])

    final_response = generate_final_response(prompt, new_data)
    if not final_response:
        return jsonify({'error': 'Failed to generate final response'}), 500

    return jsonify({'response': final_response})


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the response history and redirect to the home page."""
    global response_history
    response_history.clear()  # Clear the response history
    return jsonify({"message": "Response history cleared."}), 200


def requires_more_data(prompt):
    """Determine if the prompt requires more data from the database."""
    if not response_history.get('previous_response'):
        return True  # If no previous data, we need new data

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that determines if the user's query can be answered "
                "with the provided data. Respond 'yes' if more data is needed, otherwise respond 'no'."
            )
        },
        {
            "role": "user",
            "content": f"Prompt: {prompt}\nExisting Data: {json.dumps(response_history.get('previous_response', []), default=convert_to_serializable)}"
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50,
            temperature=0.0,
            n=1,
        )

        return response.choices[0].message['content'].strip().lower() == 'yes'
    except Exception as e:
        logging.error("Error determining if more data is needed: %s", e)
        return True  # Default to needing more data if GPT fails


def ensure_semicolon(sql_query):
    """Ensure the SQL query starts with SELECT and ends with a semicolon."""
    pattern = r"\bSELECT\b.*?(?=;|$)"
    match = re.search(pattern, sql_query, re.IGNORECASE | re.DOTALL)
    if match:
        sql_query = match.group(0).strip()
        return sql_query + ';' if not sql_query.endswith(';') else sql_query
    return sql_query


def generate_sql_query(prompt, schema):
    """Generate SQL query based on user prompt and schema."""
    messages = [
        {
            "role": "system",
            "content": (
                    "You are an assistant that converts user prompts into safe, "
                    "read-only SQL queries. Here is the database schema:\n\n" + schema
            )
        },
        {"role": "user", "content": f"Generate an SQL query for the following prompt: {prompt}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=5000,
            temperature=0,
            n=1,
        )

        sql_query = response.choices[0].message['content'].strip()
        return ensure_semicolon(sql_query)

    except Exception as e:
        logging.error("Error generating SQL query: %s", e)
        return None


def is_safe_sql(sql_query):
    """Check if the SQL query is safe and read-only."""
    try:
        parsed = sqlparse.parse(sql_query)
        return all(statement.get_type() == 'SELECT' for statement in parsed)
    except Exception as e:
        logging.error("Error parsing SQL query: %s", e)
        return False


def execute_sql_query(sql_query):
    """Execute the SQL query and return the result."""
    if not is_safe_sql(sql_query):
        logging.warning("Unsafe SQL query detected, not executing: %s", sql_query)
        return []  # Return an empty list to proceed with GPT response

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)

        # Fetch limited results to prevent sending too much data
        results = cursor.fetchmany(500)  # Fetch up to 500 rows
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()

        return [dict(zip(colnames, row)) for row in results] if results else []
    except Exception as e:
        logging.error("Error executing SQL query: %s", e)
        return []  # Return an empty list to allow GPT to handle the response


def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable ones."""
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)  # Convert Decimal to float
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()  # Convert datetime/date to ISO 8601 string
    else:
        return obj


def chunk_data(data, chunk_size=100):
    """Split data into chunks of specified size."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def generate_final_response(prompt, db_data):
    """
    Generate the final response by passing the prompt and data to GPT.
    If no data is available from the database or the query is incorrect, GPT will answer directly.
    """
    final_response = []

    if db_data:  # If database returned data, use it to generate the response
        db_data_serializable = convert_to_serializable(db_data)

        for chunk in chunk_data(db_data_serializable):
            messages = [
                {"role": "system", "content": "You are an assistant that helps users analyze data."},
                {"role": "user", "content": f"Prompt: {prompt}\nData: {json.dumps(chunk)}"}
            ]

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.5,
                    n=1,
                )
                final_response.append(response.choices[0].message['content'].strip())

            except Exception as e:
                logging.error("Error generating GPT response: %s", e)
                return None

    else:  # If no data is available or query failed, use GPT to generate the response
        messages = [
            {"role": "system", "content": "You are an assistant that helps users answer queries."},
            {"role": "user", "content": f"Prompt: {prompt}\nData: No data was available from the database."}
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=5000,
                temperature=0.5,
                n=1,
            )
            final_response.append(response.choices[0].message['content'].strip())

        except Exception as e:
            logging.error("Error generating GPT fallback response: %s", e)
            return None

    return " ".join(final_response)


if __name__ == "__main__":
    app.run(debug=True)
