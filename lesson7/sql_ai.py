from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

# Create in-memory SQLite database
conn = sqlite3.connect(':memory:', check_same_thread=False)
cursor = conn.cursor()

# Read and execute the SQL file
def initialize_database():
    sql_file_path = 'financials.sql'
    
    if os.path.exists(sql_file_path):
        with open(sql_file_path, 'r') as f:
            sql_script = f.read()
            cursor.executescript(sql_script)
            conn.commit()
        print("✓ Database initialized with financials data")
    else:
        print("❌ financials.sql file not found!")

@tool
def get_table_schema() -> str:
    """Get the schema of the financials table to understand its structure."""
    cursor.execute("PRAGMA table_info(financials)")
    columns = cursor.fetchall()
    
    schema = "Table: financials\nColumns:\n"
    for col in columns:
        schema += f"  - {col[1]} ({col[2]})\n"
    
    return schema

@tool
def execute_sql_query(query: str) -> str:
    """
    Execute a SQL query on the financials database and return the results.
    
    Args:
        query: A valid SQL SELECT query
    
    Returns:
        Query results as formatted text
    """
    try:
        # Security: Only allow SELECT queries
        if not query.strip().upper().startswith('SELECT'):
            return "ERROR: Only SELECT queries are allowed for security reasons."
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        if not results:
            return "No results found."
        
        # Format results as a table
        output = "Query Results:\n"
        output += "-" * 80 + "\n"
        
        # Header
        output += " | ".join(f"{col:15}" for col in column_names) + "\n"
        output += "-" * 80 + "\n"
        
        # Rows
        for row in results:
            output += " | ".join(f"{str(val):15}" for val in row) + "\n"
        
        output += "-" * 80 + "\n"
        output += f"Total rows: {len(results)}\n"
        
        return output
    
    except Exception as e:
        return f"ERROR executing query: {str(e)}"

@tool
def get_sample_data() -> str:
    """Get a few sample rows from the financials table."""
    cursor.execute("SELECT * FROM financials LIMIT 5")
    results = cursor.fetchall()
    
    column_names = [description[0] for description in cursor.description]
    
    output = "Sample Data (First 5 rows):\n"
    output += "-" * 80 + "\n"
    output += " | ".join(f"{col:12}" for col in column_names) + "\n"
    output += "-" * 80 + "\n"
    
    for row in results:
        output += " | ".join(f"{str(val):12}" for val in row) + "\n"
    
    return output

def create_my_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [
        get_table_schema,
        execute_sql_query,
        get_sample_data
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

def main():
    # Initialize database
    initialize_database()
    
    agent = create_my_agent()
    chat_history = InMemoryChatMessageHistory()
    
    print("=" * 60)
    print("SQL QUERY ASSISTANT")
    print("=" * 60)
    print("Ask me questions about the financial data!")
    print("\nExamples:")
    print("  - List me top 3 users based on income")
    print("  - Show clients with credit score above 800")
    print("  - What's the average loan amount?")
    print("  - Find clients with past due payments")
    print("\nType 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            conn.close()
            break

        messages = [
            ("system", """You are a SQL query assistant that helps users query a financial database.

DATABASE SCHEMA:
Table: financials
Columns:
  - client_id (INTEGER PRIMARY KEY)
  - income (FLOAT)
  - credit_score (INTEGER)
  - loan_amount (FLOAT)
  - past_due (INTEGER) - number of past due payments
  - monthly_expenses (FLOAT)
  - savings_balance (FLOAT)

YOUR WORKFLOW:
1. When user asks a question about the data, analyze what they want
2. Write a precise SQL SELECT query to get that data
3. Call execute_sql_query with your SQL query
4. When you get the results, present them in a clear, readable format
5. Add helpful context or insights about the data

QUERY WRITING RULES:
- Always use proper SQL syntax
- Use ORDER BY for "top" or "highest/lowest" queries
- Use LIMIT to restrict number of results
- Use WHERE clauses for filtering
- Use aggregate functions (AVG, SUM, COUNT, MAX, MIN) when appropriate
- Use GROUP BY when aggregating by categories

EXAMPLES:
User: "List me top 3 users based on income"
SQL: SELECT client_id, income FROM financials ORDER BY income DESC LIMIT 3

User: "Show clients with credit score above 800"
SQL: SELECT client_id, credit_score, income FROM financials WHERE credit_score > 800

User: "What's the average loan amount?"
SQL: SELECT AVG(loan_amount) as average_loan FROM financials

User: "Find clients with past due payments"
SQL: SELECT client_id, past_due, loan_amount FROM financials WHERE past_due > 0

RESPONSE FORMAT:
1. Briefly acknowledge the question
2. Execute the query using the tool
3. Present results in a friendly way
4. Add insights or observations if relevant

Never show the SQL query to the user unless they specifically ask for it. Just show the results.""")
        ]
        
        # Add chat history
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(("assistant", msg.content))
        
        messages.append(("user", user_input))
        
        try:
            # Track iterations
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                response = agent.invoke(messages)
                
                # If no tool calls, we're done
                if not response.tool_calls:
                    reply = response.content
                    if reply and reply.strip():
                        print("\nAssistant:", reply)
                    break
                
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    try:
                        if tool_name == "get_table_schema":
                            result = get_table_schema.invoke({})
                        elif tool_name == "execute_sql_query":
                            result = execute_sql_query.invoke(tool_args)
                        elif tool_name == "get_sample_data":
                            result = get_sample_data.invoke({})
                        else:
                            result = f"Unknown tool: {tool_name}"
                        
                        tool_results.append(result)
                    except Exception as e:
                        error_msg = f"Error with {tool_name}: {str(e)}"
                        tool_results.append(error_msg)
                
                # Add tool results and continue
                messages.append(("assistant", response.content or ""))
                messages.append(("user", f"[Tool results: {' | '.join(tool_results)}]"))
            
            # Update chat history
            chat_history.add_user_message(user_input)
            if 'reply' in locals() and reply:
                chat_history.add_ai_message(reply)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
