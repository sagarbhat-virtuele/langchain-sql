import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

def get_gemini_sql_agent():
    # --- Database configuration ---
    db_user = "root"
    db_password = "password"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    # --- Connect to MySQL database ---
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )

    # --- Initialize Gemini model ---
    api_key = os.getenv("GOOGLE_API_KEY") or "AIzaSyC4Y5A0263oYV8vmyqsisqBPN2oIGPUIhA"
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # or "gemini-1.5-pro"
        google_api_key=api_key,
        temperature=0.2
    )

    # --- Initialize Toolkit and Tools ---
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # --- Define System Prompt ---
    system_prompt = f"""
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {db.dialect} query to run,
    then look at the results of the query and return the answer.

    Unless the user specifies a specific number of examples they wish to obtain, 
    always limit your query to at most 5 results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table —
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start, you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.
    """

    # --- Create the Agent ---
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )

    return agent
