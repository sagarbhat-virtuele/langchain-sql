from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model

# Human Intervention 
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 

# os.environ["GOOGLE_API_KEY"] = "..."

# model = init_chat_model("google_genai:gemini-2.5-flash")

# Replace this with your actual Gemini API key
api_key = "AIzaSyC4Y5A0263oYV8vmyqsisqBPN2oIGPUIhA"

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-pro"
    google_api_key=api_key,
    temperature=0.2
)

# Database connection
db_user = "root"
db_password = "password"
db_host = "localhost"
db_name = "atliq_tshirts"

db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
    sample_rows_in_table_info=3
)

# print(db.table_info)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

# for tool in tools:
#     print(f"{tool.name}: {tool.description}\n")

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

from langchain.agents import create_agent


agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
)

# question = "Which genre on average has the longest tracks?"
# question = "How many t-shirts do we have left for Nike in extra small size and white color?"
# question = "How many t-shirts do we have left in medium size and black color?"
# question = "How much is the price of the inventory for all small size t-shirts?"
# question = "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?"
# question = "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?"
# question = "If we have to sell all the T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?"
# question = "How many white color Levi's shirt I have?"
# question = "Which brand t-shirt will generate the highest revenue if we sell all size t shirts today after discounts?"
# question = "If we have to sell all the T-shirts today. How much revenue our store will generate with and without discount?"
question = "How much is the price of the inventory for all small size t-shirts ?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()