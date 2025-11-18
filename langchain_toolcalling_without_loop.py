import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load .env
load_dotenv(dotenv_path="./.env")

# Debug print
print("Loaded KEY =", os.getenv("GROQ_API_KEY"))

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

@tool
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello {name}, welcome to the LangChain Groq agent!"


tools = [multiply, greet]

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
).bind_tools(tools)

system_prompt = SystemMessage(
    content="""
You are a tool-calling AI agent.
Use tools when needed.
If no tool is needed, answer normally.
Do NOT output tool JSON.
Return only final answer.
"""
)

agent = ChatPromptTemplate.from_messages([
    system_prompt,
    ("human", "{input}")
]) | llm

def ask_agent(query: str):
    response = agent.invoke({"input": query})

    if response.tool_calls:
        for tc in response.tool_calls:
            tool_name = tc["name"]
            args = tc["args"]
            tool_obj = next(t for t in tools if t.name == tool_name)
            result = tool_obj.func(**args)

        final = llm.invoke([
            system_prompt,
            HumanMessage(content=query),
            response,
            HumanMessage(content=str(result))
        ])
        return final.content

    return response.content

if __name__ == "__main__":
    print("\n=== Example 1: Tool Calling ===")
    print(ask_agent("Multiply 12 and 9"))

    print("\n=== Example 2: Tool Calling ===")
    print(ask_agent("Greet Anant"))

    print("\n=== Example 3: Normal Answer ===")
    print(ask_agent("What is the capital of India?"))