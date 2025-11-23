from langchain_openai import ChatOpenAI
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

serper = GoogleSerperAPIWrapper()

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return serper.run(query)

def create_my_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [search]
    
    # Bind tools to the model
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools

def main():
    agent = create_my_agent()
    chat_history = InMemoryChatMessageHistory()
    
    print("Agent ready. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        # Create messages with history
        messages = [
            ("system", "You are a helpful AI assistant that can search the web when needed.")
        ]
        
        # Add chat history
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(("assistant", msg.content))
        
        # Add current user message
        messages.append(("user", user_input))
        
        # Invoke the agent
        response = agent.invoke(messages)
        
        # Check if tool calls are needed
        if response.tool_calls:
            # Execute tool calls
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call["name"] == "search":
                    result = search.invoke(tool_call["args"])
                    tool_results.append(result)
            
            # Add tool results to messages and get final response
            messages.append(("assistant", response.content or ""))
            messages.append(("user", f"Search results: {' '.join(tool_results)}"))
            final_response = agent.invoke(messages)
            reply = final_response.content
        else:
            reply = response.content
        
        print("\nAgent:", reply)
        
        # Update chat history
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(reply)

if __name__ == "__main__":
    main()