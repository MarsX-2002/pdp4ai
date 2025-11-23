from langchain_openai import ChatOpenAI
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
import time

load_dotenv()

serper = GoogleSerperAPIWrapper()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store generated memes
meme_history = []

@tool
def search_trump_news(query: str = "latest") -> str:
    """Search for latest news and events about Donald Trump."""
    full_query = f"Donald Trump {query} news 2025"
    return serper.run(full_query)

@tool
def generate_and_save_meme(prompt: str, style: str, filename: str) -> str:
    """
    Generate a meme image using DALL-E 3 and immediately save it locally.
    
    Args:
        prompt: Description of the meme to generate
        style: Style of the meme (e.g., 'classic meme', 'political cartoon', 'satirical', 'comic strip')
        filename: Name to save the file as (without extension, no spaces)
    
    Returns:
        Status message with saved file path
    """
    try:
        # Create a detailed prompt for DALL-E
        full_prompt = f"Create a {style} style image: {prompt}. Make it humorous and meme-worthy, suitable for social media sharing."
        
        print(f"   Prompt: {prompt[:100]}...")
        
        # Generate image
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=full_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        print(f"   ✓ Generated! URL: {image_url[:80]}...")
        
        # Immediately download and save
        os.makedirs("memes", exist_ok=True)
        
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean filename - remove spaces and special chars
        clean_filename = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in filename)
        filepath = f"memes/{clean_filename}_{timestamp}.png"
        
        with open(filepath, "wb") as f:
            f.write(image_response.content)
        
        print(f"   💾 Saved to: {filepath}")
        
        # Store in history
        meme_history.append({
            "prompt": prompt,
            "style": style,
            "filepath": filepath,
            "url": image_url,
            "timestamp": datetime.now().isoformat()
        })
        
        return f"SUCCESS: Meme saved to {filepath}"
    
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool
def list_generated_memes() -> str:
    """List all memes generated in this session with their local file paths."""
    if not meme_history:
        return "No memes generated yet."
    
    result = "\n" + "=" * 60 + "\n"
    result += "GENERATED MEMES\n"
    result += "=" * 60 + "\n"
    for i, meme in enumerate(meme_history, 1):
        result += f"\n{i}. Style: {meme['style']}\n"
        result += f"   File: {meme['filepath']}\n"
        result += f"   Prompt: {meme['prompt'][:80]}...\n"
        result += f"   Time: {meme['timestamp']}\n"
    result += "=" * 60
    
    return result

def create_my_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
    tools = [
        search_trump_news,
        generate_and_save_meme,
        list_generated_memes
    ]
    
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

def main():
    agent = create_my_agent()
    chat_history = InMemoryChatMessageHistory()
    
    print("=" * 60)
    print("TRUMP MEME GENERATOR")
    print("=" * 60)
    print("I'll search for latest Trump news and generate meme images!")
    print("All memes are automatically saved to the 'memes/' folder.")
    print("\nCommands:")
    print("  - 'start' - Search news and generate memes")
    print("  - 'list memes' - Show all generated memes")
    print("  - 'quit' - Exit")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit"]:
            if meme_history:
                print(f"\n✓ Generated {len(meme_history)} memes saved in 'memes/' folder")
            print("Goodbye!")
            break
        
        # Handle list command
        if user_input.lower() in ["list memes", "list", "show memes"]:
            result = list_generated_memes.invoke({})
            print(result)
            continue

        messages = [
            ("system", """You are an autonomous meme generator that creates political humor memes.

CRITICAL WORKFLOW - When user says "start":

1. SEARCH: Call search_trump_news to get recent Trump news

2. ANALYZE: From search results, identify 4 different interesting/newsworthy items

3. GENERATE 4 MEMES: For EACH news item, call generate_and_save_meme with:
   - A creative, funny prompt describing the meme
   - One of these styles (use each once):
     * "classic internet meme format"
     * "political cartoon editorial style"
     * "satirical pop art illustration"
     * "comic strip panel style"
   - A descriptive filename (no spaces, use underscores)
   
4. SUMMARIZE: After all 4 memes, tell user what you created

PROMPT WRITING RULES:
- Be specific about visual elements (people, objects, actions, setting)
- Make it humorous but appropriate
- Reference the actual news event clearly
- Include visual details for DALL-E to work with
- Keep it family-friendly

Example good prompt: "Donald Trump at a podium with an oversized chart showing upward arrows, pointing enthusiastically while confetti falls around him, with photographers taking pictures in the foreground"

Example good filename: "trump_chart_enthusiasm"

IMPORTANT:
- Generate ALL 4 memes without asking for permission
- Use generate_and_save_meme (which saves automatically)
- Each meme should be about a DIFFERENT news item
- Use DIFFERENT styles for variety
- Only speak after generating all memes""")
        ]
        
        # Add chat history
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(("assistant", msg.content))
        
        messages.append(("user", user_input))
        
        try:
            # Track iterations to prevent infinite loops
            max_iterations = 15
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                response = agent.invoke(messages)
                
                # If no tool calls and we have a text response, we're done
                if not response.tool_calls:
                    reply = response.content
                    if reply and reply.strip():
                        print("\n" + "=" * 60)
                        print(reply)
                        print("=" * 60)
                    break
                
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    try:
                        if tool_name == "search_trump_news":
                            print(f"\n🔍 Searching for Trump news...")
                            result = search_trump_news.invoke(tool_args)
                        elif tool_name == "generate_and_save_meme":
                            style = tool_args.get('style', 'meme')
                            print(f"\n🎨 Generating {style}...")
                            result = generate_and_save_meme.invoke(tool_args)
                            time.sleep(2)  # Rate limiting for DALL-E
                        elif tool_name == "list_generated_memes":
                            result = list_generated_memes.invoke({})
                        else:
                            result = f"Unknown tool: {tool_name}"
                        
                        tool_results.append(result)
                    except Exception as e:
                        error_msg = f"Error with {tool_name}: {str(e)}"
                        print(f"\n❌ {error_msg}")
                        tool_results.append(error_msg)
                
                # Add tool results and continue
                messages.append(("assistant", response.content or "Processing..."))
                messages.append(("system", f"Tool results: {' | '.join(tool_results)}. Continue generating remaining memes or provide final summary if all 4 are done."))
            
            # Update chat history with final state
            chat_history.add_user_message(user_input)
            if 'reply' in locals() and reply:
                chat_history.add_ai_message(reply)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()