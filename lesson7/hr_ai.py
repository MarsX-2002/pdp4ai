from langchain_openai import ChatOpenAI
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import json
from typing import Optional

load_dotenv()

serper = GoogleSerperAPIWrapper()

# Store resume data
resume_data = {
    "personal_info": {},
    "summary": "",
    "experience": [],
    "education": [],
    "skills": [],
    "hobbies": "",
    "certifications": [],
    "projects": []
}

@tool
def search(query: str) -> str:
    """Search the web for information about resume best practices, industry standards, job descriptions, etc."""
    return serper.run(query)

@tool
def save_personal_info(name: str, email: str, phone: str, location: str, linkedin: str = "") -> str:
    """Save personal information for the resume. ALL fields (name, email, phone, location) are REQUIRED."""
    resume_data["personal_info"] = {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "linkedin": linkedin
    }
    return f"✓ Personal information saved"

@tool
def save_summary(summary: str) -> str:
    """Save professional summary for the resume."""
    resume_data["summary"] = summary
    return f"✓ Professional summary saved"

@tool
def add_experience(company: str, position: str, start_date: str, end_date: str, responsibilities: str) -> str:
    """Add work experience entry to the resume. All fields are REQUIRED."""
    experience = {
        "company": company,
        "position": position,
        "start_date": start_date,
        "end_date": end_date,
        "responsibilities": responsibilities
    }
    resume_data["experience"].append(experience)
    return f"✓ Added experience: {position} at {company}"

@tool
def add_education(institution: str, degree: str, field: str, graduation_date: str, gpa: str = "") -> str:
    """Add education entry to the resume. Institution, degree, field, and graduation_date are REQUIRED."""
    education = {
        "institution": institution,
        "degree": degree,
        "field": field,
        "graduation_date": graduation_date,
        "gpa": gpa
    }
    resume_data["education"].append(education)
    return f"✓ Added education: {degree} in {field} from {institution}"

@tool
def add_skills(skills: str) -> str:
    """Add skills to the resume. Provide skills as comma-separated list."""
    skill_list = [s.strip() for s in skills.split(",")]
    resume_data["skills"].extend(skill_list)
    resume_data["skills"] = list(set(resume_data["skills"]))  # Remove duplicates
    return f"✓ Added {len(skill_list)} skills"

@tool
def add_hobbies(hobbies: str) -> str:
    """Add hobbies/interests to the resume."""
    resume_data["hobbies"] = hobbies
    return f"✓ Hobbies saved"

@tool
def add_project(name: str, description: str, technologies: str) -> str:
    """Add a project to the resume. All fields are REQUIRED."""
    project = {
        "name": name,
        "description": description,
        "technologies": technologies
    }
    resume_data["projects"].append(project)
    return f"✓ Added project: {name}"

@tool
def check_resume_completeness() -> str:
    """Check what sections of the resume are complete and what's missing."""
    status = {
        "personal_info": bool(resume_data["personal_info"]),
        "summary": bool(resume_data["summary"]),
        "experience": len(resume_data["experience"]),
        "education": len(resume_data["education"]),
        "skills": len(resume_data["skills"]),
        "hobbies": bool(resume_data["hobbies"]),
        "projects": len(resume_data["projects"])
    }
    return json.dumps(status)

@tool
def generate_resume() -> str:
    """Generate the final resume in formatted text."""
    resume = []
    
    # Personal Info
    if resume_data["personal_info"]:
        pi = resume_data["personal_info"]
        resume.append("\n" + "=" * 60)
        resume.append(f"{pi.get('name', '').upper()}")
        resume.append(f"{pi.get('email', '')} | {pi.get('phone', '')} | {pi.get('location', '')}")
        if pi.get('linkedin'):
            resume.append(f"LinkedIn: {pi.get('linkedin')}")
        resume.append("=" * 60)
        resume.append("")
    
    # Summary
    if resume_data["summary"]:
        resume.append("PROFESSIONAL SUMMARY")
        resume.append("-" * 60)
        resume.append(resume_data["summary"])
        resume.append("")
    
    # Experience
    if resume_data["experience"]:
        resume.append("WORK EXPERIENCE")
        resume.append("-" * 60)
        for exp in resume_data["experience"]:
            resume.append(f"{exp['position']} | {exp['company']}")
            resume.append(f"{exp['start_date']} - {exp['end_date']}")
            resume.append(exp['responsibilities'])
            resume.append("")
    
    # Education
    if resume_data["education"]:
        resume.append("EDUCATION")
        resume.append("-" * 60)
        for edu in resume_data["education"]:
            resume.append(f"{edu['degree']} in {edu['field']}")
            resume.append(f"{edu['institution']} | {edu['graduation_date']}")
            if edu.get('gpa'):
                resume.append(f"GPA: {edu['gpa']}")
            resume.append("")
    
    # Skills
    if resume_data["skills"]:
        resume.append("SKILLS")
        resume.append("-" * 60)
        resume.append(", ".join(resume_data["skills"]))
        resume.append("")
    
    # Projects
    if resume_data["projects"]:
        resume.append("PROJECTS")
        resume.append("-" * 60)
        for proj in resume_data["projects"]:
            resume.append(f"{proj['name']}")
            resume.append(f"{proj['description']}")
            resume.append(f"Technologies: {proj['technologies']}")
            resume.append("")
    
    # Hobbies
    if resume_data["hobbies"]:
        resume.append("HOBBIES & INTERESTS")
        resume.append("-" * 60)
        resume.append(resume_data["hobbies"])
        resume.append("")
    
    if not resume:
        return "Your resume is empty. Let's start building it!"
    
    return "\n".join(resume)

def show_current_resume():
    """Display current resume progress"""
    resume_text = generate_resume.invoke({})
    print("\n" + "=" * 60)
    print("CURRENT RESUME PROGRESS")
    print("=" * 60)
    print(resume_text)
    print("=" * 60)

def create_my_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [
        search,
        save_personal_info,
        save_summary,
        add_experience,
        add_education,
        add_skills,
        add_hobbies,
        add_project,
        check_resume_completeness,
        generate_resume
    ]
    
    # Bind tools to the model
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools

def main():
    agent = create_my_agent()
    chat_history = InMemoryChatMessageHistory()
    
    print("=" * 60)
    print("HR RESUME ASSISTANT")
    print("=" * 60)
    print("I'll guide you through creating a professional resume!")
    print("Just answer my questions and I'll handle the rest.")
    print("\nCommands:")
    print("  - 'quit' or 'exit' to exit")
    print("  - 'show resume' to see current progress")
    print("  - 'start' or 'begin' to start building resume")
    print("=" * 60)

    # Flag to track if we've started
    started = False

    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        if user_input.lower() in ["quit", "exit"]:
            print("\nGoodbye! Your resume data is saved in this session.")
            break
        
        # Handle show resume command
        if user_input.lower() in ["show resume", "show", "display resume"]:
            show_current_resume()
            continue
        
        # Auto-start on first message or explicit start
        if not started and (user_input.lower() in ["start", "begin", "let's start", "lets start"] or len(chat_history.messages) == 0):
            started = True
            user_input = "I'm ready to start building my resume"

        # Create messages with history
        messages = [
            ("system", """You are an expert HR assistant that AUTOMATICALLY guides users through creating a complete resume.

IMPORTANT FIRST INTERACTION:
- If this is the first message or user wants to start, greet them briefly and immediately ask for their full name
- Example: "Great! Let's build your resume. First, what is your full name?"

CRITICAL WORKFLOW RULES:
1. Check resume completeness using check_resume_completeness tool at the START of each response
2. Based on what's missing, automatically move to the next section WITHOUT asking user what they want
3. Follow this exact order: Personal Info → Education → Work Experience → Skills → Projects → Hobbies → Summary → Generate

SECTION COLLECTION PROCESS:
Personal Info:
- Ask for name, email, phone, location one by one
- For LinkedIn, if user says "no" or "skip", immediately save with empty LinkedIn and move to next section
- NEVER wait for user to say what's next - YOU control the flow

Education:
- Ask: institution, degree, field, graduation date
- For GPA, if they say "no" or skip, immediately save without GPA
- After adding, ask "Any more education to add?" If no, move to next section

Work Experience:
- Ask: company, position, start date, end date, responsibilities
- After adding, ask "Any more work experience?" If no, move to next section

Skills:
- Ask them to list skills (comma-separated)
- After saving, immediately move to next section

Projects (Optional):
- Ask "Do you have any projects to add?" 
- If no/skip, immediately move to next section
- If yes, collect: name, description, technologies

Hobbies (Optional):
- Ask "Any hobbies or interests?"
- If no/skip, immediately move to next section

Summary:
- Based on all collected info, write a professional summary
- Save it automatically
- Then immediately call generate_resume

HANDLING USER RESPONSES:
- "no", "skip", "none", "nope" = move to next section immediately
- After EVERY tool save, check completeness and continue automatically
- NEVER say "What would you like to add next?"
- NEVER wait for user direction

RESPONSE PATTERN:
After saving data: "✓ [Section] saved! Now, [ask first question of next section]"
Example: "✓ Personal info saved! Now let's add your education. What institution did you attend?"
""")
        ]
        
        # Add chat history
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                messages.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(("assistant", msg.content))
        
        # Add current user message
        messages.append(("user", user_input))
        
        try:
            # Invoke the agent
            response = agent.invoke(messages)
            
            # Check if tool calls are needed
            if response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    try:
                        # Execute the appropriate tool
                        if tool_name == "search":
                            result = search.invoke(tool_args)
                        elif tool_name == "save_personal_info":
                            result = save_personal_info.invoke(tool_args)
                        elif tool_name == "save_summary":
                            result = save_summary.invoke(tool_args)
                        elif tool_name == "add_experience":
                            result = add_experience.invoke(tool_args)
                        elif tool_name == "add_education":
                            result = add_education.invoke(tool_args)
                        elif tool_name == "add_skills":
                            result = add_skills.invoke(tool_args)
                        elif tool_name == "add_hobbies":
                            result = add_hobbies.invoke(tool_args)
                        elif tool_name == "add_project":
                            result = add_project.invoke(tool_args)
                        elif tool_name == "check_resume_completeness":
                            result = check_resume_completeness.invoke({})
                        elif tool_name == "generate_resume":
                            result = generate_resume.invoke({})
                            print("\n" + "=" * 60)
                            print("YOUR COMPLETE RESUME")
                            print("=" * 60)
                            print(result)
                            print("\n" + "=" * 60)
                            print("\n✓ Resume generated successfully!")
                            print("You can type 'show resume' to view it again anytime.")
                            continue
                        else:
                            result = f"Unknown tool: {tool_name}"
                        
                        tool_results.append(f"{result}")
                    except Exception as e:
                        tool_results.append(f"Error with {tool_name}: {str(e)}")
                
                # Add tool results to messages and get final response
                messages.append(("assistant", response.content or ""))
                messages.append(("user", f"[SYSTEM: Tools executed - {' | '.join(tool_results)}. Now check completeness and continue to next section automatically.]"))
                final_response = agent.invoke(messages)
                reply = final_response.content
            else:
                reply = response.content
            
            if reply and reply.strip():
                print("\nAssistant:", reply)
            
            # Update chat history
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(reply)
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's continue. Please answer the question.")

if __name__ == "__main__":
    main()
