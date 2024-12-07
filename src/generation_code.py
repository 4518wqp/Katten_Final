import re
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage

# At the top of generation.py
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Initialize chat history
chat_history = []

def extract_sections_from_input(user_input):
    """Extract section references from user input."""
    # Updated regex to match more section number formats
    pattern = r'\d{4}\.\d{2}(?:\([a-z]\))?(?:\([a-z0-9]+\))?'
    matches = re.findall(pattern, user_input)
    # For debugging
    print(f"Input text: {user_input}")
    print(f"Found sections: {matches}")
    return matches

def find_section_content(segments, section_refs=None, selected_sections=None):
    """Find content for specified section references in segments.
    
    Args:
        segments (list): List of all available segments
        section_refs (list, optional): List of section references from user input
        selected_sections (list, optional): List of sections selected via checkboxes
    """
    section_content = {}
    
    if selected_sections:
        # Use sections selected via checkboxes
        for section in selected_sections:
            section_ref = re.search(r'\d{1,4}\.\d{2}\([a-z]\)(?:\(\w+\))?', section['header'])
            if section_ref:
                section_content[section_ref.group(0)] = section['content']
    elif section_refs:
        # Use sections specified in user input
        for ref in section_refs:
            for segment in segments:
                if ref in segment['header']:
                    section_content[ref] = segment['content']
                    break
    
    return section_content

# Define the ChatPromptTemplate
template = """
Based on the following data, generate a response addressing the specified legal issue:
{content}

Response (format):
- Section: {section}
- Response:
"""

# Update this in generation_test2.py

def generate_response(segments, user_input, selected_sections=None):
    """
    Generate response based on segments and user input, with deduplication.
    
    Args:
        segments (list): List of dictionaries containing section information
        user_input (str): User's query or input
        selected_sections (list, optional): List of sections selected via checkboxes
    
    Returns:
        dict: Generated responses for each section, without duplicates
    """
    # Extract section references from user input
    section_refs = extract_sections_from_input(user_input)
    section_content = find_section_content(segments, section_refs)
    
    responses = {}
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    if section_content:  # If specific sections are found
        # Track content to prevent duplicates
        seen_content = set()
        
        # Sort sections to maintain consistent order
        sorted_sections = sorted(section_content.items())
        
        for section, content in sorted_sections:
            # Create a hash of the content to check for duplicates
            content_hash = hash(content)
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                
                # Create messages for this section
                messages = [
                    SystemMessage(content=f"You are a legal assistant specializing in USPTO trademark law. Focus on section {section}."),
                    HumanMessage(content=f"""
                        Based on this USPTO TMEP section, generate a response:

                        Section {section}:
                        {content}

                        Query: {user_input}

                        Please provide a focused response specific to this section.
                    """)
                ]
                
                try:
                    response = llm(messages)
                    responses[section] = response.content
                except Exception as e:
                    print(f"Error generating response for section {section}: {str(e)}")
                    continue
            
    else:
        # Handle follow-up questions
        messages = [
            SystemMessage(content="You are a legal assistant specializing in USPTO trademark law."),
            HumanMessage(content=f"""
                Based on the user's query, provide a relevant response:
                
                Query: {user_input}
                
                Please provide a clear and detailed response addressing the user's question.
            """)
        ]
        
        try:
            response = llm(messages)
            responses["general"] = response.content
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise e
    
    return responses