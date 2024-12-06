import streamlit as st
import sys
from pathlib import Path
import PyPDF2
from typing import List, Dict
import json
from datetime import datetime
import pprint
import uuid

# Basic setup
st.set_page_config(page_title="RebutBot", page_icon="📝", layout="wide")

# Setup paths and imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.absolute() 
function_calls_dir = parent_dir / "function_calls"
sys.path.append(str(function_calls_dir))

from ingestion_code import parse_subsections
from retriever_code import retrieve
from generation_code import generate_response, extract_sections_from_input

class SessionState:
    def __init__(self):
        self.messages: List[Dict] = []
        self.office_action_text: str = None
        self.current_sections: List[Dict] = []

def init_session_state():
    if 'state' not in st.session_state:
        st.session_state.state = SessionState()
    return st.session_state.state

def debug_print_state(state: SessionState, prefix: str = ""):
    """
    Print the current state of the session for debugging purposes.
    
    Args:
        state: The current SessionState object
        prefix: Optional prefix to identify the point where debug was called
    """
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'point': prefix,
        'state_summary': {
            'messages_count': len(state.messages),
            'has_office_action': bool(state.office_action_text),
            'current_sections_count': len(state.current_sections),
            'messages': [{'role': m['role'], 'content_preview': m['content'][:100] + '...' 
                         if len(m['content']) > 100 else m['content']} 
                        for m in state.messages],
        }
    }

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return " ".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def display_chat_message(message: Dict, state: SessionState):
    role = message["role"]
    content = message["content"]
    
    # Display message with appropriate styling
    st.markdown(f"""
        <div class="streamlit-chat-message" data-role="{role}">
            {content}
        </div>
    """, unsafe_allow_html=True)
    
    # Display sections if present
    if "sections" in message:
        display_sections(message["sections"], state)

def display_sections(sections: List[Dict], state: SessionState):
    """Display retrieved sections with improved UI using navy blue theme"""
    st.markdown("### Retrieved Sections")
    
    if isinstance(sections[0], str):
        # Extract individual references
        content = sections[0].strip('[]"').replace('\\n', '\n')
        references = content.split("- Reference[")
        
        for ref in references[1:]:  # Skip the first empty split
            if not ref.strip():
                continue
            
            try:
                # Split into subsection, hyperlink, and summary
                ref_num = ref.split(']:')[0] if ']' in ref else ""
                
                subsection = ""
                hyperlink = ""
                summary = ""
                
                # Parse each section separately for better reliability
                if "- Subsection:" in ref:
                    subsection = ref.split("- Subsection:", 1)[1].split("- Hyperlink:")[0].strip()
                
                if "- Hyperlink:" in ref:
                    hyperlink = ref.split("- Hyperlink:", 1)[1].split("- Summary of content:")[0].strip()
                
                if "- Summary of content:" in ref:
                    summary = ref.split("- Summary of content:", 1)[1].strip()
                
                # Create a clean, organized display using a container with navy blue accents
                with st.container():
                    st.markdown(f"""
                        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6;'>
                            <h4 style='color: #0B3D91; margin-bottom: 0.5rem;'>Reference {ref_num}</h4>
                            <div style='margin-left: 1rem;'>
                                <p style='font-weight: bold; margin-bottom: 0.5rem; color: #0B3D91;'>Subsection:</p>
                                <p style='margin-left: 1rem; margin-bottom: 1rem;'>{subsection}</p>
                                <p style='font-weight: bold; margin-bottom: 0.5rem; color: #0B3D91;'>Hyperlink:</p>
                                <p style='margin-left: 1rem; margin-bottom: 1rem;'>
                                    <a href="{hyperlink}" target="_blank" style="color: #0B3D91; text-decoration: none;">{hyperlink}</a>
                                </p>
                                <p style='font-weight: bold; margin-bottom: 0.5rem; color: #0B3D91;'>Summary:</p>
                                <p style='margin-left: 1rem;'>{summary}</p>
                            </div>
                        </div>
                        <div style='margin-bottom: 1rem;'></div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error parsing reference: {e}")

        # Display help text with navy blue styling
        st.markdown("""
            <div style='background-color: #f0f7ff; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;'>
                <p style='color: #0B3D91; margin: 0;'>
                    To generate a response, please specify the section numbers in your query (e.g., "Help me with sections 1210.04(a) and 1210.05(b)")
                </p>
            </div>
        """, unsafe_allow_html=True)

def handle_generation_response(prompt: str, state: SessionState):
    """
    Handle generation of responses while avoiding duplication.
    
    Args:
        prompt: User's input prompt
        state: Current session state
    """
    try:
        # Generate response
        response = generate_response(
            segments=state.current_sections,
            user_input=prompt
        )
        
        # Create a single formatted response string
        formatted_response = []
        for section, content in response.items():
            if section != "general":  # For section-specific responses
                formatted_response.append(f"#### Section {section}\n{content}")
            else:  # For general follow-up questions
                formatted_response.append(content)
        
        formatted_content = "\n\n".join(formatted_response)
        
        # Add to message history once
        assistant_response = {
            "role": "assistant",
            "content": formatted_content
        }
        state.messages.append(assistant_response)
        
        # Display download buttons
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}"
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download as JSON",
                data=json.dumps(response, indent=4),
                file_name=f"{filename}.json",
                mime="application/json",
                key=f"download_json_{uuid.uuid4()}"
            )
        with col2:
            st.download_button(
                "Download as Text",
                data=formatted_content,
                file_name=f"{filename}.txt",
                mime="text/plain",
                key=f"download_text_{uuid.uuid4()}"
            )
            
        return True
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.write("Error details:", e)
        debug_print_state(state, "Error Generating Response")
        return False


def main():
    st.title("RebutBot")
    
    # Add description message with navy blue styling to match your theme
    st.markdown("""
        <div style='background-color: #f0f7ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            <p style='color: #0B3D91; margin: 0; font-weight: 500;'>
                The current prototype specializes in USPTO TMEP §1200 responses.
            </p>
        </div>
    """, unsafe_allow_html=True)

    state = init_session_state()

    # Initialize session state variables
    if 'retrieval_method' not in st.session_state:
        st.session_state.retrieval_method = "vector"
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5

    # Sidebar configuration
    with st.sidebar:
        # Center the logo using columns
        left_col, center_col, right_col = st.columns([1, 6, 1])

        with center_col:
        # Add logo at the top of sidebar
            st.image("images/logo.png", width=200)  # Adjust width as needed
        
        st.header("Configuration")
        
        uploaded_file = st.file_uploader(
            "Upload Office Action Letter (PDF)",
            type="pdf",
            help="Upload the USPTO Office Action letter you want to analyze"
        )
        
        if uploaded_file and not state.office_action_text:
            state.office_action_text = extract_text_from_pdf(uploaded_file)
            if state.office_action_text:
                st.success("✅ Office Action letter processed successfully!")
                debug_print_state(state, "After PDF Upload")
        
        st.subheader("Retrieval Settings")
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["vector", "BM25", "RFF"],
            help="Select the method used to retrieve relevant sections"
        )
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant sections to retrieve"
        )

    # Main chat interface
    with st.container():
        # Display existing messages
        for message in state.messages:
            display_chat_message(message, state)
        
        # Debug current state before user input
        debug_print_state(state, "Before User Input")

        # Chat input
        if prompt := st.chat_input("Ask about specific issues or sections...", key="chat_input"):
            if not state.office_action_text:
                st.error("Please upload an Office Action letter first!")
                debug_print_state(state, "Error: No Office Action")
            else:
                # Add user message
                user_message = {"role": "user", "content": prompt}
                state.messages.append(user_message)
                
                # Debug state after user message
                debug_print_state(state, "After User Message")
                
                # Check if this is a section-specific query or a general query
                section_refs = extract_sections_from_input(prompt)
                
                if section_refs or "generate" in prompt.lower() or "help" in prompt.lower():
                    with st.spinner("Generating response..."):
                        success = handle_generation_response(prompt, state)
                        if success:
                            st.rerun()  # Refresh to show new messages

                else:
                    # This is a general query - do retrieval
                    with st.spinner("Searching relevant sections..."):
                        try:
                            segments, results = retrieve(
                                query=prompt,
                                methods=retrieval_method,
                                topK=top_k
                            )
                            
                            state.current_sections = segments
                            
                            # Create assistant response
                            assistant_message = {
                                "role": "assistant",
                                "content": "I found these relevant sections from the manual:",
                                "sections": [results]
                            }
                            state.messages.append(assistant_message)
                            
                            # Debug state after assistant response
                            debug_print_state(state, "After Assistant Response")
                            
                            # Force refresh to show new messages and sections
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                            st.write("Error details:", e)
                            debug_print_state(state, "Error During Search")

if __name__ == "__main__":
    main()