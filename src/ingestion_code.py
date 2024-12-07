import os
import re
from pathlib import Path
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Access the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

def get_trademark_guide_path(filenumber="1200"):
    """Get the absolute path to the Trademark Guidebook PDF."""
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.absolute()
    guide_path = project_root / "Trademark Guidebook" / f"tmep-{filenumber}.pdf"
    
    if not guide_path.exists():
        raise FileNotFoundError(f"Trademark Guide not found at: {guide_path}")
    
    return str(guide_path)

def parse_subsections(filenumber="1200", toc=35213):
    """Parse PDF into subsections with detailed error handling."""
    try:
        file_path = './parsed_segments/'+filenumber+'.txt'
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                segments = eval(file.read())
        else:
            filename = get_trademark_guide_path(filenumber)
            print(f"Opening PDF file: {filename}")
            
            with pdfplumber.open(filename) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            if not text:
                raise ValueError("No text extracted from PDF")
            
            print(f"Total text length: {len(text)}")

            table_of_content = text[:toc]
            content_lines = table_of_content.split('\n')
            temp = str(int(filenumber.strip('0')))
            toc_sections = [line for line in content_lines if line.strip().startswith(temp)]
            
            print(f"Found {len(toc_sections)} TOC sections")

            if not toc_sections:
                raise ValueError("No sections found starting with "+temp)

            segments = []
            main_text = text[toc:]

            for i in range(len(toc_sections)):
                current_section = toc_sections[i]
                print(f"Processing section {i + 1}: {current_section[:50]}...")

                try:
                    if i < len(toc_sections) - 1:
                        next_section = toc_sections[i + 1]
                        start_pattern = re.escape(current_section)
                        end_pattern = re.escape(next_section)
                        match = re.search(f'({start_pattern})(.*?)({end_pattern})', main_text, re.DOTALL)
                        
                        if match:
                            content = match.group(1) + match.group(2)
                            segments.append({
                                'header': current_section.strip(),
                                'content': content.strip()
                            })
                            print(f"Successfully processed section {i + 1}")
                        else:
                            print(f"No match found for section {i + 1}")
                    else:
                        start_pattern = re.escape(current_section)
                        match = re.search(f'({start_pattern})(.*?)$', main_text, re.DOTALL)
                        
                        if match:
                            content = match.group(1) + match.group(2)
                            segments.append({
                                'header': current_section.strip(),
                                'content': content.strip()
                            })
                            print("Successfully processed last section")
                        else:
                            print("No match found for last section")

                except Exception as section_error:
                    print(f"Error processing section {i + 1}: {str(section_error)}")
                    continue

            if not segments:
                raise ValueError("No segments were successfully parsed")

            print(f"Successfully parsed {len(segments)} segments")

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write(str(segments))
        return segments

    except Exception as e:
        print(f"Error in parse_subsections: {str(e)}")
        raise Exception(f"Error parsing subsections: {str(e)}")

def create_vectordb(segments, filename):
    """Create vector database with error handling."""
    try:
        if not segments:
            raise ValueError("No segments provided for vector database creation")

        persist_directory = './chroma_local_db/'+filename
        collection_name = "Katten_Embed"+filename
        embeddings = OpenAIEmbeddings()
        
        if os.path.isdir(persist_directory) and bool(os.listdir(persist_directory)):
            existing_db = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            return existing_db
        else:
            os.makedirs(persist_directory, exist_ok=True)
            os.chmod(persist_directory, 0o777)
            
            vector_db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)
            vector_db.delete_collection()
            vector_db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)

            for segment in segments:
                content = segment['content']
                metadata = {'header': segment['header']}
                vector_db.add_texts([content], metadatas=[metadata])

            vector_db.persist()
            return vector_db

    except Exception as e:
        print(f"Error in create_vectordb: {str(e)}")
        raise Exception(f"Error creating vector database: {str(e)}")