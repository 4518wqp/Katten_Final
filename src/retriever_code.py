from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
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
import pdfplumber
import os
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from .hyperlink import find_link

from .ingestion_code import parse_subsections, create_vectordb

# Load environment variables
load_dotenv(override=True)

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Access the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

DIR_NAME = "src"

def retrieve(filenumber="1200", toc=35213, topK=5, query="", methods="vector", csv_path=f"{DIR_NAME}/hyperlink_df.csv"):
    """Retrieve relevant sections with enhanced error handling."""
    try:
        print(f"Starting retrieval with method: {methods}")
        
        segments = parse_subsections(filenumber, toc)
        if not segments:
            raise ValueError("No segments found in document")

        print(f"Creating vector database for {len(segments)} segments")
        
        vector_db = create_vectordb(segments, filenumber)

        result = None
        if methods == "vector":
            result = retrieve_subsection_vector(topK, vector_db, query, csv_path)
        elif methods == "BM25":
            result = retrieve_subsection_bm25(topK, vector_db, query, csv_path)
        elif methods == "RFF":
            result = retrieve_subsection_rff(topK, vector_db, query, csv_path)
        else:
            raise ValueError(f"Unknown retrieval method: {methods}")

        if not result:
            raise ValueError("No results returned from retrieval method")

        print("Retrieval completed successfully")
        return segments, result

    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        raise Exception(f"Retrieval error: {str(e)}")

def retrieve_subsection_vector(topK=5, vector_db=None, query="", csv_path=f"{DIR_NAME}/hyperlink_df.csv"):
    """Vector-based retrieval with hyperlink addition."""
    try:
        if not vector_db:
            raise ValueError("Vector database is not initialized")

        if not query:
            raise ValueError("Query is empty")
        
        # Get the top K results directly from the vector database
        search_k = max(1, min(topK, 10))
        results = vector_db.similarity_search(query, k=search_k)
        
        # Format results programmatically like BM25 and RFF do
        formatted_results = []
        for idx, doc in enumerate(results, 1):
            header_name = doc.metadata['header']
            hyperlink = find_link(csv_path, header_name)
            
            result = f"- Reference[{idx}]: \n"
            result += f"    - Subsection: {header_name}\n"
            result += f"    - Hyperlink: {hyperlink}\n"
            result += f"    - Summary of content: {doc.page_content[:500]}...\n"
            formatted_results.append(result)
        
        return "\n".join(formatted_results)

    except Exception as e:
        print(f"Error in retrieve_subsection_vector: {str(e)}")
        raise Exception(f"Vector retrieval error: {str(e)}")

def retrieve_subsection_bm25(topK=5, vector_db=None, query="", csv_path=f"{DIR_NAME}/hyperlink_df.csv"):
    """BM25-based retrieval with hyperlink addition."""
    try:
        docs = vector_db.get()
        texts = docs['documents']
        metadatas = docs['metadatas']
        
        tokenized_corpus = [doc.split() for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[-topK:][::-1]
        
        formatted_results = []
        for idx, i in enumerate(top_indices, 1):
            if scores[i] > 0:
                header_name = metadatas[i]['header']
                hyperlink = find_link(csv_path, header_name)  # Get hyperlink
                
                result = f"- Reference[{idx}]: \n"
                result += f"    - Subsection: {header_name}\n"
                result += f"    - Hyperlink: {hyperlink}\n"
                result += f"    - Summary of content: {texts[i][:500]}...\n"  # Truncate long content
                formatted_results.append(result)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        print(f"Error in retrieve_subsection_bm25: {str(e)}")
        raise Exception(f"BM25 retrieval error: {str(e)}")

def retrieve_subsection_rff(topK=5, vector_db=None, query="", csv_path=f"{DIR_NAME}/hyperlink_df.csv"):
    """RFF-based retrieval with hyperlink addition."""
    try:
        embeddings = OpenAIEmbeddings()
        docs = vector_db.get()
        texts = docs['documents']
        metadatas = docs['metadatas']
        
        rff = RBFSampler(n_components=100)
        
        # Get query embedding and transform
        query_embedding = embeddings.embed_query(query)
        query_embedding_normalized = normalize(np.array(query_embedding).reshape(1, -1))
        rff.fit(query_embedding_normalized)
        query_rff = rff.transform(query_embedding_normalized)
        
        scores = []
        for text in texts:
            doc_embedding = embeddings.embed_query(text)
            doc_normalized = normalize(np.array(doc_embedding).reshape(1, -1))
            doc_rff = rff.transform(doc_normalized)
            score = cosine_similarity(query_rff, doc_rff)[0][0]
            scores.append(score)
        
        top_indices = np.array(scores).argsort()[-topK:][::-1]
        formatted_results = []
        for idx, i in enumerate(top_indices, 1):
            if scores[i] > 0:
                header_name = metadatas[i]['header']
                hyperlink = find_link(csv_path, header_name)  # Get hyperlink
                
                result = f"- Reference[{idx}]: \n"
                result += f"    - Subsection: {header_name}\n"
                result += f"    - Hyperlink: {hyperlink}\n"
                result += f"    - Summary of content: {texts[i][:500]}...\n"  # Truncate long content
                formatted_results.append(result)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        print(f"Error in retrieve_subsection_rff: {str(e)}")
        raise Exception(f"RFF retrieval error: {str(e)}")
