# USPTO Office Action Letter Rebuttal Assistant

This Streamlit app helps users retrieve and summarize information from the TEMP manual using a RAG system to prepare responses to USPTO office action letters.

## Features
- Ask questions in a chat-based interface.
- Retrieve relevant TEMP manual sections and subsections using BM-25 and summarize content.
- Option to select and compile sections into a rebuttal.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone git@github.com:4518wqp/Katten_Capstone.git 
   cd Katten_Capstone

------- 
Xenia 

Step 1: Create a virtual environment with the specified python version 
python3.11 -m venv mycap

Step 2: activate your virtual environment 
source mycap/bin/activate

Step 3: upgrade pip install 
pip install --upgrade pip

Step 4: pip install the necessary libraries 
pip install -r requirements.txt

Step 5: Save a copy of the Trademark Guidebook in the main directory that has this exact name and the contents is a series of pdf documents of each TEMP section. 

Step 6: Export the OpenAI API and LangChain API keys and check that they were correctly set

export OPENAI_API_KEY = "key_value”
export LANGCHAIN_API_KEY = "key_value”

echo $OPENAI_API_KEY 
echo $LANGCHAIN_API_KEY

Step7: Run your streamlit app from the root directory 
streamlit run app.py 


