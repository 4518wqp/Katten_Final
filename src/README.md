# Description
This folder contains the functions that will be called by the frontend (streamlit)

# retriever.py
- function: retrieve(filenumber="0600", toc=5500, topK = 5, query = "", methods = "vector")
- Arguments:
    - filenumber: the number of chapter that the client want to search from
    - toc: the end of 'Table of Contents' of that chapter. For example, chapter 0600's table of contents end at approx. 5500 characters from the start.
    - topK: the number of relevant subsections the client want to search
    - query: the query from the client
    - methods: "vector", "BM25", "RFF" are supported methods
- Return: segments (need to be passed directly to generation); results (top K related subsections)


# generation.py

function: generate_response(segments, user_input)

- segments: A list of dictionaries containing segment headers and their corresponding content. Each dictionary should have two keys: header (the title of the section) and content (the text of the section).
- user_input: A string containing the user's input, which includes references to specific sections (e.g., "601.01(c)(iii)") and any questions or requests for clarification related to legal issues.
- Return: A dictionary containing responses based on the user input, where each key is a section reference and its value is the generated response from the model. If there are follow-up questions that do not reference specific sections, the key "follow_up" will hold the response generated for those questions.

Usage Example:
segments = [
    {"header": "601.01(c)(iii)", "content": "Content related to subsection 601.01(c)(iii)."},
    {"header": "601.01(b)", "content": "Content related to subsection 601.01(b)."},
    # Additional segments...
]

user_input = "I think 601.01(c)(iii), 601.01(b) are appropriate for addressing the issue. Write me a response based on these subsections."

responses = generate_response(segments, user_input)
