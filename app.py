from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Import SessionState class
class SessionState:
    def __init__(self):
        self.pdf = None
        self.text_extracted = False
        self.embedding_done = False

# Function to create OpenAI object (cached)
@st.cache(suppress_st_warning=True)
def create_openai():
    return OpenAI()

# Function to extract text from PDF (cached)
@st.cache(suppress_st_warning=True)
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks (cached)
@st.cache(suppress_st_warning=True)
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create knowledge base (cached)
@st.cache(suppress_st_warning=True)
def create_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Create a session state object
    state = SessionState()
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Save the PDF in session state
    if pdf is not None:
        state.pdf = pdf
    
    # Check if PDF is uploaded
    if state.pdf is not None:
        # Extract text from PDF (cached)
        text = extract_text_from_pdf(state.pdf)
        
        # Set text extraction completion in session state
        state.text_extracted = True
        
        # Check if text extraction is complete
        if state.text_extracted:
            # Split text into chunks (cached)
            chunks = split_text_into_chunks(text)
            
            # Set embedding completion in session state
            state.embedding_done = True
            
            # Check if embeddings are done
            if state.embedding_done:
                # Create OpenAI object (cached)
                llm = create_openai()
                
                # Create knowledge base (cached)
                knowledge_base = create_knowledge_base(chunks)
                
                # show user input
                user_question = st.text_input("Ask a question about your PDF:")
                if user_question:
                    docs = knowledge_base.similarity_search(user_question)
                    
                    chain = load_qa_chain(llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        print(cb)
                    
                    st.write(response)

if __name__ == '__main__':
    main()
