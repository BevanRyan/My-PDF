from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Importing SessionState from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
from session_state import SessionState

# Adding a function to process the PDF in chunks
def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache(suppress_st_warning=True)
def get_qa_response(docs, user_question):
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
    return response

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Create a session state object
    state = SessionState.get(pdf=None)
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Save the PDF in session state
    if pdf is not None:
        state.pdf = pdf
    
    # Check if PDF is uploaded
    if state.pdf is not None:
        text = process_pdf(state.pdf)
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Get the response using cached function
            response = get_qa_response(docs, user_question)
            st.write(response)

if __name__ == '__main__':
    main()
