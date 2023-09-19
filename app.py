from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import concurrent.futures  # For parallel processing
import cProfile  # For profiling

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_pdf(pdf):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        text = executor.submit(extract_text_from_pdf, pdf).result()
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        text = process_pdf(pdf)
        
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
            # Reduce Similarity Searches: Limit the number of searches
            docs = knowledge_base.similarity_search(user_question)[:10]  # Adjust the limit as needed
            
            # Profiling and Optimization: Profile the code for optimization
            profiler = cProfile.Profile()
            profiler.enable()
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
            
            profiler.disable()
            st.write(response)
            
            # Profiling and Optimization: Print profiling results
            profiler.print_stats(sort='cumulative')

if __name__ == '__main__':
    main()


