from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
@st.cache(resources={
    'pdf_reader': st.cache.Permanent(PdfReader)
})
def load_pdf_reader():
    return PdfReader
pdf_reader = load_pdf_reader()



from langchain.text_splitter import CharacterTextSplitter
@st.cache(resources={
    'text_splitter': st.cache.Permanent(CharacterTextSplitter)
})
def load_text_splitter():
    return CharacterTextSplitter
text_splitter = load_text_splitter()

from langchain.embeddings.openai import OpenAIEmbeddings
@st.cache(resources={
    'openai_embeddings': st.cache.Permanent(OpenAIEmbeddings)
})
def load_openai_embeddings():
    return OpenAIEmbeddings
openai_embeddings = load_openai_embeddings()


from langchain.vectorstores import FAISS
@st.cache(resources={
    'faiss': st.cache.Permanent(FAISS)
})
def load_faiss():
    return FAISS
faiss = load_faiss()

from langchain.chains.question_answering import load_qa_chain
@st.cache(resources={
    'qa_chain': st.cache.Permanent(load_qa_chain)
})
def load_question_answering_chain():
    return load_qa_chain
question_answering_chain = load_question_answering_chain()

from langchain.llms import OpenAI
@st.cache(resources={
    'openai': st.cache.Permanent(OpenAI)
})
def load_openai():
    return OpenAI
openai_instance = load_openai()

from langchain.callbacks import get_openai_callback
@st.cache(resources={
    'openai_callback': st.cache.Permanent(get_openai_callback)
})
def load_openai_callback():
    return get_openai_callback
openai_callback = load_openai_callback()


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
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
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()
