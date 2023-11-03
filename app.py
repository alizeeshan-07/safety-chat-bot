import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
import PyPDF2
from dotenv import load_dotenv
import urllib.parse
import re
import os

# WARNING: Including API keys directly in the code is not recommended for security reasons!
# Ideally, use environment variables or secure vaults to store sensitive information.


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
docs_folder = 'docs'  # Define the directory to store uploaded documents

# Ensure the 'docs' directory exists
if not os.path.exists(docs_folder):
    os.makedirs(docs_folder)

def save_uploaded_file(uploaded_file):
    with open(os.path.join(docs_folder, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(docs_folder, uploaded_file.name)

def read_and_textify(file_paths):
    text_list = []
    sources_list = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            pdfReader = PyPDF2.PdfReader(f)
            for i in range(len(pdfReader.pages)):
                pageObj = pdfReader.pages[i]
                text = pageObj.extract_text()
                pageObj.clear()
                text_list.append(text)
                sources_list.append(os.path.basename(file_path) + "_page_" + str(i))
    return [text_list, sources_list]

st.set_page_config(layout="centered", page_title="DOXS")
st.sidebar.image('logo-3.png')
st.header("DOXS")
st.write("---")

uploaded_files = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf"])

if not uploaded_files:
    st.info("Upload files to analyze.")
else:
    saved_file_paths = [save_uploaded_file(uploaded_file) for uploaded_file in uploaded_files]
    st.write(f"{len(saved_file_paths)} document(s) loaded..")

    textify_output = read_and_textify(saved_file_paths)
    documents = textify_output[0]
    sources = textify_output[1]

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k': 2}

    llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    retriever = vStore.as_retriever(search_type="similarity", search_kwargs={"k":4})

    rqa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")

    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = rqa({"query": user_q})
                st.subheader('Your response:')
                st.write(result["result"])

                st.subheader('Source Document/Part:')
                source_text = result['source_documents'][0].page_content.replace('\n', ' ')
                st.write(source_text)
                source_string = result['source_documents'][0].metadata.get('source', 'Unknown')
                file_name, page = source_string.rsplit('_', 1)

                # Correctly construct the path to the file stored in 'docs' directory
                local_path = os.path.join(docs_folder, file_name)
                # Encode the path for URL
                link_path = f"file://{urllib.parse.quote(os.path.abspath(local_path))}#page={page}"
                st.markdown(f"File: {file_name}, Page: [{page}]({link_path})")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
