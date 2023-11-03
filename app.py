import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv

# Initialize langchain and OpenAI components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA

# Ensure a directory exists for storing documents
docs_dir = 'docs'
os.makedirs(docs_dir, exist_ok=True)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Read and textify PDFs function
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            text_list.append(text)
            sources_list.append(file.name + "_page_" + str(i+1))  # Start page count at 1
        file.close()  # Close the file after reading
    return [text_list, sources_list]

# Save uploaded file function
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(docs_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False

# Generate file link function
def generate_file_link(file_name):
    # Replace spaces with %20 for URL encoding
    safe_file_name = file_name.replace(' ', '%20')
    return f'./{docs_dir}/{safe_file_name}'

# Streamlit app layout configuration
st.set_page_config(layout="centered", page_title="DOXS")
st.sidebar.image('logo-3.png')
st.header("DOXS")
st.write("---")

# File uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.write(f"{len(uploaded_files)} document(s) loaded..")
    for uploaded_file in uploaded_files:
        if save_uploaded_file(uploaded_file):
            st.success(f"Saved file: {uploaded_file.name}")
        else:
            st.error(f"Failed to save file: {uploaded_file.name}")
else:
    st.info("Upload files to analyse.")

# If files are uploaded, process them
if uploaded_files:
    saved_files_paths = [os.path.join(docs_dir, f.name) for f in uploaded_files]
    saved_files = [open(file_path, "rb") for file_path in saved_files_paths]
    textify_output = read_and_textify(saved_files)

    documents = textify_output[0]
    sources = textify_output[1]

    # Extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k': 2}

    llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    retriever = vStore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Create the chain to answer questions
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
                
                # Display the source document/part where the answer was derived from
                st.subheader('Source Document/Part:')
                source_text = result['source_documents'][0].page_content.replace('\n', ' ')
                st.write(source_text)
                source_string = result['source_documents'][0].metadata['source']
                file_name, page = source_string.rsplit('_', 1)
                st.subheader('Page Resources')
                # Generating HTML link to open the PDF
                file_path = generate_file_link(file_name)
                st.markdown(f"File: {file_name}, Page: [{page}]({file_path})", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
