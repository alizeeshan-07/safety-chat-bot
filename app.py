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
import re
import os
import fitz  # PyMuPDF
import io

# WARNING: Including API keys directly in the code is not recommended for security reasons!
# Ideally, use environment variables or secure vaults to store sensitive information.


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

def reconstruct_paragraphs(text):
    # This pattern looks for occurrences of two or more newline characters
    # which might indicate the separation between paragraphs.
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Alternatively, if paragraphs in the text are separated by more than one newline character,
    # you can adjust the regex to match that specific pattern.
    # For example, if there's always at least three newlines between paragraphs, you can use:
    # text = re.sub(r'\n{3,}', '\n\n', text)

    # If the paragraphs are separated by a specific pattern of whitespace characters (e.g., spaces or tabs),
    # you would need to identify that pattern and replace it accordingly.
    # For instance, if there are usually four or more spaces indicating a new paragraph, use:
    # text = re.sub(r' {4,}', '\n\n', text)

    # If none of the above methods work due to the inconsistency of patterns,
    # you may need to manually insert paragraph breaks or use a more advanced text extraction library.

    return text.strip()  # Remove leading and trailing whitespace

def get_pdf_page_text(pdf_stream, page_number):
    # Open the PDF file from the bytes stream
    pdf = fitz.open(stream=pdf_stream)
    # Get the specified page
    page = pdf.load_page(page_number - 1)  # page numbers start from 0
    # Get the text of the page
    text = page.get_text()
    # Close the PDF file
    pdf.close()
    # Return the text data
    return text

def read_and_textify(files):
    text_list = []
    sources_list = []
    file_streams = {}  # Dictionary to hold file streams
    for file in files:
        file.seek(0)  # Reset the file stream to the start
        file_stream = file.read()
        file_streams[file.name] = file_stream  # Store the file stream
        if str(file.name).endswith('.pdf'):
            pdf = fitz.open(stream=file_stream)
            for i in range(len(pdf)):
                text = get_pdf_page_text(io.BytesIO(file_stream), i + 1)
                text_list.append(text)
                sources_list.append(file.name + "_page_" + str(i))
            pdf.close()
        else:
            # Handle other file types (e.g., .txt) if necessary
            pass
    return text_list, sources_list, file_streams

def pdf_page_to_image(pdf_stream, page_index):
    # Open the PDF file from the byte stream
    pdf = fitz.open(stream=pdf_stream)
    
    # Get the specified page using 0-based indexing
    page = pdf[page_index]  # Use indexing instead of load_page for simplicity
    
    # Render page to an image (pixmap)
    pix = page.get_pixmap()
    
    # Convert the pixmap to an image in PNG format
    img_data = pix.tobytes("png")
    
    # Close the PDF file
    pdf.close()
    
    # Return the image data
    return img_data
st.set_page_config(layout="centered", page_title="DOXS")
st.sidebar.image('logo-3.png')
st.header("DOXS")
st.write("---")
  
# file uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])

if uploaded_files is None:
    st.info(f"""Upload files to analyse""")
elif uploaded_files:
    st.write(str(len(uploaded_files)) + " document(s) loaded..")
    textify_output = read_and_textify(uploaded_files)
    documents, sources, file_streams = textify_output 

    # textify_output = read_and_textify(uploaded_files)
    # documents = textify_output[0]
    # sources = textify_output[1]
  
    # extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k': 2}

    llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    retriever = vStore.as_retriever(search_type="similarity", search_kwargs={"k":1})

# create the chain to answer questions
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
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
                # source_text = result['source_documents']
                source_text = result['source_documents'][0].page_content.replace('\n', ' ')
                source_text = reconstruct_paragraphs(source_text)
                with st.expander("Show source text"):
                    st.text_area("", source_text, height=300, disabled=False)

                # Use the file_streams dictionary to access the byte stream
                for doc in result['source_documents']:
                    source_string = doc.metadata.get('source', 'Unknown')
                    file_name, page_str = source_string.rsplit('_page_', 1)
                    page_number = int(page_str)  # Extract the page number from the metadata
                    
                    # If the page number from the metadata is 0, we assume it should be 1
                    # since PDF pages typically start at 1.
                    # if page_number == 0:
                    #     page_number = 1
                    
                    # Adjust for 0-based indexing for PyMuPDF
                    page_index = page_number
                    
                    # Check if the file stream is stored
                    if file_name in file_streams:
                        # Convert the byte stream back to a BytesIO object
                        byte_stream = io.BytesIO(file_streams[file_name])
                        # Reset the stream position
                        byte_stream.seek(0)
                        
                        # Debug: Confirm the page index
                        st.write(f"Page = {page_index}")
                        
                        # Generate the image data
                        img_data = pdf_page_to_image(byte_stream, page_index)
                        
                        # Display the image
                        st.image(img_data, caption=f'Page {page_number} of {file_name}')
                    else:
                        st.error(f"Could not find the file stream for {file_name}.")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')
