import streamlit as st
from PyPDF2 import PdfReader
import os

def extract_text_from_page(pdf_path, page_number):
    try:
        reader = PdfReader(pdf_path)
        if page_number < len(reader.pages):
            page = reader.pages[page_number]
            return page.extract_text()
        else:
            return None
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app starts here
st.title('PDF Text Viewer')

# Folder containing PDF documents
docs_folder = 'docs'

# Ensure the docs folder exists and contains PDFs
if os.path.exists(docs_folder) and len(os.listdir(docs_folder)) > 0:
    # Get list of PDFs in the folder
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]
    pdf_file = st.selectbox('Select a PDF file:', pdf_files)
    page_number = st.number_input('Page number', min_value=1, value=1)

    if st.button('Show Page'):
        pdf_path = os.path.join(docs_folder, pdf_file)
        text = extract_text_from_page(pdf_path, page_number - 1)  # Page numbers are zero-indexed
        if text:
            st.text_area("Text content:", text, height=300)
        else:
            st.error('Page number out of range.')
else:
    st.error(f"Make sure the 'docs' folder exists and contains PDF files.")
