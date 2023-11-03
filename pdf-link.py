import streamlit as st
import PyPDF2
from io import BytesIO

def extract_text_from_pdf(uploaded_file):
    """Extract text from the uploaded PDF."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    total_pages = len(pdf_reader.pages)
    
    # Extract text from each page
    text_by_page = [pdf_reader.getPage(i).extractText() for i in range(total_pages)]
    return text_by_page
    
    # Extract text from each page
    text_by_page = [pdf_reader.getPage(i).extractText() for i in range(total_pages)]
    return text_by_page

def search_in_pdf(query, text_by_page):
    """Search for a query in the extracted PDF text and return the page number."""
    for index, page_text in enumerate(text_by_page):
        if query.lower() in page_text.lower():
            return index
    return None

# Streamlit UI
st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    st.write("Uploaded PDF!")
    text_by_page = extract_text_from_pdf(uploaded_file)
    
    query = st.text_input("Ask a question about the uploaded PDF:")
    
    if query:
        page_number = search_in_pdf(query, text_by_page)
        
        if page_number is not None:
            st.write(f"The answer can be found on page {page_number + 1} of the PDF.")
        else:
            st.write("Sorry, I couldn't find an answer to that in the PDF.")

if __name__ == "__main__":
    pass
