import os

# Check if the PDF file exists in the current directory
pdf_filename = 'Patient-PDF-Extentded.pdf'
if not os.path.isfile(pdf_filename):
    print(f"The file {pdf_filename} does not exist in the current directory.")
    exit()

# Ask the user for the page number they want to open
page_number = input("Enter the page number to open in 'myfile.pdf': ")

try:
    # Convert the input to an integer
    page_number = int(page_number)
except ValueError:
    # Handle the exception if the input is not an integer
    print("Please enter a valid page number.")
else:
    # Create a URL that points to the PDF at the specified page
    url = f"file://{os.path.abspath(pdf_filename)}#page={page_number}"
    
    # Output the URL
    print(f"Copy and paste the following link into your web browser to open the PDF at page {page_number}:\n{url}")
