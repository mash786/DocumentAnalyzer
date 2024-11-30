import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import docx  # python-docx for Word file extraction
import google.generativeai as genai

# Manually pass the API key for Google Gemini API
genai.configure(api_key="Your API Key")

# Streamlit interface
def main():
    st.set_page_config(page_title="Dynamic AI-Driven Document Analysis", page_icon=":robot_face:", layout="wide")

    st.title("Dynamic AI-Driven Document Analysis System")
    st.markdown("""
    This app allows you to upload different types of documents (PDF or Word), 
    and get meaningful analysis, including keyword extraction, summarization, and more.
    """)
    
    # File upload section
    st.subheader("Step 1: Upload Your Document")
    uploaded_file = st.file_uploader("Upload a Document (PDF, Word)", type=["pdf", "docx"], label_visibility="collapsed")
    
    if uploaded_file:
        # Document Processing
        with st.spinner("Processing your document..."):
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                document_text = extract_pdf_text(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                document_text = extract_docx_text(uploaded_file)
        
        # Document Preview
        st.subheader("Document Content Preview")
        st.text_area("Preview of Document Content", document_text[:1500], height=300)

        # Custom Question Input
        st.subheader("Step 2: Ask Your Custom Question")
        custom_question = st.text_area("Enter your question related to the document", placeholder="What would you like to know?", height=100)

        if st.button("Generate Answer"):
            if custom_question:
                with st.spinner("Generating answer..."):
                    answer = perform_analysis(custom_question, document_text)
                    st.subheader("Answer:")
                    st.write(answer)
            else:
                st.warning("Please enter a question.")

# Extract text from PDF function
def extract_pdf_text(uploaded_file):
    """Extract text from PDF using PyMuPDF (fitz)."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pdf_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pdf_text += page.get_text("text")
    return pdf_text

# Extract text from Word document function
def extract_docx_text(uploaded_file):
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(uploaded_file)
    doc_text = ""
    for para in doc.paragraphs:
        doc_text += para.text + "\n"
    return doc_text

# Perform text analysis using Google Gemini API
def perform_analysis(custom_question, document_text):
    """Answer custom questions based on document content using Google Gemini API."""
    try:
        # Prepare the context for analysis
        context = f"Here is the document content:\n\n{document_text}\n\nQuestion: {custom_question}\nAnswer:"

        # Using Google Gemini API to generate the analysis
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(context)

        # Return the response's text
        return response.text
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Run the app
if __name__ == "__main__":
    main()
