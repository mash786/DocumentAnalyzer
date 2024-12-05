import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import docx  # python-docx for Word file extraction
import google.generativeai as genai
import asyncio
from PIL import Image
import pytesseract
import io
import pandas as pd  # To create tabular data

# Configure Tesseract OCR path (adjust for your OS)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Securely fetch API key from Streamlit secrets
api_key = st.secrets["general"]["api_key"]
genai.configure(api_key=api_key)

# Streamlit interface
def main():
    st.set_page_config(page_title="Dynamic AI-Driven Document Analysis", page_icon=":robot_face:", layout="wide")
    st.title("Dynamic AI-Driven Document Analysis System")
    st.markdown("""
    This app allows you to upload multiple documents (PDF or Word),
    and ask questions based on their content. Each document is analyzed separately.
    """)

    # File upload section
    st.subheader("Step 1: Upload Your Documents")
    uploaded_files = st.file_uploader("Upload Documents (PDF, Word)", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            document_chunks, extracted_text = process_files_concurrently(uploaded_files)

        # Cache the results so that the file processing doesn't happen again
        st.session_state.document_chunks = document_chunks
        st.session_state.extracted_text = extracted_text

        # Question Input
        st.subheader("Step 2: Ask Your Custom Question(s)")
        custom_question = st.text_area("Enter your question(s) related to the documents", placeholder="What would you like to know?", height=100)

        # Buttons for generating answers
        col1, col2 = st.columns([1, 1])

        with col1:
            generate_normal_button = st.button("Generate Answer (Normal Form)")

        with col2:
            generate_tabulated_button = st.button("Generate Answer (Tabulated Form)")

        if generate_normal_button or generate_tabulated_button:
            if custom_question:
                questions = [q.strip() for q in custom_question.split("\n") if q.strip()]
                answers = generate_answers_for_multiple_questions(questions, document_chunks, uploaded_files)

                if generate_normal_button:
                    st.subheader("Answers (Normal Form):")
                    display_answers_normal(answers, uploaded_files)
                elif generate_tabulated_button:
                    st.subheader("Answers (Tabulated Form):")
                    display_answers_tabulated(answers, uploaded_files)
            else:
                st.warning("Please enter a question.")

# Display answers in normal format
def display_answers_normal(answers, uploaded_files):
    for question, relevant_answers in answers.items():
        st.markdown(f"### Question: {question}")
        if relevant_answers:
            for idx, answer in relevant_answers.items():
                st.markdown(f"**From Document {idx + 1} ({uploaded_files[idx].name}):**\n{answer}\n")
        else:
            st.markdown("No relevant answers found.")

# Display answers in tabulated format
def display_answers_tabulated(answers, uploaded_files):
    # Prepare the data for the table
    table_data = []
    for question, relevant_answers in answers.items():
        if relevant_answers:
            for idx, answer in relevant_answers.items():
                table_data.append([f"Document {idx + 1} ({uploaded_files[idx].name})", question, answer])

    # Create the dataframe
    df = pd.DataFrame(table_data, columns=["Document", "Question", "Answer"])
    
    # Display the table
    st.write(df)

# Asynchronously extract text from PDF with OCR support and chunking
def extract_pdf_text(uploaded_file, chunk_size=5000):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pdf_text = ""
    extracted_text = ""

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            pdf_text += text
            extracted_text += f"Page {page_num + 1}:\n{text}\n\n"

            # Extract text from images on the page
            images = page.get_images(full=True)
            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image)
                    pdf_text += ocr_text
                    extracted_text += f"Image OCR (Page {page_num + 1}):\n{ocr_text}\n\n"
                except Exception:
                    continue  # Skip problematic images
        except Exception:
            continue  # Skip problematic pages

    return chunk_text(pdf_text, chunk_size), extracted_text

# Asynchronously extract text from Word documents with chunking
def extract_docx_text(uploaded_file, chunk_size=5000):
    doc = docx.Document(uploaded_file)
    doc_text = "".join(para.text + "\n" for para in doc.paragraphs)
    return chunk_text(doc_text, chunk_size), doc_text

# Chunk text into manageable pieces
def chunk_text(text, chunk_size=5000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Cache the file processing to avoid re-running on every interaction
@st.cache_data
def process_files_concurrently(uploaded_files):
    """Process multiple files concurrently."""
    document_chunks = []
    extracted_text = ""
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            chunks, text = extract_pdf_text(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            chunks, text = extract_docx_text(uploaded_file)
        else:
            chunks, text = [], ""
        document_chunks.append(chunks)
        extracted_text += text

    return document_chunks, extracted_text

# Perform text analysis using Google Gemini API
def perform_analysis(question, chunk):
    context = f"Document Content:\n\n{chunk}\n\nQuestion: {question}\nAnswer the question based on the above document content:"
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(context)
        return response.text.strip()
    except Exception as e:
        return f"Error occurred: {e}"

# Generate answers for multiple questions and documents
def generate_answers_for_multiple_questions(questions, document_chunks, uploaded_files):
    all_answers = {}
    for question in questions:
        relevant_answers = {}

        # Keep track of documents already processed for the question
        processed_docs = set()

        for doc_idx, chunks in enumerate(document_chunks):
            if doc_idx in processed_docs:
                continue  # Skip this document if it already provided an answer for the question

            for chunk in chunks:
                answer = perform_analysis(question, chunk)
                if answer:
                    relevant_answers[doc_idx] = answer
                    processed_docs.add(doc_idx)  # Mark this document as processed for the question
                    break  # Break once an answer is found for the document

        # If answers are found, add them; otherwise, return None
        all_answers[question] = relevant_answers if relevant_answers else None

    return all_answers

# Run the app
if __name__ == "__main__":
    main()
