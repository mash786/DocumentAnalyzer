import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import docx  # python-docx for Word file extraction
import google.generativeai as genai
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Access the API key securely from Streamlit's secrets management
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
            document_texts = process_files_concurrently(uploaded_files)

        # Question Input
        st.subheader("Step 2: Ask Your Custom Question(s)")
        custom_question = st.text_area("Enter your question(s) related to the documents", placeholder="What would you like to know?", height=100)

        if st.button("Generate Answer"):
            if custom_question:
                with st.spinner("Generating answers..."):
                    questions = custom_question.split("\n")  # Split the input into multiple questions
                    answers = asyncio.run(generate_answers_for_multiple_questions(questions, document_texts, uploaded_files))
                    st.subheader("Answers:")

                    # Display answers
                    for question, relevant_answers in answers.items():
                        if relevant_answers:
                            for idx, answer in relevant_answers.items():
                                st.markdown(f"**Answer from Document {idx + 1} ({uploaded_files[idx].name}):**\n{answer}\n")
                        else:
                            st.markdown(f"**Answer for Question:** {question} - No relevant answers found.")
            else:
                st.warning("Please enter a question.")

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    """Extract text from PDF using PyMuPDF (fitz)."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pdf_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pdf_text += page.get_text("text")
    return pdf_text[:5000]  # Truncate to 5000 characters to optimize API calls

# Extract text from Word document
def extract_docx_text(uploaded_file):
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(uploaded_file)
    doc_text = ""
    for para in doc.paragraphs:
        doc_text += para.text + "\n"
    return doc_text[:5000]  # Truncate to 5000 characters to optimize API calls

# Perform text analysis using Google Gemini API
async def perform_analysis(custom_question, document_text):
    """Answer custom questions based on document content using Google Gemini API."""
    try:
        context = f"Here is the document content:\n\n{document_text}\n\nQuestion: {custom_question}\nAnswer:"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await asyncio.to_thread(model.generate_content, context)
        return response.text
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Concurrently process files
def process_files_concurrently(uploaded_files):
    """Process multiple files concurrently."""
    def process_file(uploaded_file):
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            return extract_pdf_text(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_docx_text(uploaded_file)
        return ""

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, uploaded_files))
    return results

# Check document relevance and generate answers for each question
async def generate_answers_for_multiple_questions(questions, document_texts, uploaded_files):
    """Generate answers for multiple questions and relevant documents."""
    async def is_relevant(document_text, question):
        context = f"Document Content:\n\n{document_text}\n\nQuestion: {question}\nIs this question relevant to this document? Answer 'Yes' or 'No':"
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await asyncio.to_thread(model.generate_content, context)
            return "yes" in response.text.lower()
        except Exception:
            return False

    all_answers = {}
    for question in questions:
        relevant_answers = {}
        for idx, document_text in enumerate(document_texts):
            if await is_relevant(document_text, question):  # Await the relevance check
                answer = await perform_analysis(question, document_text)
                relevant_answers[idx] = answer
        if relevant_answers:
            all_answers[question] = relevant_answers
        else:
            all_answers[question] = None

    return all_answers

# Run the app
if __name__ == "__main__":
    main()
