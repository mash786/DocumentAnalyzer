import os
import openai
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF for PDF extraction
from PIL import Image
import pytesseract  # Tesseract OCR for image extraction
import io
import docx  # python-docx for Word file extraction
import asyncio
from collections import defaultdict

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the OpenAI API key directly
openai.api_key = st.secrets["general"]["api_key"]  # Replace with your OpenAI API key

# Streamlit interface
def main():
    st.set_page_config(page_title="Dynamic AI-Driven Document Analysis", page_icon=":robot_face:", layout="wide")

    st.title("Dynamic AI-Driven Document Analysis System")
    st.markdown("""This app allows you to upload multiple documents (PDF or Word),
    and ask questions based on their content. The documents will be combined, 
    and the AI will answer based on the combined text.""")

    st.subheader("Step 1: Upload Your Documents")
    uploaded_files = st.file_uploader("Upload Documents (PDF, Word)", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        st.subheader("Step 2: Ask Your Custom Question(s)")
        custom_question = st.text_area("Enter your question(s) related to the documents", placeholder="What would you like to know?", height=100)

        if st.button("Generate Answer"):
            if custom_question:
                with st.spinner("Processing uploaded files and generating answers..."):
                    # Extract keywords from the question
                    keywords = extract_keywords_from_question(custom_question)
                    # Process the files and extract relevant content
                    combined_text = process_files_with_keywords(uploaded_files, keywords)
                    questions = custom_question.split("\n")
                    # Generate answers concurrently for multiple questions
                    answers = asyncio.run(generate_answers_for_multiple_questions(questions, combined_text))
                    st.subheader("Answers:")

                    for question, answer in answers.items():
                        if answer:
                            st.markdown(f"**Question:** {question}\n\n**Answer:** {answer}")
                        else:
                            st.markdown(f"**Question:** {question} - No relevant answers found.")
            else:
                st.warning("Please enter a question.")

def extract_keywords_from_question(question):
    """Extract keywords from the custom question using simple heuristics."""
    common_words = set(["what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an", "of", "in", "to", "on", "for", "with", "and", "or"])
    keywords = [word.lower() for word in question.split() if word.lower() not in common_words]
    return keywords

def process_files_with_keywords(uploaded_files, keywords):
    """Process files and extract relevant text based on keywords."""
    def process_file(uploaded_file):
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            return extract_pdf_text_with_keywords(uploaded_file, keywords)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_docx_text_with_keywords(uploaded_file, keywords)
        return ""

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, uploaded_files))
    
    combined_text = "\n".join(results)
    return combined_text

def extract_pdf_text_with_keywords(uploaded_file, keywords):
    """Extract text from PDF and filter based on keywords."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    relevant_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        relevant_text += filter_text_by_keywords(raw_text, keywords)

        # Process images using OCR
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(img_pil)
            relevant_text += filter_text_by_keywords(ocr_text, keywords)

    return relevant_text[:5000]  # Truncate to optimize API calls

def extract_docx_text_with_keywords(uploaded_file, keywords):
    """Extract text from a DOCX file and filter based on keywords."""
    doc = docx.Document(uploaded_file)
    relevant_text = ""
    for para in doc.paragraphs:
        raw_text = para.text
        relevant_text += filter_text_by_keywords(raw_text, keywords)
    return relevant_text[:5000]

def filter_text_by_keywords(text, keywords):
    """Filter text to retain lines containing any of the specified keywords."""
    lines = text.splitlines()
    relevant_lines = [line for line in lines if any(keyword in line.lower() for keyword in keywords)]
    return "\n".join(relevant_lines)

async def generate_answers_for_multiple_questions(questions, combined_text):
    """Generate answers for multiple questions using the combined document content."""
    all_answers = {}
    for question in questions:
        answer = await perform_analysis(question, combined_text)
        all_answers[question] = answer
    return all_answers

async def perform_analysis(custom_question, combined_text):
    """Answer custom questions based on combined document content."""
    try:
        prompt = f"Here is the combined document content:\n\n{combined_text}\n\nQuestion: {custom_question}\nAnswer:"
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = chat_completion['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    main()
