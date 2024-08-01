import streamlit as st
import PyPDF2
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize the sentence-transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    texts = []
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        texts.append((page_num, text))
    return texts

# Function to get embeddings using sentence-transformers
def get_embeddings(texts):
    embeddings = model.encode(texts)
    return np.array(embeddings)

# Function to compare two lists of embeddings
def compare_texts(texts1, texts2):
    comparisons = []
    for (page1, text1), (page2, text2) in zip(texts1, texts2):
        embeddings1 = get_embeddings([text1])
        embeddings2 = get_embeddings([text2])
        similarity = cosine_similarity(embeddings1, embeddings2)
        comparisons.append((page1, page2, similarity, text1, text2))
    return comparisons

# Function to summarize differences using GPT-3.5-turbo
def summarize_differences(api_key, comparisons):
    openai.api_key = api_key
    summaries = []
    for page1, page2, similarity, text1, text2 in comparisons:
        if np.mean(similarity) < 0.95:  # Threshold for displaying differences
            prompt = (
                f"Compare the following network configuration texts in detail, identifying changes in the new document. "
                f"Do not use 'Text 1' and 'Text 2', use the laptop names instead. Provide the text snippet or sentences "
                f"from the PDFs specifying the parameters that have changed, quantifying the changes, and highlighting "
                f"potential impacts on the network. Suggest necessary testing areas based on the changes. Provide references "
                f"or page numbers from the texts:\n\n"
                f"Laptop 1:\n{text1}\n\n"
                f"Laptop 2:\n{text2}\n\n"
                "Provide a clear and organized summary of the differences, noting the parameters and quantifying the changes. "
                "Include a snippet of the text with the differences highlighted and reference the text. Give the sentences in "
                "specific pages or sections from the PDFs. Highlight the potential impact on the network and suggest necessary testing areas."
            )
            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                prompt=prompt,
                max_tokens=500,
                temperature=0.5,
            )
            summary = response.choices[0].text.strip()
            summaries.append((page1, page2, summary))
    return summaries

# Function to answer questions about PDFs using GPT-3.5-turbo
def answer_question(api_key, question, texts):
    openai.api_key = api_key
    combined_text = "\n".join(text for _, text in texts)
    prompt = (
        f"Here are the contents of the PDF:\n\n{combined_text}\n\n"
        f"Question: {question}\n"
        "Provide a clear and concise answer based on the contents of the PDF."
    )
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=500,
        temperature=0.5,
    )
    answer = response.choices[0].text.strip()
    return answer

# Streamlit UI
st.title("PDF Comparison App")

api_key = st.text_input("Enter your OpenAI API Key", type="password")

uploaded_file1 = st.file_uploader("Choose the first PDF", type="pdf")
uploaded_file2 = st.file_uploader("Choose the second PDF", type="pdf")

if api_key and uploaded_file1 and uploaded_file2:
    if st.button('Compare PDFs'):
        with st.spinner("Extracting text from PDFs..."):
            texts1 = extract_text_from_pdf(uploaded_file1)
            texts2 = extract_text_from_pdf(uploaded_file2)
        
        with st.spinner("Comparing texts..."):
            comparisons = compare_texts(texts1, texts2)
        
        with st.spinner("Summarizing differences..."):
            summaries = summarize_differences(api_key, comparisons)
        
        st.subheader("Comparison Result")
        total_similarity = np.mean([np.mean(sim) for _, _, sim, _, _ in comparisons])
        st.write(f"Overall Similarity Score: {total_similarity:.4f}")
        
        st.subheader("Detailed Differences")
        for page1, page2, summary in summaries:
            st.write(f"Page {page1 + 1} vs Page {page2 + 1}")
            st.write(summary)
    
        st.success("Comparison completed!")

question = st.text_input("Ask a question about the PDFs:")
if api_key and question and uploaded_file1 and uploaded_file2:
    if st.button('Ask Question'):
        with st.spinner("Extracting text from PDFs..."):
            texts1 = extract_text_from_pdf(uploaded_file1)
            texts2 = extract_text_from_pdf(uploaded_file2)
        
        with st.spinner("Answering question..."):
            answer = answer_question(api_key, question, texts1 + texts2)
        
        st.subheader("Answer")
        st.write(answer)

else:
    if not api_key:
        st.error("Please enter your OpenAI API Key.")
    if not uploaded_file1 or not uploaded_file2:
        st.info("Please upload two PDF files to compare.")
    if question and (not uploaded_file1 or not uploaded_file2):
        st.info("Please upload two PDF files to ask a question.")
