import streamlit as st
import requests
import pdfplumber
from io import BytesIO

st.title("IMDB PDF Dataset Reader")

# 🔴 YAHAN apni GitHub wali PDF ka RAW link paste karo
PDF_URL = "https://raw.githubusercontent.com/USERNAME/REPO_NAME/BRANCH/Imdb%20dataset.pdf"


@st.cache_data
def load_pdf_from_github(url):
    response = requests.get(url)
    response.raise_for_status()

    pdf_file = BytesIO(response.content)

    all_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    return all_text


try:
    text = load_pdf_from_github(PDF_URL)

    st.subheader("PDF Content (Preview)")
    st.text_area("Dataset Text", text, height=400)

except Exception as e:
    st.error("PDF load nahi ho pa rahi. Raw link check karo.")
    st.write(e)