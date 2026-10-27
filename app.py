import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
import fitz
import streamlit as st

# Load environment variables from a .env file if present
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Document Review & Comparison Agent",
    page_icon="📄",
    layout="centered",
)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("Configuration")
env_api_key = os.environ.get("GEMINI_API_KEY", "")

api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=env_api_key,
    help="Loaded automatically from .env if configured.",
)

if api_key_input:
    os.environ["GEMINI_API_KEY"] = api_key_input

mode = st.sidebar.radio(
    "Select Analysis Mode", ["Single Document Review", "Two Document Comparison"]
)

MODEL = "gemini-2.5-flash"

# --- HELPER FUNCTIONS ---


def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


def chunk_text(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def call_gemini_with_retry(client, prompt, retries=3, delay=3):
    """Helper function to handle temporary 503 high demand errors with retry logic."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt)
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                    continue
            raise e


def analyze_chunk_review(client, chunk, idx):
    prompt = f"""
    You are a critical document reviewer.
    Analyze the following text chunk.
    Return:
    - Main claims
    - Assumptions
    - Any ambiguity or vagueness

    Chunk {idx}:
    {chunk}
    """
    response = call_gemini_with_retry(client, prompt)
    return response.text


def analyze_chunk_for_comparison(client, chunk, doc_label, idx):
    prompt = f"""
    You are analyzing a document chunk for cross-document comparison.
    Extract:
    - Key claims and any numbers or quantitative analysis (1-3 bullets)
    - Assumptions (explicit or implicit - if any)
    - Important terms or definitions
    
    Basically anything of importance in finding contradictions with other documents.
    Document: {doc_label}
    Chunk {idx}:
    {chunk}
    """
    response = call_gemini_with_retry(client, prompt)
    return response.text.strip()


# --- MAIN APP INTERFACE ---
st.title("📄 Document Intelligence & Comparison Agent")
st.write(
    "Analyze long documents for clarity gaps, redundancies, or perform deep cross-document comparison."
)

if not os.environ.get("GEMINI_API_KEY"):
    st.warning(
        "⚠️ API key not found. Please check your `.env` file or enter it in the sidebar."
    )
else:
    client = genai.Client()

    if mode == "Single Document Review":
        st.subheader("Stage 1: Single Document Deep Review")
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Run Document Review"):
                with st.spinner("Extracting and chunking text..."):
                    pdf_bytes = uploaded_file.read()
                    text = extract_text(pdf_bytes)

                    if not text:
                        st.error(
                            "❌ No extractable text found in this PDF. It might be a scanned image. Please use a text-based PDF."
                        )
                    else:
                        chunks = chunk_text(text)
                        st.info(
                            f"Document split into {len(chunks)} chunks. Analyzing..."
                        )

                        progress_bar = st.progress(0)
                        chunk_analyses = []

                        error_occurred = False
                        for i, ch in enumerate(chunks):
                            try:
                                analysis = analyze_chunk_review(client, ch, i)
                                chunk_analyses.append(analysis)
                                progress_bar.progress((i + 1) / len(chunks))
                            except Exception as e:
                                st.error(
                                    f"API Error on chunk {i}: {str(e)}"
                                )
                                error_occurred = True
                                break

                        if not error_occurred:
                            with st.spinner(
                                "Generating final comprehensive review..."
                            ):
                                final_prompt = f"""
                                You are reviewing a full document using chunk-level analyses.
                                Based on the analyses below, produce:
                                1. Overall summary (5-7 bullets)
                                2. Redundancies (what is repeated and where)
                                3. Internal conflicts or inconsistencies
                                4. Rewrite suggestions with difficulty level (Easy / Medium / Hard)
                                Chunk Analyses:
                                {chr(10).join(chunk_analyses)}
                                """
                                final_review = call_gemini_with_retry(
                                    client, final_prompt
                                )

                            st.success("Review complete!")

                            # HTML Report Rendering
                            html_stage1 = f"""
                            <div style="border: 1px solid rgba(128,128,128,0.3); border-radius: 14px; padding: 18px; margin: 16px 0; font-family: system-ui, sans-serif;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div style="font-size: 20px; font-weight: 600;">📋 Document Review Report</div>
                                    <div style="font-size: 13px; opacity: 0.7;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                                </div>
                                <div style="margin-top: 14px; line-height: 1.5; white-space: pre-wrap;">
                                    {final_review.text}
                                </div>
                            </div>
                            """
                            st.markdown(html_stage1, unsafe_allow_html=True)

    elif mode == "Two Document Comparison":
        st.subheader("Stage 2: Two-Document Cross Comparison")
        uploaded_files = st.file_uploader(
            "Upload TWO related PDF documents", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files:
            if len(uploaded_files) != 2:
                st.error("Please upload exactly TWO PDF documents.")
            else:
                if st.button("Compare Documents"):
                    with st.spinner("Processing both documents..."):
                        docA_text = extract_text(uploaded_files[0].read())
                        docB_text = extract_text(uploaded_files[1].read())

                        if not docA_text or not docB_text:
                            st.error(
                                "❌ One or both PDFs contain no extractable text."
                            )
                        else:
                            chunks_A = chunk_text(docA_text)
                            chunks_B = chunk_text(docB_text)

                            st.info(
                                f"Document A chunks: {len(chunks_A)} | Document B chunks: {len(chunks_B)}. Analyzing cross-references..."
                            )

                            total_chunks = len(chunks_A) + len(chunks_B)
                            progress_bar = st.progress(0)
                            current_step = 0

                            analyses_A = []
                            error_occurred = False
                            for i, ch in enumerate(chunks_A):
                                try:
                                    analyses_A.append(
                                        analyze_chunk_for_comparison(
                                            client, ch, "Document A", i
                                        )
                                    )
                                    current_step += 1
                                    progress_bar.progress(
                                        current_step / total_chunks
                                    )
                                except Exception as e:
                                    st.error(
                                        f"API Error in Doc A chunk {i}: {str(e)}"
                                    )
                                    error_occurred = True
                                    break

                            if not error_occurred:
                                analyses_B = []
                                for i, ch in enumerate(chunks_B):
                                    try:
                                        analyses_B.append(
                                            analyze_chunk_for_comparison(
                                                client, ch, "Document B", i
                                            )
                                        )
                                        current_step += 1
                                        progress_bar.progress(
                                            current_step / total_chunks
                                        )
                                    except Exception as e:
                                        st.error(
                                            f"API Error in Doc B chunk {i}: {str(e)}"
                                        )
                                        error_occurred = True
                                        break

                            if not error_occurred:
                                with st.spinner(
                                    "Synthesizing final comparison report..."
                                ):
                                    comparison_prompt = f"""
                                    You are comparing two documents.
                                    Analyze them and produce:
                                    1. Direct conflicts or contradictions
                                    2. Differences in assumptions
                                    3. List of Topics covered in A but missing in B
                                    4. List of Topics covered in B but missing in A
                                    Document A chunks:
                                    {chr(10).join(analyses_A)}
                                    Document B chunks:
                                    {chr(10).join(analyses_B)}
                                    """
                                    comparison = call_gemini_with_retry(
                                        client, comparison_prompt
                                    )

                                st.success("Comparison complete!")

                                # HTML Report Rendering
                                html_stage2 = f"""
                                <div style="border: 1px solid rgba(128,128,128,0.3); border-radius: 14px; padding: 18px; margin: 16px 0; font-family: system-ui, sans-serif;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="font-size: 20px; font-weight: 600;">⚖️ Document Comparison Report</div>
                                        <div style="font-size: 13px; opacity: 0.7;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                                    </div>
                                    <div style="margin-top: 14px; line-height: 1.5; white-space: pre-wrap;">
                                        {comparison.text}
                                    </div>
                                </div>
                                """
                                st.markdown(
                                    html_stage2, unsafe_allow_html=True
                                )