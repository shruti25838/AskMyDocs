import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import SerpAPIWrapper
search = SerpAPIWrapper(serpapi_api_key=st.secrets["SERPAPI_API_KEY"])
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
from dotenv import load_dotenv
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from loader import load_pdf
from rag_pipeline import rag_pipeline
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType


load_dotenv()

st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

# ----- CSS Styling -----
st.markdown("""
    <style>
    body { background-color: #121212; }
    .stFileUploader {
        background-color: #1a1a1a !important;
        border: 1px solid #666 !important;
        border-radius: 0.75rem;
        padding: 1.2rem;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
    }
    div.stButton > button {
        background-color: #c0392b;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 0.95rem;
    }
    .chat-question {
        font-weight: 600;
        margin-top: 1rem;
        color: #f5f5f5;
    }
    .chat-answer {
        background-color: #1e1e1e;
        padding: 0.8rem 1.2rem;
        border-radius: 0.5rem;
        color: #e0e0e0;
        border-left: 4px solid #c0392b;
        line-height: 1.6;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }
    hr.divider {
        border: none;
        border-top: 1px solid #333;
        margin: 2rem 0;
    }
    .chunk-box {
        background-color: #2a2a2a;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #c0392b;
        border-radius: 0.5rem;
    }
    .chunk-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #c0392b;
    }
    </style>
""", unsafe_allow_html=True)

# ----- Title -----
st.title("📄 AskMyDocs - Chat with Your PDFs")

# ----- Sidebar Settings -----
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["gpt-3.5-turbo-16k", "gpt-4"])
max_tokens = st.sidebar.slider("Max tokens", 100, 4000, 800)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
hybrid_mode = st.sidebar.checkbox("Smart Search Mode", value=True)
preview_chunks = st.sidebar.checkbox("Show Document Segments", value=False)
use_web_fallback = st.sidebar.checkbox("Enable Web Help", value=True)

# ----- Session State Initialization -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "similar_questions" not in st.session_state:
    st.session_state.similar_questions = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ----- Helper Functions -----
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

def get_similar_questions(user_question):
    prompt = [
        SystemMessage(content="You are a helpful assistant in a document Q&A app. Based on the user's question, suggest 2 to 3 short, casual follow-up questions that the user might ask next. Keep them simple, informal, and under 12 words each."),
        HumanMessage(content=f"Original question: {user_question}")
    ]
    try:
        response = chat(prompt).content.strip()
        return [q.strip("\u2022- \n") for q in response.split("\n") if q.strip()]
    except Exception as e:
        return [f"⚠️ Could not generate suggestions: {e}"]

search = SerpAPIWrapper()
search_tool = Tool(name="Web Search", func=search.run, description="Search for information online")
agent = initialize_agent([search_tool], ChatOpenAI(temperature=0.3), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

def web_fallback_answer(question):
    try:
        return f"(From web)\n\n{agent.run(question)}"
    except Exception as e:
        return f"(Web search failed): {e}"

def generate_chat_pdf(chat_history):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 11)
    text.textLine("AskMyDocs Chat History")
    text.textLine("========================")
    for i, (q, a) in enumerate(chat_history):
        text.textLine(f"\nQ{i+1}: {q}")
        for line in a.split("\n"):
            text.textLine(line[:110])
        text.textLine("-" * 80)
    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ----- PDF Upload -----
st.markdown("### Upload a PDF to get started")
pdfs = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True, key="pdf_upload")

if pdfs:
    full_text = ""
    for pdf in pdfs:
        path = f"temp_{pdf.name}"
        with open(path, "wb") as f:
            f.write(pdf.read())
        full_text += load_pdf(path) + "\n"

    if full_text.strip():
        qa, retriever, chunks = rag_pipeline(
            full_text, model_choice, temperature, max_tokens,
            hybrid=hybrid_mode, return_chunks=True
        )
        st.session_state.qa_chain, st.session_state.retriever = qa, retriever

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Ask a Question")
            user_input = st.text_area("Type your question:", value=st.session_state.prefill, key="input_box")
            if st.button("Submit") and user_input.strip():
                questions = [q.strip() for q in user_input.split("\n") if q.strip()]
                for question in questions:
                    try:
                        result = st.session_state.qa_chain.invoke({"query": question})
                        llm_answer = result.get("result", "")
                        if use_web_fallback and (
                            not llm_answer.strip() or any(phrase in llm_answer.lower() for phrase in [
                                "i don't know", "i am not sure", "not available", "cannot find",
                                "unsure", "no information", "not mentioned"])):
                            llm_answer = web_fallback_answer(question)

                        similar_qs = get_similar_questions(question)
                        st.session_state.similar_questions = similar_qs
                        answer = f"{llm_answer}"

                    except Exception as e:
                        answer = f"⚠️ Error: {e}"

                    st.session_state.chat_history.append((question, answer))
                st.session_state.prefill = ""
                st.rerun()

        with col2:
            if st.session_state.similar_questions and st.session_state.chat_history:
                st.markdown("### Suggested Follow-up Questions")
                for i, sq in enumerate(st.session_state.similar_questions):
                    if st.button(sq, key=f"suggested_{i}"):
                        st.session_state.prefill = sq
                        st.rerun()

        if st.session_state.chat_history:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("### Chat History")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"<div class='chat-question'>Q: {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-answer'>{a}</div>", unsafe_allow_html=True)

            st.download_button(
                "Download Chat History (PDF)",
                data=generate_chat_pdf(st.session_state.chat_history),
                file_name="chat_history.pdf",
                mime="application/pdf"
            )

        if preview_chunks:
            with st.expander("View Key Sections from Uploaded Document"):
                for i, c in enumerate(chunks[:15]):
                    st.markdown(f"""
                        <div class='chunk-box'>
                            <div class='chunk-title'>Key Insight {i+1}</div>
                            <div>{c.strip()}</div>
                        </div>
                    """, unsafe_allow_html=True)

# Optional prefill from URL param
query_params = st.query_params
if "prefill" in query_params:
    st.session_state.prefill = query_params["prefill"]
    st.query_params.clear()
