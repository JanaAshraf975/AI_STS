import streamlit as st
from Modules.pdf_utils import extract_text_from_pdf, chunk_text
from Modules.vectoredb import build_vectoredb, retrive_context
from Modules.llm import rewrite_query, ask_gemin


st.title("PDF RAG ChatBot")

# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []      # chat history
if "file_key" not in st.session_state:
    st.session_state.file_key = None    # which PDF is currently loaded
if "collection" not in st.session_state:
    st.session_state.collection = None
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0

# ---------- sidebar ----------
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------- upload ----------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    # Process the PDF only when a NEW file is uploaded
    # (Streamlit reruns this script on every interaction)
    if st.session_state.file_key != file_key:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(pdf_file=uploaded_file)
            chunks = chunk_text(text)

            if not chunks:
                st.error("No text found in this PDF (it may be a scanned image).")
                st.stop()

            st.session_state.collection = build_vectoredb(chunks)
            st.session_state.n_chunks = len(chunks)
            st.session_state.file_key = file_key
            st.session_state.messages = []   # new PDF -> new conversation

    st.success(f"{st.session_state.n_chunks} chunks loaded")

    # ---------- show previous messages ----------
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("context"):
                with st.expander("Retrieved context"):
                    st.write(m["context"])

    # ---------- new question ----------
    question = st.chat_input("Ask your PDF")

    if question:
        # history sent to the LLM: only role + content
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                standalone = rewrite_query(question, history)               # 1. clear question
                context = retrive_context(standalone,
                                          st.session_state.collection)      # 2. search + rerank
                answer = ask_gemin(question, context, history)              # 3. answer
            st.markdown(answer)
            with st.expander("Retrieved context"):
                st.write(context)

        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "context": context}
        )