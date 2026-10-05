import streamlit as st
from pptx import Presentation

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI College Assistant",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🎓 AI College Assistant")
st.write("Ask questions from your college study material.")

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎓 AI College Assistant")

st.sidebar.write("### About")

st.sidebar.write(
    "This application uses RAG to answer questions "
    "from uploaded college study material."
)

st.sidebar.write("### Technologies")

st.sidebar.write("• Streamlit")
st.sidebar.write("• LangChain")
st.sidebar.write("• ChromaDB")
st.sidebar.write("• Sentence Transformers")
st.sidebar.write("• Ollama")
st.sidebar.write("• Llama 3.2")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# LOAD OLLAMA
# --------------------------------------------------

@st.cache_resource
def load_llm():

    return OllamaLLM(
        model="llama3.2"
    )


embeddings = load_embeddings()
llm = load_llm()


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.sidebar.write("### 📚 Upload Study Material")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or PPTX",
    type=["pdf", "pptx"]
)


# --------------------------------------------------
# PROCESS UPLOADED PPTX
# --------------------------------------------------

if uploaded_file:

    st.sidebar.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # Check PPTX
    if uploaded_file.name.endswith(".pptx"):

        presentation = Presentation(uploaded_file)

        all_text = ""

        # Extract text from slides
        for slide in presentation.slides:

            for shape in slide.shapes:

                if hasattr(shape, "text"):

                    text = shape.text.strip()

                    if text:
                        all_text += text + "\n"


        # --------------------------------------------------
        # SPLIT TEXT INTO CHUNKS
        # --------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(all_text)

        st.sidebar.info(
            f"Created {len(chunks)} text chunks"
        )


        # --------------------------------------------------
        # CREATE CHROMA DATABASE
        # --------------------------------------------------

        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory="./uploaded_chroma_db"
        )

        st.sidebar.success(
            "✅ Document processed successfully!"
        )


        # --------------------------------------------------
        # QUESTION INPUT
        # --------------------------------------------------

        st.subheader("📚 Ask Your Question")

        question = st.text_input(
            "Enter your question:",
            placeholder="Example: What are log records?"
        )


        # --------------------------------------------------
        # RAG QUESTION ANSWERING
        # --------------------------------------------------

        if question:

            with st.spinner("🔍 Searching your study material..."):

                # Retrieve relevant chunks
                results = vectorstore.similarity_search(
                    question,
                    k=3
                )


            # Combine retrieved information
            context = "\n\n".join(
                result.page_content
                for result in results
            )


            # --------------------------------------------------
            # PROMPT
            # --------------------------------------------------

            prompt = f"""
You are an AI College Assistant.

Answer the student's question using ONLY
the information provided in the context.

If the answer is not available in the context,
say:

"Sorry, I could not find this information
in the uploaded study material."

Keep the answer simple and easy to understand.

Context:
{context}

Question:
{question}

Answer:
"""


            # --------------------------------------------------
            # OLLAMA RESPONSE
            # --------------------------------------------------

            with st.spinner("🤖 Generating answer..."):

                answer = llm.invoke(prompt)


            # --------------------------------------------------
            # DISPLAY ANSWER
            # --------------------------------------------------

            st.subheader("🤖 AI Answer")

            st.write(answer)


            # --------------------------------------------------
            # SHOW SOURCE INFORMATION
            # --------------------------------------------------

            with st.expander("📖 View Retrieved Study Material"):

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    st.write(
                        f"### Retrieved Section {i}"
                    )

                    st.write(
                        result.page_content
                    )


# --------------------------------------------------
# NO FILE UPLOADED
# --------------------------------------------------

else:

    st.info(
        "👈 Please upload a PDF or PPTX file "
        "from the sidebar to start asking questions."
    )