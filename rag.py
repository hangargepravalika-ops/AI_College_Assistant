from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. Connect to Ollama
llm = OllamaLLM(model="llama3.2")

# 4. Ask a question
question = "What are log records?"

# 5. Retrieve relevant information
results = vectorstore.similarity_search(question, k=3)

# 6. Combine retrieved information
context = "\n\n".join(
    result.page_content for result in results
)

# 7. Create prompt
prompt = f"""
You are an AI College Assistant.

Answer the question using ONLY the information provided in the context.

Context:
{context}

Question:
{question}

Give a simple and clear answer.
"""

# 8. Generate answer using Ollama
answer = llm.invoke(prompt)

print("\nQuestion:")
print(question)

print("\nAI College Assistant Answer:")
print(answer)