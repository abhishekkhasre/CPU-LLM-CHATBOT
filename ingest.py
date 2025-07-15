# ingest.py
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdf(pdf_path="data/attention.pdf"):
    print("Loading PDF...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    print("Splitting...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    print("Embedding with Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="vectorstore")
    vectorstore.persist()
    print("Database Created Successfully...")

if __name__ == "__main__":
    ingest_pdf()
