# 🤖 CPU-ChatBot – Lightweight RAG-based Chatbot with Local LLM

This is a lightweight Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Ollama**, **ChromaDB**, and **Streamlit (optional)**.  
It uses **local models only**, making it ideal for development on **CPU-only systems** (no GPU required!).

The chatbot can answer questions based on the content of a PDF document – in this case:  
📄 **`data/attention.pdf`** (the famous "Attention Is All You Need" paper).

---

## 🧠 How It Works

1. **Ingest & Embed**:
   - Parses the PDF.
   - Splits it into chunks.
   - Converts each chunk into a vector embedding using `nomic-embed-text` (via Ollama).
   - Stores the vectors in a local **ChromaDB** instance (`vectorstore/` folder).

2. **Chat Interaction**:
   - When the user asks a question, the system:
     - Embeds the query.
     - Retrieves relevant chunks using similarity search.
     - Passes the top results into a prompt for the main LLM to answer in context.

3. **All models run locally** using **Ollama**, with no internet or cloud API needed after setup.

---

## 📁 Project Structure

```bash
CPU-CHATBOT/
├── data/
│   └── attention.pdf              # Input document
│   └── eval_questions.csv            # CSV with Question and GroundTruth Answers
├── vectorstore/                  # Chroma vector DB
├── venv/                         # Python virtual environment
├── accuracy.csv                  # Accuracy Tested CSV
├── app.py                        # Streamlit app
├── chat_chain.py                 # Loads LangChain + retriever chain
├── evaluation.py                 # Check the accuracy of the ChatBot
├── ingest.py                     # Embeds PDF into vectorstore
├── test.ipynb                    # Sample notebook to test chatbot
├── requirements.txt              # Python dependencies
```

## 🔍 Explanation Summary

### 🔧 What this is:
A **fully local chatbot** that answers questions from a PDF using retrieval-augmented generation (RAG), powered by:
- A local embedding model (from Ollama)
- A local language model (like `gemma`)
- A lightweight vector store (Chroma)

### ⚡️ Why it’s useful:
- Runs on any CPU laptop
- Uses no paid APIs
- Helps explore documents deeply (like research papers)

---

## 💻 Setup Recap

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh    # or download from website (Windows)

# 2. Pull models
ollama pull nomic-embed-text
ollama pull gemma3:1b

# 3. Python setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 4. Ingest PDF (Not needed if vectorstore is already created)
python ingest.py  

# 5. Chat
streamlit run app.py  # to run streamlit on localhost

