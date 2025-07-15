from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def load_chain():
    # Step 1: Embedding model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Step 2: Prompt for document chain
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question in detailed based only on the provided context:

<context>
{context}
</context>

Question: {input}
Answer:"""
    )

    # Step 3: LLM and document chain
    llm = Ollama(model="gemma3:1b")
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Step 4: Retriever from Chroma vectorstore
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # optional: increase recall

    # Step 5: Retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

