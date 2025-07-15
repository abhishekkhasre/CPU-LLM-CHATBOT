# app.py
import streamlit as st
from chat_chain import load_chain

st.set_page_config(page_title="📄 PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF using Ollama + LangChain")

if "chain" not in st.session_state:
    st.session_state.chain = load_chain()

user_query = st.text_input("Ask something about the PDF:")

if user_query:
    response = st.session_state.chain.invoke({"input": user_query})
    print(response)
    st.markdown(f"**Answer:** {response["answer"]}")
