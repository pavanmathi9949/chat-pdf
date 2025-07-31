import streamlit as st
import requests

# Set your API endpoints 
trigger_endipoint = "http://localhost:5000/index"
query_endpoint = "http://localhost:5000/query"

# Set page config to add an app icon and title
st.set_page_config(page_title="Student Document Query", page_icon="📚")

# Sidebar title
st.sidebar.title("📄 Upload Your Course Document")

# PDF Upload Section
uploaded_pdf = st.sidebar.file_uploader("Upload Your PDF Document 📝", type="pdf")

# Button to Trigger API
if st.sidebar.button("📤 Upload & Query"):
    if uploaded_pdf is not None:
        with st.spinner("Uploading your document to our API... ⏳"):
            files = {"file": (uploaded_pdf.name, uploaded_pdf, "application/pdf")}
            try:
                response = requests.post(trigger_endipoint, files=files)
                response.raise_for_status()
                st.success("🎉 Document uploaded successfully! Now, feel free to ask your questions below!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Upload failed: {e}")
    else:
        st.warning("⚠️ Please upload a PDF document first.")

# RAG QA function for querying
def rag_qa(question):
    headers = {'Content-Type': 'application/json'}
    data = {"question": question}
    try:
        response = requests.post(query_endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get('answer', 'No answer found for your question.')
    except requests.exceptions.RequestException as e:
        return f"❌ Error querying API: {e}"

# Main App Title
st.title("📝 Your Study Buddy")
st.caption("Upload & Query Your Notes")

# Chat interface section
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input section for querying
if prompt := st.chat_input("❓ Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        output_response = rag_qa(prompt)
        message_placeholder.markdown(output_response)
    st.session_state.messages.append({"role": "assistant", "content": output_response})