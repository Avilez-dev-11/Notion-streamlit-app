import streamlit as st
from openai import OpenAI
import PyPDF2
from io import BytesIO

# Initialize session state for chat history if not already done
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []



# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["api_key"]

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)
# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to generate a response from OpenAI


def generate_response(messages):
    try:
        response = client.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            stream=False  # Set to True if you want streaming responses
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"
    

# Set page configuration
st.set_page_config(page_title="📄 Document Q&A & Chat", layout="wide")

# Show title and description.
st.title("📄  Document Question Answering & General Chat")

# Tabs for separating functionalities
tab1, tab2 = st.tabs(["📄 Document Q&A", "💬 General Chat"])

with tab1:
    st.header("Document Question Answering")
    st.write(
        "Upload a document (TXT, MD, PDF) below and ask a question about it – GPT-4 will answer!"
    )

    # File uploader supporting TXT, MD, PDF
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, .pdf)",
        type=["txt", "md", "pdf"]
    )

    # Text area for question
    question = st.text_area(
        "Ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=uploaded_file is None
    )

    if uploaded_file and question:
        with st.spinner("Processing your request..."):
            # Extract text based on file type
            if uploaded_file.type == "application/pdf":
                document = extract_text_from_pdf(BytesIO(uploaded_file.read()))
            else:
                document = uploaded_file.read().decode()

            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Here is the document:\n\n{document}"},
                {"role": "user", "content": question}
            ]

            # Generate response
            answer = generate_response(messages)

            st.success("**Answer:**")
            st.write(answer)

with tab2:
    st.header("General Chat")
    st.write("Start chatting with GPT-4 without uploading any documents.")

    # Input for user message
    user_input = st.text_input("You:", placeholder="Type your message here...")

    if user_input:
        # Append user message to chat history
        st.session_state['chat_history'].append(
            {"role": "user", "content": user_input})

        with st.spinner("GPT-4 is typing..."):
            # Generate response based on chat history
            response = generate_response(st.session_state['chat_history'])

            # Append assistant response to chat history
            st.session_state['chat_history'].append(
                {"role": "assistant", "content": response})

            # Display the response
            st.write(f"**GPT-4:** {response}")

    # Display chat history
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            if chat['role'] == 'user':
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**GPT-4:** {chat['content']}")

# Stream the response to the app using `st.write_stream`.
st.write_stream(response)
