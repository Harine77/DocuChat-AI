import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models.llms import LLM
from htmlTemplates import css, bot_template, user_template
from typing import Optional, List, ClassVar
import requests
import os

# ✅ Groq LLM integration for LangChain-compatible use
class GroqLLM(LLM):
    model: ClassVar[str] = "llama3-8b-8192"
    temperature: ClassVar[float] = 0.7
    max_tokens: ClassVar[int] = 1000
    groq_api_key: str

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_api_key}"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"

# ✅ Load Groq model
def load_groq_llm():
    api_key = os.getenv("GROQ_API_KEY")
    return GroqLLM(groq_api_key=api_key)

# ✅ Extract PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

# ✅ Split text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# ✅ Vectorstore creation
def get_vectorstore(text_chunks, embedding):
    return FAISS.from_texts(texts=text_chunks, embedding=embedding)

# ✅ Conversation chain setup
def get_conversation_chain(vectorstore):
    llm = load_groq_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# ✅ User input handler
def handle_userinput(user_question):
    if st.session_state.conversation:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            template = user_template if i % 2 == 0 else bot_template
            st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ✅ Enhanced UI Components
def render_hero_section():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-bottom: 2rem; color: white;">
        <h1 style="font-size: 3rem; margin: 0; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🚀 DocuChat AI
        </h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.9;">
            Transform Your PDFs into Interactive Conversations
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                🤖 AI-Powered
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                ⚡ Lightning Fast
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                🔒 Secure
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_about_section():
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border-left: 5px solid #667eea;">
        <h2 style="color: #333; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">💡</span> What is DocuChat AI?
        </h2>
        <p style="color: #666; line-height: 1.6; margin-bottom: 1rem;">
            DocuChat AI revolutionizes how you interact with your PDF documents. Instead of manually searching through pages, 
            simply upload your PDFs and ask questions in natural language. Our advanced AI, powered by Groq's LLaMA3 8B model, 
            understands your documents and provides accurate, contextual answers instantly.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <strong>Upload PDFs</strong>
                <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">Support for multiple PDF documents</p>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <strong>Ask Questions</strong>
                <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">Natural language queries</p>
            </div>
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                <strong>Get Answers</strong>
                <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">Precise, context-aware responses</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_stats():
    col1, col2, col3, col4 = st.columns(4)

    stat_style = """
    <div style="
        background: linear-gradient(135deg, {color1}, {color2});
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        color: white;
    ">
        <div style="font-size: 1.4rem; margin-bottom: 0.3rem;">{icon}</div>
        <div style="font-size: 0.85rem;">{label}</div>
    </div>
    """

    with col1:
        st.markdown(
            stat_style.format(
                color1="#ff6b6b", color2="#ee5a24", icon="🚀", label="Ultra Fast"
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            stat_style.format(
                color1="#4ecdc4", color2="#44a08d", icon="🎯", label="Accurate"
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            stat_style.format(
                color1="#a8e6cf", color2="#7fcdcd", icon="🔒", label="Secure"
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            stat_style.format(
                color1="#ffd93d", color2="#ff6b6b", icon="⚡", label="AI-Powered"
            ),
            unsafe_allow_html=True,
        )


def render_chat_interface():
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;">
        <h2 style="color: #333; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">💬</span> Chat with Your Documents
        </h2>
    """, unsafe_allow_html=True)
    
    # Chat status indicator
    if st.session_state.conversation:
        st.markdown("""
        <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #c3e6cb;">
            <strong>✅ Ready to Chat!</strong> Your documents are processed and ready for questions.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #f8d7da; color: #721c24; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #f5c6cb;">
            <strong>⏳ Upload & Process PDFs</strong> Please upload your documents first to start chatting.
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced text input
    user_question = st.text_input(
        "Ask something about your documents:",
        placeholder="e.g., 'What are the main topics discussed in the document?'",
        help="Type your question here and press Enter"
    )
    
    if user_question:
        handle_userinput(user_question)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 2rem; color: white;">
            <h3 style="margin: 0; font-size: 1.5rem;">🛠️ Control Panel</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Manage your documents</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📁 Document Upload")
        st.markdown("Upload your PDF documents to get started. Multiple files are supported.")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Select one or more PDF files to upload"
        )
        
        if pdf_docs:
            st.markdown(f"**📊 {len(pdf_docs)} file(s) selected**")
            for i, pdf in enumerate(pdf_docs, 1):
                st.markdown(f"• {pdf.name}")
        
        process_button = st.button(
            "🚀 Process Documents",
            use_container_width=True,
            type="primary"
        )
        
        if process_button:
            if pdf_docs:
                with st.spinner("🔄 Processing your documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        chunks = get_text_chunks(raw_text)
                        embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                        vectorstore = get_vectorstore(chunks, embedding)
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        st.success("✅ Documents processed successfully!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least one PDF file first.")
        
        st.markdown("---")
        
        # API Test Section
        st.markdown("### 🔧 System Status")
        
        if st.button("🧪 Test API Connection", use_container_width=True):
            try:
                with st.spinner("Testing connection..."):
                    llm = load_groq_llm()
                    result = llm._call("Hello! Just testing the connection.")
                    if "Error" not in result:
                        st.success("✅ API connection successful!")
                    else:
                        st.error(f"❌ API Error: {result}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = None
            st.rerun()
        
        st.markdown("---")
        
        # Help section
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload multiple PDFs for comprehensive analysis
        - Ask specific questions for better results
        - Use natural language - no special commands needed
        - Try questions like:
          - "What is the main topic?"
          - "Summarize the key points"
          - "Find information about..."
        """)

# ✅ Main Streamlit app
def main():
    load_dotenv()
    
    # Page configuration
    st.set_page_config(
        page_title="DocuChat AI - Chat with Your PDFs",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stButton > button {
            border-radius: 20px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
            border: 2px solid #e1e5e9;
            padding: 0.75rem;
        }
        .stTextInput > div > div > input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        .stFileUploader > div {
            border-radius: 10px;
            border: 2px dashed #667eea;
            padding: 1rem;
            text-align: center;
            background: rgba(102, 126, 234, 0.05);
        }
        .stSpinner > div {
            border-color: #667eea;
        }
        .stSidebar > div {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Include original CSS
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # Render UI components
    render_hero_section()
    render_about_section()
    render_stats()
    render_chat_interface()
    render_sidebar()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <p style="margin: 0;">Built with ❤️ using Streamlit & Groq AI • DocuChat AI © 2024</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
            Powered by LLaMA3 8B • Vector Search • Advanced NLP
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()