# DocuChat AI

**Conversational AI for Document Intelligence**

Transform your PDFs into interactive chat experiences with instant, accurate answers powered by advanced AI.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![License](https://img.shields.io/github/license/Harine77/DocuChat-AI?style=flat-square)](LICENSE)

## Overview

DocuChat AI is a Retrieval-Augmented Generation (RAG) application that enables natural language conversations with multiple PDF documents. Built with modern AI technologies, it provides fast, contextual responses by combining semantic search with large language models.

## Key Features

- **Multi-Document Processing** - Upload and query multiple PDFs simultaneously
- **Conversational Interface** - Memory-enabled chat for follow-up questions
- **Semantic Search** - FAISS-powered vector similarity matching
- **Fast Inference** - Sub-2-second response times via Groq API
- **Context Awareness** - Maintains conversation history for coherent interactions

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Groq LLaMA3-8B | Fast inference and response generation |
| **Framework** | LangChain | LLM orchestration and workflow management |
| **Vector DB** | FAISS | Semantic search and similarity matching |
| **Embeddings** | Sentence Transformers | Text vectorization |
| **Interface** | Streamlit | Web application framework |
| **Processing** | PyPDF2 | PDF text extraction |

## Architecture

The system implements a standard RAG pipeline:

1. **Document Ingestion** - PDFs are processed and text extracted using PyPDF2
2. **Text Chunking** - Content split into overlapping chunks (1000 chars, 200 overlap)
3. **Vectorization** - Chunks converted to embeddings using Sentence Transformers
4. **Storage** - Vectors stored in FAISS database for fast retrieval
5. **Query Processing** - User questions embedded and matched against stored vectors
6. **Response Generation** - Groq LLM generates answers using retrieved context
7. **Memory Management** - ConversationBufferMemory maintains chat history

## Quick Start

### Installation

```bash
git clone https://github.com/Harine77/DocuChat-AI.git
cd DocuChat-AI
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env file
```

### Run Application

```bash
streamlit run app.py
```

## Usage

1. **Upload Documents** - Add multiple PDF files via the interface
2. **Ask Questions** - Type natural language queries about your documents
3. **Get Answers** - Receive contextual responses with source references
4. **Continue Conversation** - Ask follow-up questions with maintained context

## Performance

| Metric | Local (Ollama) | Cloud (Groq) |
|--------|----------------|--------------|
| Model | Gemma 2B | LLaMA3-8B |
| Response Time | 5+ minutes | <2 seconds |
| Memory Usage | 1.7GB | API-based |
| Scalability | Limited | High |

## Project Structure

```
DocuChat-AI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── .env.example          # Environment template
├── utils/
│   ├── pdf_processor.py  # PDF handling
│   ├── embeddings.py     # Vector operations
│   └── chat_engine.py    # LLM integration
└── README.md             # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Project Link:** [https://github.com/Harine77/DocuChat-AI](https://github.com/Harine77/DocuChat-AI)

---

*Built with modern AI technologies for intelligent document interaction*
