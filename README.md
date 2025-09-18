# BizTalk Document RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system built with **Pydantic AI** and **ChromaDB** for intelligent document querying. This system features a web-based chatbot interface powered by **Streamlit** that can crawl, index, and query documentation from various sources including websites, sitemaps, and text files.

## 🚀 Features

### Core Capabilities
- **🔍 Intelligent Document Retrieval**: Uses ChromaDB vector database for semantic search
- **🤖 AI-Powered Responses**: Leverages Azure OpenAI with Pydantic AI for contextual answers
- **🌐 Web Crawling**: Automated document ingestion from URLs, sitemaps, and text files
- **💬 Interactive Chat Interface**: Strstreaming reamlit-based web UI with real-time esponses
- **📊 Smart Message Management**: Automatic conversation history limiting and token management

### Advanced Features
- **📚 Multi-Source Ingestion**: Support for regular web pages, XML sitemaps
- **⚡ Batch Processing**: Parallel crawling with memory-adaptive dispatching
- **🎯 Hierarchical Chunking**: Smart text segmentation by markdown headers
- **🔄 Real-time Streaming**: Live response generation with delta updates
- **📈 Usage Monitoring**: Token estimation and conversation metrics
- **⚙️ Configurable Settings**: Customizable message limits and retrieval parameters

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   Pydantic AI   │────│  Azure OpenAI   │
│   (Frontend)    │    │    (Agent)      │    │   (LLM Model)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│    ChromaDB     │    │   Crawl4AI      │
│  (Vector Store) │    │  (Web Crawler)  │
└─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Azure OpenAI account and API key
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd BTDocument-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   AZURE_OPENAI_API_KEY=your_azure_openai_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-02-01
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   ```

4. **Install browser dependencies for Crawl4AI**
   ```bash
   playwright install
   ```

## 🚀 Quick Start

### 1. Index Documents
First, crawl and index documents into ChromaDB:

```bash
# Index from a website
python insert_docs.py https://example.com/docs

# Index from a sitemap
python insert_docs.py https://example.com/sitemap.xml

# Index with custom settings
python insert_docs.py https://example.com/docs --collection my-docs --chunk-size 800
```

### 2. Launch the Chat Interface
```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to access the web interface.

### 3. Query via Command Line (Optional)
```bash
python rag_agent.py --question "How do I configure BizTalk Server?"
```

## 📚 Usage Guide

### Document Ingestion

The `insert_docs.py` script supports multiple document sources:

**Basic Usage:**
```bash
python insert_docs.py <URL> [OPTIONS]
```

**Options:**
- `--collection`: ChromaDB collection name (default: "docs")
- `--db-dir`: Database directory (default: "./chroma_db")
- `--embedding-model`: Embedding model (default: "all-MiniLM-L6-v2")
- `--chunk-size`: Maximum chunk size in characters (default: 1000)
- `--max-concurrent`: Parallel crawler sessions (default: 5)
- `--batch-size`: ChromaDB insert batch size (default: 100)

**Examples:**
```bash
# Crawl a documentation site
python insert_docs.py https://docs.microsoft.com/biztalk/

# Process a sitemap with custom settings
python insert_docs.py https://example.com/sitemap.xml --collection biztalk-docs --chunk-size 1200

# Index with specific embedding model
python insert_docs.py https://example.com --embedding-model all-mpnet-base-v2
```

### Chat Interface Features

#### 💬 Conversation Management
- **Automatic Message Limiting**: Keeps conversations within optimal token limits
- **Smart History Preservation**: Maintains question-answer pairs
- **Real-time Metrics**: Track message count and token usage
- **Configurable Limits**: Adjust message retention (5-50 messages)

#### 🔍 Query Capabilities
- **Semantic Search**: Find relevant information using natural language
- **Source Attribution**: Responses include document sources
- **Contextual Answers**: Leverages retrieved documents for accurate responses
- **Streaming Responses**: Real-time answer generation

#### ⚙️ Settings & Controls
- **Message Limit Slider**: Customize conversation history length
- **Clear History**: Reset conversation for fresh start
- **Usage Monitoring**: Visual progress bars and warnings
- **Performance Metrics**: Token estimation and memory usage

## 🛠️ Configuration

### Environment Variables
```env
# Required Azure OpenAI settings
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini

# Optional ChromaDB settings
CHROMA_DB_DIR=./chroma_db
DEFAULT_COLLECTION=docs
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Application Settings

**Message Management (streamlit_app.py):**
```python
MAX_MESSAGES = 20  # Maximum messages in conversation
MAX_TOKENS_ESTIMATE = 4000  # Token limit estimation
```

**Crawling Settings (insert_docs.py):**
```python
DEFAULT_CHUNK_SIZE = 1000  # Characters per chunk
DEFAULT_BATCH_SIZE = 100   # ChromaDB batch size
DEFAULT_MAX_CONCURRENT = 5  # Parallel crawlers
```

## 📁 Project Structure

```
BTDocument-RAG/
├── streamlit_app.py      # Main web interface
├── rag_agent.py          # Pydantic AI agent implementation
├── insert_docs.py        # Document crawling and indexing
├── utils.py              # ChromaDB utilities
├── requirements.txt      # Python dependencies
├── MESSAGE_MANAGEMENT.md # Message management documentation
├── .env                  # Environment variables (create this)
├── chroma_db/           # ChromaDB database directory
│   ├── chroma.sqlite3   # Database file
│   └── collections/     # Collection data
└── __pycache__/         # Python cache files
```

## 🔧 Components

### Core Modules

**`streamlit_app.py`** - Web Interface
- Streamlit-based chat interface
- Real-time message streaming
- Conversation management
- Usage monitoring and controls

**`rag_agent.py`** - AI Agent
- Pydantic AI agent implementation
- Azure OpenAI integration
- RAG workflow orchestration
- Document retrieval tool

**`insert_docs.py`** - Document Processor
- Multi-source web crawling
- Intelligent content chunking
- ChromaDB indexing
- Batch processing

**`utils.py`** - Utilities
- ChromaDB client management
- Collection operations
- Query formatting
- Batch processing helpers

## 🚀 Advanced Usage

### Custom Embedding Models
```bash
# Use different embedding models
python insert_docs.py https://example.com --embedding-model sentence-transformers/all-mpnet-base-v2
```

### Batch Document Processing
```python
# Process multiple URLs programmatically
urls = [
    "https://docs.microsoft.com/biztalk/",
    "https://example.com/docs/",
    "https://another-site.com/help/"
]

for url in urls:
    # Process each URL with insert_docs.py
    subprocess.run(["python", "insert_docs.py", url])
```

### Custom Retrieval Settings
```python
# Customize RAG agent behavior
response = await run_rag_agent(
    question="How do I configure BizTalk?",
    collection_name="biztalk-docs",
    n_results=10,  # More results for complex queries
    embedding_model="all-mpnet-base-v2"
)
```

## 🔍 Troubleshooting

### Common Issues

**1. Azure OpenAI Connection Errors**
- Verify API key and endpoint in `.env` file
- Check Azure OpenAI service availability
- Ensure deployment name matches your Azure configuration

**2. ChromaDB Errors**
- Check database directory permissions
- Verify sufficient disk space
- Restart if database locks occur

**3. Crawling Issues**
- Install playwright browsers: `playwright install`
- Check website accessibility and robots.txt
- Reduce concurrent sessions if experiencing timeouts

**4. Memory Issues**
- Reduce chunk size for large documents
- Decrease batch size for ChromaDB operations
- Lower max concurrent crawlers

### Performance Optimization

**For Large Document Sets:**
```bash
# Optimize for large datasets
python insert_docs.py https://large-site.com \
  --chunk-size 800 \
  --batch-size 50 \
  --max-concurrent 3
```

**For Better Search Quality:**
```bash
# Use more sophisticated embedding model
python insert_docs.py https://example.com \
  --embedding-model sentence-transformers/all-mpnet-base-v2
```

## 📈 Monitoring & Metrics

The application provides comprehensive monitoring:

- **Message Count**: Track conversation length
- **Token Estimation**: Monitor context usage
- **Memory Usage**: Visual progress indicators
- **Retrieval Quality**: Relevance scores in responses
- **Performance Metrics**: Response times and throughput

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pydantic AI** - For the excellent AI agent framework
- **ChromaDB** - For the powerful vector database
- **Crawl4AI** - For efficient web crawling capabilities
- **Streamlit** - For the intuitive web interface framework
- **Azure OpenAI** - For providing the language model capabilities

## 📞 Support

For questions, issues, or contributions:

1. Check existing [Issues](../../issues)
2. Create a new issue with detailed description
3. Include relevant logs and configuration details
4. Follow the issue template for faster resolution

---

**Made with ❤️ for intelligent document querying**
