# Sierra Leone Laws System

A Retrieval-Augmented Generation (RAG) system for querying Sierra Leone legal documents using LangChain, FAISS, and Hugging Face models. Now with REST API support!

## Overview

This project implements a question-answering system that allows users to query legal documents from Sierra Leone. It uses:
- **FAISS** for efficient vector similarity search
- **HuggingFace Embeddings** for document encoding
- **Mixtral-8x7B** LLM for generating answers
- **LangChain** for orchestrating the RAG pipeline
- **FastAPI** for REST API access

## Features

- 📄 Load multiple PDF documents from a directory
- 🌐 Load documents from web URLs
- 🔍 Semantic search using FAISS vector store
- 🤖 AI-powered question answering with source attribution
- 💾 Persistent vector store with automatic caching
- 🚀 REST API with streaming support
- ⚡ Intelligent cache management (only rebuilds when files change)
- 📊 Source relevance filtering
- 🔄 Hot-reload API with manual rebuild endpoint

## Prerequisites

- Python 3.8+
- Hugging Face API token (for Mixtral model access)

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. **Install dependencies**
```bash
pip install langchain langchain-huggingface langchain-community
pip install sentence-transformers faiss-cpu pypdf
pip install python-dotenv huggingface-hub
pip install fastapi uvicorn pydantic
```

3. **Set up environment variables**

Create a `.env` file in the root directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

## Project Structure

```
.
├── data/
│   ├── pdfs/              # Place your PDF files here
│   └── urls.txt           # List of URLs to scrape (one per line)
├── SL_Laws_faiss/         # Generated FAISS vector store (created automatically)
├── dataLoading.py         # Data loading and vectorstore creation with caching
├── rag.py                 # RAG chain configuration
├── main.py                # FastAPI REST API server
├── vectorstore_metadata.json  # Cache metadata (auto-generated)
├── .env                   # Environment variables (create this)
└── README.md              # This file
```

## Usage

### Step 1: Prepare Your Data

1. **Add PDF files**: Place your Sierra Leone legal PDFs in `data/pdfs/`

2. **Add URLs**: Create `data/urls.txt` with one URL per line:
```text
https://example.com/legal-document-1
https://example.com/legal-document-2
```

### Step 2: Start the API Server

The vectorstore will be automatically created on first run:

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will:
- ✅ Automatically build the vectorstore on first run
- ⚡ Load from cache on subsequent runs (instant startup!)
- 🔄 Detect when files change and rebuild automatically
- 📊 Track metadata about your document collection

**Access the API:**
- 🏠 Base URL: `http://localhost:8000`
- 📖 Interactive docs: `http://localhost:8000/docs`
- 🔍 Alternative docs: `http://localhost:8000/redoc`

### Step 3: Query via API

#### Using the Interactive Docs (Easiest)
1. Open `http://localhost:8000/docs` in your browser
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Enter your question and click "Execute"

#### Using cURL
```bash
# Simple question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are my rights if I am arrested?"}'

# Streaming response
curl -N -X POST "http://localhost:8000/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I register a business?"}'
```

### Step 4: Managing the Vectorstore

#### Check Status
```bash
curl http://localhost:8000/status
```

#### Force Rebuild (when you add new files)
```bash
curl -X POST http://localhost:8000/rebuild
```

**When to rebuild:**
- ✅ After adding new PDFs to `data/pdfs/`
- ✅ After updating `data/urls.txt`
- ✅ If you want to refresh all data

**Note:** The system automatically detects file changes, but you can force a rebuild if needed.

---

## 🚀 Using the API in Your Project

The Sierra Leone Laws API can be easily integrated into any application. Here are examples for different platforms:

### Python Integration

#### Basic Example
```python
import requests

API_URL = "http://localhost:8000"

def ask_legal_question(question):
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Answer: {data['answer']}\n")
        
        print("Sources:")
        for source in data['sources']:
            print(f"  - {source['source']} (Page {source['page']})")
    else:
        print(f"Error: {response.status_code}")

# Use it
ask_legal_question("What are my rights if I'm arrested?")
```

#### Streaming Example
```python
import requests
import json

def ask_with_streaming(question):
    response = requests.post(
        f"{API_URL}/ask-stream",
        json={"question": question},
        stream=True
    )
    
    print("Answer: ", end="", flush=True)
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                
                if data['type'] == 'answer_chunk':
                    print(data['content'], end="", flush=True)
                elif data['type'] == 'complete':
                    print("\n\nSources:")
                    for source in data['sources']:
                        print(f"  - {source['source']}")

ask_with_streaming("How do I register a business?")
```

#### Full-Featured Client Class
```python
import requests
import json
from typing import Optional, List, Dict

class SierraLeoneLawsClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def ask(self, question: str) -> Dict:
        """Ask a question and get full response"""
        response = requests.post(
            f"{self.base_url}/ask",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    
    def ask_stream(self, question: str, callback=None):
        """Ask with streaming response"""
        response = requests.post(
            f"{self.base_url}/ask-stream",
            json={"question": question},
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if callback:
                        callback(data)
                    yield data
    
    def get_status(self) -> Dict:
        """Get vectorstore status"""
        response = requests.get(f"{self.base_url}/status")
        return response.json()
    
    def rebuild(self) -> Dict:
        """Force rebuild of vectorstore"""
        response = requests.post(f"{self.base_url}/rebuild")
        return response.json()
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

# Usage
client = SierraLeoneLawsClient()

# Check health
if client.health_check():
    print("✅ API is running")

# Ask a question
result = client.ask("What are the requirements for voting?")
print(f"Answer: {result['answer']}")

# Stream a response
def on_chunk(data):
    if data['type'] == 'answer_chunk':
        print(data['content'], end='', flush=True)

for _ in client.ask_stream("How does the court system work?", on_chunk):
    pass
```

### JavaScript/Node.js Integration

#### Basic Fetch Example
```javascript
const API_URL = 'http://localhost:8000';

async function askLegalQuestion(question) {
    try {
        const response = await fetch(`${API_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });
        
        const data = await response.json();
        
        console.log('Answer:', data.answer);
        console.log('\nSources:');
        data.sources.forEach(source => {
            console.log(`  - ${source.source} (Page ${source.page})`);
        });
    } catch (error) {
        console.error('Error:', error);
    }
}

// Use it
askLegalQuestion("What are my rights if I'm arrested?");
```

#### Streaming with Fetch API
```javascript
async function askWithStreaming(question) {
    const response = await fetch(`${API_URL}/ask-stream`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    console.log('Answer: ');

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'answer_chunk') {
                    process.stdout.write(data.content);
                } else if (data.type === 'complete') {
                    console.log('\n\nSources:');
                    data.sources.forEach(source => {
                        console.log(`  - ${source.source}`);
                    });
                }
            }
        }
    }
}

askWithStreaming("How do I register a business?");
```

#### Client Class for Node.js
```javascript
const axios = require('axios');

class SierraLeoneLawsClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.client = axios.create({ baseURL });
    }

    async ask(question) {
        const response = await this.client.post('/ask', { question });
        return response.data;
    }

    async askStream(question, onChunk) {
        const response = await this.client.post('/ask-stream', 
            { question },
            { responseType: 'stream' }
        );

        response.data.on('data', chunk => {
            const lines = chunk.toString().split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (onChunk) onChunk(data);
                }
            }
        });
    }

    async getStatus() {
        const response = await this.client.get('/status');
        return response.data;
    }

    async rebuild() {
        const response = await this.client.post('/rebuild');
        return response.data;
    }

    async healthCheck() {
        try {
            await this.client.get('/health');
            return true;
        } catch {
            return false;
        }
    }
}

// Usage
const client = new SierraLeoneLawsClient();

(async () => {
    const result = await client.ask("What are voting requirements?");
    console.log(result.answer);
})();
```

### React Integration

#### Basic Hook
```jsx
import { useState } from 'react';

const API_URL = 'http://localhost:8000';

function useLegalQuestion() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const ask = async (question) => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`${API_URL}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            
            const data = await response.json();
            return data;
        } catch (err) {
            setError(err.message);
            throw err;
        } finally {
            setLoading(false);
        }
    };
    
    return { ask, loading, error };
}

// Use in component
function LegalAssistant() {
    const { ask, loading, error } = useLegalQuestion();
    const [result, setResult] = useState(null);
    const [question, setQuestion] = useState('');
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        const data = await ask(question);
        setResult(data);
    };
    
    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input 
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a legal question..."
                />
                <button disabled={loading}>
                    {loading ? 'Searching...' : 'Ask'}
                </button>
            </form>
            
            {error && <div className="error">{error}</div>}
            
            {result && (
                <div>
                    <h3>Answer:</h3>
                    <p>{result.answer}</p>
                    <h4>Sources:</h4>
                    <ul>
                        {result.sources.map((source, i) => (
                            <li key={i}>
                                {source.source} - Page {source.page}
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
```

#### Streaming Hook
```jsx
import { useState, useCallback } from 'react';

function useStreamingQuestion() {
    const [streaming, setStreaming] = useState(false);
    const [answer, setAnswer] = useState('');
    const [sources, setSources] = useState([]);
    
    const askStreaming = useCallback(async (question) => {
        setStreaming(true);
        setAnswer('');
        setSources([]);
        
        const response = await fetch(`${API_URL}/ask-stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    
                    if (data.type === 'answer_chunk') {
                        setAnswer(prev => prev + data.content);
                    } else if (data.type === 'complete') {
                        setSources(data.sources);
                        setStreaming(false);
                    }
                }
            }
        }
    }, []);
    
    return { askStreaming, streaming, answer, sources };
}
```

### cURL Examples

```bash
# Basic question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are my rights if I am arrested?"}'

# Streaming response
curl -N -X POST http://localhost:8000/ask-stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I register a business?"}'

# Get examples
curl http://localhost:8000/examples

# Health check
curl http://localhost:8000/health

# Check status
curl http://localhost:8000/status

# Rebuild vectorstore
curl -X POST http://localhost:8000/rebuild
```

### API Response Formats

#### Standard Response (`/ask`)
```json
{
  "question": "What are my rights if I'm arrested?",
  "answer": "According to Sierra Leone law, if you are arrested...",
  "sources": [
    {
      "source": "constitution",
      "page": 15,
      "content_preview": "Every person who is arrested shall be..."
    }
  ],
  "status": "success"
}
```

#### Streaming Response (`/ask-stream`)
Server-Sent Events format:
```
data: {"type": "answer_chunk", "content": "According ", "done": false}
data: {"type": "answer_chunk", "content": "to ", "done": false}
...
data: {"type": "complete", "sources": [...], "done": true}
```

#### Error Response
```json
{
  "error": "Error message here",
  "status": "error"
}
```

### Integration Tips

1. **Error Handling**: Always wrap API calls in try-catch blocks
2. **Timeout**: Set appropriate timeouts (30-60s for complex questions)
3. **Rate Limiting**: Consider implementing rate limiting on your end
4. **Caching**: Cache frequently asked questions to reduce load
5. **CORS**: Configure CORS settings in `main.py` for your frontend domain
6. **Environment**: Use environment variables for API URL in production
7. **Retry Logic**: Implement exponential backoff for failed requests

### Production Deployment

For production use, update CORS settings in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## Configuration

### Chunking Parameters

In `dataLoading.py`, adjust these parameters for different chunking strategies:

```python
text_chunks = chunk_documents(
    all_documents,
    chunk_size=1000,      # Size of each chunk in characters
    chunk_overlap=200     # Overlap between chunks
)
```

### Retrieval Parameters

In `rag.py`, adjust the number and quality of retrieved documents:

```python
retriever=vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={
        "k": 5,           # Number of documents to retrieve
        "fetch_k": 20,    # Number of documents to fetch before filtering
        "lambda_mult": 0.7  # Diversity of results (0=max diversity, 1=min diversity)
    }
)
```

### LLM Temperature

Control response creativity in `rag.py`:

```python
base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.1,  # Lower = more focused, Higher = more creative
    max_new_tokens=512,
)
```

## How It Works

### Intelligent Caching System

The system now includes smart caching that:
- 📊 **Tracks file changes** using MD5 hashing
- ⚡ **Loads instantly** from cache if nothing changed
- 🔄 **Auto-rebuilds** only when files are added/modified
- 💾 **Saves metadata** about your document collection
- 🎯 **Filters sources** for relevance to questions

### Data Loading Pipeline (`dataLoading.py`)

1. **Change Detection**
   - Computes hash of all PDF files and URLs
   - Compares with previous hash from metadata
   - Only rebuilds if changes detected

2. **Document Loading**
   - Loads PDFs using `PyPDFLoader`
   - Loads web pages using `WebBaseLoader`
   - Combines all documents into a single list

3. **Text Chunking**
   - Splits documents into smaller chunks using `RecursiveCharacterTextSplitter`
   - Maintains context with overlapping chunks

4. **Embedding Generation**
   - Converts text chunks into vector embeddings
   - Uses `sentence-transformers/all-MiniLM-L6-V2` model

5. **Vector Store Creation**
   - Creates FAISS index from embeddings
   - Saves to disk with metadata for fast reloading

### RAG Pipeline (`rag.py`)

1. **Question Processing**
   - User asks a question
   - Question is converted to embedding

2. **Document Retrieval**
   - FAISS finds top-k most similar document chunks using MMR
   - Filters for relevance to the question
   - Returns relevant context

3. **Answer Generation**
   - Combines retrieved context with question
   - Sends to Mixtral LLM via prompt template
   - Returns answer with source citations

### API Layer (`main.py`)

1. **FastAPI Server**
   - Provides REST endpoints for all operations
   - Handles CORS for cross-origin requests
   - Implements streaming responses

2. **Source Filtering**
   - Checks relevance of retrieved documents
   - Only shows sources that actually answer the question
   - Falls back to top 3 if no highly relevant sources found

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint

### `GET /status`
Get vectorstore status and metadata

### `POST /ask`
Ask a legal question (standard response)
- Request: `{"question": "your question"}`
- Response: Full answer with sources

### `POST /ask-stream`
Ask a legal question with streaming response
- Request: `{"question": "your question"}`
- Response: Server-Sent Events stream

### `POST /rebuild`
Force rebuild of vectorstore from scratch
- Use after adding new documents
- Takes 1-2 minutes to complete

### `GET /examples`
Get example questions and usage tips

### `GET /docs`
Interactive API documentation (Swagger UI)

### `GET /redoc`
Alternative API documentation (ReDoc)

## Troubleshooting

### Common Issues

**1. "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**2. "FAISS index not found"**
- The system will automatically build it on first run
- Wait 1-2 minutes for initial build

**3. "HuggingFace API token error"**
- Ensure your `.env` file contains a valid token
- Check token permissions at https://huggingface.co/settings/tokens

**4. "Out of memory error"**
- Reduce `chunk_size` in `dataLoading.py`
- Reduce `k` (number of retrieved documents) in `rag.py`

**5. Slow response times**
- First query loads the model (can take 30-60 seconds)
- Subsequent queries are faster
- Consider using smaller models for local deployment

**6. API not accessible from other devices**
- Make sure you're using `host="0.0.0.0"` in uvicorn
- Check firewall settings

**7. CORS errors in browser**
- Update `allow_origins` in `main.py` to include your frontend URL

## Performance Optimization

### For faster embedding generation:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-V2",
    model_kwargs={'device': 'cuda'},  # Use GPU if available
    encode_kwargs={'normalize_embeddings': True}
)
```

### For local LLM deployment:
Consider replacing the Hugging Face API with local models using:
- Ollama
- LlamaCPP
- GPT4All

## Limitations

- Answers are limited to information in the loaded documents
- Response quality depends on document quality and chunking strategy
- API rate limits may apply for Hugging Face hosted models
- Large document sets may require significant disk space for FAISS index
- First-time model loading can take 30-60 seconds

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

- LangChain for the RAG framework
- Hugging Face for embeddings and LLM hosting
- FAISS for efficient vector search
- Sentence Transformers for embedding models
- FastAPI for the modern, fast web framework

**Note**: This system is for informational purposes only and should not be considered legal advice. Always consult qualified legal professionals for legal matters.