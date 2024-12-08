# PDF Text and Table Embedding with Multimodal RAG

This project allows you to extract text and tables from PDF documents, generate embeddings for both using transformer-based models, store them in Pinecone Vector Database, and provide a FastAPI-based query interface to interact with the stored embeddings. It utilizes models like `SentenceTransformer` for text embeddings, `TableTransformerForObjectDetection` for table embeddings, and combines them in a retrieval-augmented generation (RAG) approach using GPT for generating responses.

## Requirements

- Python 3.7+
- Virtual environment (optional but recommended)
- Pinecone API Key (for Pinecone Vector Database)
- OpenAI API Key (for GPT integration)

### Required Libraries
You can install all the required libraries using pip. Run the following command:

```bash
pip install -r requirements.txt
uvicorn tech:app --reload //run uvicorn server