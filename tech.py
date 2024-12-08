from pdf2image import convert_from_path
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import pdfplumber
import openai
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Initialize models
text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
table_transformer_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY") # Replace with your actual Pinecone API key
pinecone = Pinecone(api_key=api_key)

# Ensure the index exists
if 'pdf-index' not in pinecone.list_indexes().names():
    pinecone.create_index(
        name='pdf-index',
        dimension=384,  # Match the embedding dimension (e.g., SentenceTransformer)
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pinecone.Index("pdf-index")

# FastAPI App
app = FastAPI()

def text_to_pdf(query: str, pdf_filename: str):
    # Create a PDF file with the given query text
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter  # Size of the letter page
    
    # Set font and size
    c.setFont("Helvetica", 12)
    
    # Position and add the query text to the PDF
    c.drawString(100, height - 100, query)
    
    # Save the PDF
    c.save()
# ----------------------------
# 1. PDF Reading
# ----------------------------
def extract_text_and_tables(pdf_path):
    text_content = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text_content.append(page.extract_text())
            
            # Extract tables (standard table extraction)
            tables.extend(page.extract_tables())

    return " ".join(text_content), tables

# ----------------------------
# 2. Table Extraction with Table Transformer
# ----------------------------
def extract_tables_with_transformer(pdf_path):
    images = convert_from_path(pdf_path)  # Convert PDF pages to images
    table_data = []

    # Process each image (representing a page)
    for image in images:
        inputs = processor(images=image, return_tensors="pt")
        outputs = table_transformer_model(**inputs)

        # Collect the output embeddings for the table
        table_data.append(outputs.last_hidden_state)

    return table_data

# ----------------------------
# 3. Text Embedding
# ----------------------------
def generate_text_embeddings(text_segments):
    embeddings = text_embedding_model.encode(text_segments, convert_to_tensor=True)
    return embeddings

# ----------------------------
# 4. Table Embedding (Using Table Transformer)
# ----------------------------
def generate_table_embeddings_with_transformer(pdf_path):
    # Extract tables using the transformer-based method
    table_embeddings = extract_tables_with_transformer(pdf_path)
    
    # Convert list of table embeddings to a tensor
    table_embeddings_tensor = torch.stack(table_embeddings)  # Stack into a tensor

    # Pool embeddings (mean pooling across the second dimension)
    pooled_embeddings = table_embeddings_tensor.mean(dim=1)  # Pool the table embeddings
    return pooled_embeddings  # Shape: (num_tables, embedding_dim)

def adjust_table_embeddings(table_embeddings, target_dim=384):
    # If embeddings have 3 dimensions, pool the rows by averaging them
    if len(table_embeddings.shape) == 3:
        # Pool across rows by taking the mean over dimension 1 (the rows of the table)
        table_embeddings = table_embeddings.mean(dim=1)  # Resulting shape: [batch_size, embedding_dim]
    
    # Now check if the embedding size matches the target dimension
    if table_embeddings.shape[1] < target_dim:
        # If embedding dimension is smaller, we can pad with zeros
        padding = torch.zeros(table_embeddings.shape[0], target_dim - table_embeddings.shape[1])
        table_embeddings = torch.cat((table_embeddings, padding), dim=1)
    elif table_embeddings.shape[1] > target_dim:
        # If embedding dimension is larger, truncate it
        table_embeddings = table_embeddings[:, :target_dim]
    
    return table_embeddings

# ----------------------------
# 5. Store Embeddings in VectorDB
# ----------------------------
def store_embeddings_in_vectordb(embeddings, metadata, batch_size=200):
    assert len(embeddings) == len(metadata), "Embedding length and metadata length do not match"
    
    # Split data into smaller batches to avoid size issues
    for i in range(0, len(embeddings), batch_size):
        upsert_data = []
        for j in range(i, min(i + batch_size, len(embeddings))):
            upsert_data.append((str(j), embeddings[j].tolist(), metadata[j]))
        
        try:
            index.upsert(upsert_data)
            print(f"Batch {i//batch_size + 1} upserted successfully.")
        except Exception as e:
            print(f"Error in upserting batch {i//batch_size + 1}: {e}")

# ----------------------------
# 6. Multimodal RAG
# ----------------------------
def multimodal_rag_query(query_embedding, top_k=10):
    search_results = index.query(
        vector=query_embedding.tolist(),  # Ensure it's a list
        top_k=top_k,                       # Dynamically set the number of results to return
        include_metadata=True             # Include metadata in the response
    )
    return search_results


class QueryRequest(BaseModel):
    query: str

class StoreRequest(BaseModel):
    pdf_path: str  # Path to PDF file to extract and store embeddings

# ----------------------------
# 7. Chatbot API
# ----------------------------
# Function to call GPT-4o Mini via OpenAI API
def generate_gpt_response(results, query):
    """
    Generate a coherent response using GPT-4o Mini based on the query and results.
    """
    if not results:
        return "No relevant results were found to answer the query."

    # Create a list of messages to simulate a conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Query: '{query}'"},
    ]

    # Add results to the conversation context
    for result in results:
        content_type = result['type']
        content = result['content'][:500]  # Truncate long content to avoid exceeding token limits
        score = result['score']
        messages.append({
            "role": "assistant",
            "content": f"Type: {content_type}, Score: {score:.2f}\nContent: {content}"
        })

    # Make the API call to GPT-4o Mini
    try:
        # Provide your OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Make the API request to OpenAI's GPT-4o Mini
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Specify the GPT-4o Mini model
            messages=messages,    # Pass the conversation as messages
            max_tokens=1024,      # Adjust the max token length based on your needs
            temperature=0.7       # Control randomness (0.0 for deterministic, 1.0 for more creativity)
        )

        # Extract the generated response from the API response
        gpt_response = response['choices'][0]['message']['content'].strip()
        return gpt_response
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"


# ----------------------------
# 8. Process Pipeline
# ----------------------------
def process_pdf_pipeline(pdf_path):
    # Extract text and tables using traditional extraction
    text, tables = extract_text_and_tables(pdf_path)

    # Generate embeddings for both text and tables
    transformer_table_embeddings = generate_table_embeddings_with_transformer(pdf_path)

    # Split text into smaller segments
    text_segments = text.split(".")  # Use a more sophisticated method if needed

    # Generate text embeddings
    text_embeddings = generate_text_embeddings(text_segments)

    # Ensure table embeddings are in the correct shape
    transformer_table_embeddings = adjust_table_embeddings(transformer_table_embeddings, target_dim=384)

    # Move embeddings to the same device (MPS or CPU)
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    text_embeddings = text_embeddings.to(device)
    transformer_table_embeddings = transformer_table_embeddings.to(device)

    # Adjust table metadata to match the number of table embeddings
    table_metadata = [{'type': 'table', 'content': str(tables[i]), 'page': 1} for i in range(len(transformer_table_embeddings))]

    # Adjust text metadata to match the number of text segments
    text_metadata = [{'type': 'text', 'content': str(segment), 'page': 1} for segment in text_segments]

    # Combine text and table metadata
    metadata = text_metadata + table_metadata

    # Ensure embeddings and metadata lengths match
    assert len(text_embeddings) + len(transformer_table_embeddings) == len(metadata)

    # Concatenate the embeddings
    embeddings = torch.cat((text_embeddings, transformer_table_embeddings), dim=0)

    # Store embeddings in the vector database (Pinecone)
    store_embeddings_in_vectordb(embeddings, metadata)

# --------
@app.post("/query")
def query_pdf(request: QueryRequest):
    query = request.query

    try:
        # Generate both table and text embeddings
        text_to_pdf(query, "query_pdf.pdf")
        table_embedding = generate_table_embeddings_with_transformer("query_pdf.pdf")
        table_embedding = adjust_table_embeddings(table_embedding)
        text_embedding = generate_text_embeddings([query])[0]

        # Query VectorDB using multimodal RAG for both embeddings
        table_results = multimodal_rag_query(table_embedding)
        text_results = multimodal_rag_query(text_embedding)

        print("Raw Table Results:", table_results)
        print("Raw Text Results:", text_results)

        # Merge results if matches exist in both
        all_results = []

        def process_matches(matches, result_type):
            if matches:
                return [
                    {
                        'id': match['id'],
                        'type': match.get('metadata', {}).get('type', result_type),
                        'content': match.get('metadata', {}).get('content', 'No content available'),
                        'score': match['score']
                    }
                    for match in matches
                ]
            return []

        all_results.extend(process_matches(table_results.get('matches', []), 'table'))
        all_results.extend(process_matches(text_results.get('matches', []), 'text'))

        if not all_results:
            return jsonable_encoder({"results": "No matches found in the database."})

        # Sort all results by score in descending order
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)

        # Use GPT-4o Mini to generate a coherent response
        gpt_response = generate_gpt_response(sorted_results, query)
        print(gpt_response)

        return jsonable_encoder({"results": gpt_response})

    except Exception as e:
        return jsonable_encoder({"error": f"An error occurred during processing: {str(e)}"})

@app.post("/store")
async def store_embeddings(request: StoreRequest):
    """
    Endpoint to store embeddings for a given PDF file.
    """
    try:
        pdf_path = request.pdf_path
        # Process the PDF to extract text and tables, then generate embeddings
        process_pdf_pipeline(pdf_path)
        
        # After processing, embeddings are stored in Pinecone
        return jsonable_encoder({"status": "Embeddings stored successfully."})

    except Exception as e:
        return jsonable_encoder({"error": f"An error occurred during storing: {str(e)}"})

if __name__ == "__main__":
    # Example PDF file path
    pdf_path = "/Users/anjalitripathi/Downloads/ast_sci_data_tables_sample.pdf"
    
    # Process PDF
    process_pdf_pipeline(pdf_path)
    
    # Run FastAPI Server (Use `uvicorn <filename>:app --reload` to start server)
    print("FastAPI server is ready. Use /query endpoint to interact.")
