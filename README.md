# PDF Querying with GPT Response Generator

This project allows querying PDFs for specific information and generating coherent responses using GPT. The system processes multimodal queries by combining both table and text embeddings to enhance the retrieval process. The final response is generated using a GPT-based model that is fine-tuned to produce meaningful and coherent answers based on the retrieved results.

## Features

- **Multimodal Querying**: Combines embeddings for table data and text data to perform a robust query on the PDF.
- **GPT-Generated Responses**: Utilizes GPT for generating coherent and contextually relevant answers.
- **Dynamic Content Management**: Handles long content, truncating where necessary, to avoid token overflow issues.
- **Result Merging**: Merges table and text results, ranks them based on relevance, and uses them for generating responses.

## Requirements

The project requires the following libraries:

- Python 3.x
- Hugging Face `transformers` library
- PyTorch
- FastAPI
- `gpt_tokenizer` and `gpt_model` (ensure correct GPT model is initialized)
- The following helper functions need to be implemented in your project:
  - `generate_table_embeddings_with_transformer`
  - `generate_text_embeddings`
  - `multimodal_rag_query`
  - `text_to_pdf`
  - `adjust_table_embeddings`

### Installation

You can install the necessary dependencies with pip:

```bash
pip install requirements.txt


start the uvicorn server and use the API
