from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Embedding model
LLM_MODEL = "google/flan-t5-base"  # FLAN-T5 model
CHROMADB_DIRECTORY = "./chromadb_store"  # Directory to persist ChromaDB

# Initialize FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

# Initialize embedding model and ChromaDB vector store
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMADB_DIRECTORY,
    embedding_function=embedding_model # Pass embedding function correctly
)

@app.route('/halo', methods= ['GET'])
def hello():
    return jsonify({"status": "hello cutie"}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Healthy"}), 200

# Route to upload a PDF and process it
@app.route('/upload', methods=['POST'])
def upload_pdf():

#     vectorstore = Chroma(
#     persist_directory=CHROMADB_DIRECTORY,
#     embedding_function=embedding_model  # Pass embedding function correctly
# )

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file to a temporary location
    temp_file_path = os.path.join("./temp", file.filename)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    file.save(temp_file_path)
    try:
        loader = PDFPlumberLoader(temp_file_path)
        documents = loader.load()
        texts = [doc.page_content for doc in documents]
        vectorstore.add_texts(texts)
        os.remove(temp_file_path)

        return jsonify({"message": f"Processed and stored {len(texts)} pages"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/question', methods=['POST'])
def query_bot():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data['question']

    try:
        retrieved_docs = vectorstore.similarity_search(question, k=5)
        context = " ".join([doc.page_content for doc in retrieved_docs])
        input_text = f"Context: {context} Question: {question}"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs,max_length=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"answer": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/embed', methods=['POST'])
def generate_embeddings():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    try:
        embedding = embedding_model.embed_query(data['text'])
        return jsonify({"embeddings": embedding}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
