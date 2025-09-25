# Campus Sahayak

Campus Sahayak is an interactive AI-powered chatbot application designed to assist users with queries related to campus courses, admissions, and notices. The system enables uploading of PDF brochures/notices, dynamically updates its knowledge base, and provides relevant answers by retrieving information from uploaded documents.

## Features

- Upload PDF notices and brochures that get added to a vector database for semantic search.
- AI chatbot interface to ask questions related to campus courses, admissions, and other information.
- Multiple chat threads support with session state management.
- Query processing using LangChain with state-of-the-art embeddings and retrieval techniques.
- Integration with ChatGroq LLM for generating responses.
- Persistent vector store using Chroma for document search.
- SQLite-based conversation history checkpointing.

## Installation

1. Clone the repository.
2. Create a Python virtual environment (recommended):
python -m venv myenv
source myenv/bin/activate # On Windows use myenv\Scripts\activate

text
3. Install the required dependencies:
pip install -r requirements.txt

## Usage

### Running the Application

1. Make sure you have a `.env` file with your API key for ChatGroq:
   GROQAPIKEY=your_groq_api_key_here
2. Run the frontend Streamlit app:
   streamlit run frontend.py
3. Use the sidebar to:
- Start a new chat thread.
- Upload new PDF notices (will be added to the searchable vector store).
- Switch between chat threads.

4. Type your questions about campus information in the chat input.

### Backend Overview

- The backend loads PDF brochures from a specified folder and creates embeddings for semantic search.
- It uses LangChain's `RetrievalQA` with a ChatGroq LLM model to answer queries based on the indexed documents.
- Chat state and conversation threads are saved and restored from an SQLite database.
- You can dynamically add new PDF notices at runtime through the frontend interface.

## Code Structure

- `frontend.py`: Streamlit-based interactive UI for the chatbot, handles user input, chat threads, and PDF uploads.
- `backend.py`: Core logic for loading brochures, creating vector stores, querying the LLM, and managing chat state.
- `requirements.txt`: Python dependencies required to run the project.

## Dependencies

Key dependencies include (full list in `requirements.txt`):

- streamlit
- langchain
- langchain-groq
- chromadb
- sqlite3
- python-dotenv
- PyPDF2 (via PyPDFLoader)
- HuggingFaceEmbeddings

## Environment Variables

- `GROQAPIKEY`: Your API key for the Groq LLM service.

## Folder Structure

- `/brochures`: Directory to store PDF brochures and notices that are loaded and indexed at runtime.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License

This project is open-source and available under the MIT License.


