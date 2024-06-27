# Manta Hospital Center AI Assistant

## Overview

The Manta Hospital Center AI Assistant is a virtual assistant built using Streamlit, Langchain, and OpenAI GPT models. It is designed to provide answers to frequently asked questions (FAQs) about Manta Hospital Center by processing and analyzing the contents of a PDF file containing the center's information.

## Features

- **PDF Text Extraction**: Reads and extracts text from provided PDF files.
- **Text Chunking**: Splits the extracted text into manageable chunks for processing.
- **Vector Store Creation**: Creates a vector store from the text chunks using FAISS for efficient retrieval.
- **Conversational Retrieval Chain**: Utilizes Langchain's ConversationalRetrievalChain to create an interactive Q&A system.
- **Interactive User Interface**: Provides a user-friendly interface for users to interact with the AI assistant.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/manta-hospital-center-ai.git
   cd manta-hospital-center-ai
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPEN_AI_APIKEY=your_openai_api_key
   ```

4. Place the PDF file (`base_conocimiento_MHC.pdf`) in the root directory of the project.

## Running the Application

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided URL (usually `http://localhost:8501`).

## Application Structure

### Main Components

1. **`get_pdf_text(pdf_list)`**: Reads and extracts text from a list of PDF files.
2. **`get_text_chunks(text)`**: Splits the extracted text into chunks using Langchain's `CharacterTextSplitter`.
3. **`get_vector_store(text_chunks)`**: Creates a vector store from the text chunks using FAISS.
4. **`get_conversation_chain(vector_store)`**: Sets up the conversational retrieval chain using Langchain's `ConversationalRetrievalChain`.
5. **`handle_userInput(user_question)`**: Handles the user's input and retrieves responses from the AI assistant.
6. **`main()`**: The main function that sets up and runs the Streamlit application.

### Streamlit Interface

- **Sidebar**:
  - Displays the Manta Hospital Center logo and header.
  - Provides a button to initiate the conversation with the AI assistant.
  - Lists sample questions that users can ask.
  - Shows credits and contact information for AI training and consulting.

- **Main Area**:
  - Prompts the user to start a conversation by pressing a button.
  - Displays the chat interface where users can ask questions and receive responses from the AI assistant.

### HTML Templates

Custom HTML templates (`css`, `bot_template`, `user_template`) are used to style the chat interface and display messages.

## Usage

1. **Upload PDF**: The assistant uses a pre-defined PDF (`base_conocimiento_MHC.pdf`). Make sure it is in the root directory.
2. **Train the Assistant**: Press the "Hable con nuestra IA" button in the sidebar to train the assistant with the provided PDF content.
3. **Ask Questions**: Enter your questions in the text input field, and the AI assistant will provide answers based on the PDF content.

## Contact

For AI training and consulting, contact [BOTARMY](mailto:jjusturi@gmail.com).

## License
This project is licensed under the MIT License.

---

A project by [BOTARMY](https://botarmy-web.streamlit.app/) - Need AI training/consulting? [Get in touch](mailto:jjusturi@gmail.com).