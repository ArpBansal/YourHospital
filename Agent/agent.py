import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json
import time
# import google.generativeai as genai
from google import genai
from google.oauth2 import service_account
load_dotenv()
# credential_file = os.getenv("GOOGLE_APPLICATION_CREDENTIAL")
# # Load environment variables
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_file

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

conf = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

service_account_info = json.loads(conf)
credentials = service_account.Credentials.from_service_account_info(service_account_info)

DOCUMENT_DIR = 'document/'
COLLECTION_NAME = "health_documents"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", GEMINI_API_KEY=GEMINI_API_KEY, temperature=0.7, credentials=credentials)


print("Models initialized successfully.")
def load_documents(directory):
    loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    docx_loader = DirectoryLoader(
        directory,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        loader_kwargs={"mode": "elements"}
    )
    documents.extend(docx_loader.load())
    print(f"Loaded {len(documents)} documents.")
    pdf_loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader
    )
    documents.extend(pdf_loader.load())
    print(f"Loaded {len(documents)} documents.")
    return documents

import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFacePipeline
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.vectorstores import Chroma
import torch
from constants import CHROMA_PATH

# Load environment variables (if needed)
load_dotenv()

# Define the directory containing the documents
DOCUMENT_DIR = 'document/'

# Define the path for ChromaDB persistent storage

# Initialize Hugging Face Embeddings
# You can change the model to any suitable embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/gtr-t5-large",
    model_kwargs={'device': 'cpu'}
)


def load_documents(directory):
    """Load documents from multiple file types."""
    documents = []
    
    # Load text files
    text_loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents.extend(text_loader.load())

    # Load Word documents
    docx_loader = DirectoryLoader(
        directory,
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader,
        loader_kwargs={"mode": "elements"}
    )
    documents.extend(docx_loader.load())

    # Load PDF files
    pdf_loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader
    )
    documents.extend(pdf_loader.load())
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_and_store_embeddings(chunks):
    """Create or load ChromaDB vector store."""
    # Ensure the Chroma path exists
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print(f"unfiltered chunks: {len(chunks)}")
    filtered_chunks = []
    for chunk in chunks:
        # Create a new document with filtered metadata
        filtered_metadata = {k: v for k, v in chunk.metadata.items() 
                             if isinstance(v, (str, int, float, bool))}
        chunk.metadata = filtered_metadata
        filtered_chunks.append(chunk)

    print(f"Filtered metadata for {len(filtered_chunks)} chunks.")
    # Create or load the vector store
    vector_store = Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print("Created ChromaDB vector store.")
    return vector_store

def load_vectordb(path:str=CHROMA_PATH):
    if os.path.exists(path):
        vector_store = Chroma(persist_directory=path, embedding_function=embeddings)
        print("Loaded ChromaDB vector store.")
        return vector_store
    else:
        raise ValueError(f"ChromaDB path {path} does not exist.")

from langchain.chains import LLMChain
from langchain.chains.base import Chain

def create_health_agent(vector_store):
    """Create a custom retrieval QA chain for health-related queries."""
    prompt_template = """You are a helpful health assistant. Who will talk to the user as human and resolve their queries.

    Use Previous_Conversation to maintain consistency in the conversation.
    These are Previous_Conversation between you and user.
    Previous_Conversation: \n{previous_conversation}
    Thoroughly analyze the Context, and also use context to answer the questions, aside of your knowledge.
    Keep the answer concise.
        
    Context: {context}
    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question", "previous_conversation"]
    )

    if llm is None:
        raise ValueError("No language model initialized. Please check the model initialization.")

    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    class CustomRetrievalQA(Chain):
        retriever: object
        llm_chain: LLMChain

        @property
        def input_keys(self):
            return ['query', 'previous_conversation']

        @property
        def output_keys(self):
            return ['result']

        def _call(self, inputs):
            query = inputs['query']
            previous_conversation = inputs.get('previous_conversation', '')
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            # Prepare inputs for the LLM chain
            llm_inputs = {
                'context': context,
                'question': query,
                'previous_conversation': previous_conversation
            }

            # Generate response
            result = self.llm_chain(llm_inputs)
            return {'result': result['text']}

    # Create the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    # Create and return the custom chain
    return CustomRetrievalQA(retriever=retriever, llm_chain=llm_chain)


def agent_with_db():
    # 1. Load documents
    vector_store = load_vectordb(CHROMA_PATH)
    UPDATE_DB = os.getenv("UPDATE_DB")
    if UPDATE_DB.lower()=="true":
        UPDATE_DB = True
    if vector_store is None or UPDATE_DB is True:
        print("Loading documents...")
        print(vector_store, UPDATE_DB)
        
        documents = load_documents(DOCUMENT_DIR)

        print("Splitting documents into chunks...")
        chunks = split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")

        print("Creating and storing embeddings in ChromaDB...")
        try:
            vector_store = create_and_store_embeddings(chunks)
            print("Embeddings stored successfully in ChromaDB.")
        except Exception as e:
            print(f"An error occurred while creating or storing embeddings: {e}")
            return

    print("Creating the health agent...")
    health_agent = create_health_agent(vector_store)

    return health_agent