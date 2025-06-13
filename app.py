from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PDF_DIRECTORY = "pdf_files/"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLAMA_MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"


def load_and_process_pdfs(pdf_directory):
    """Load and process PDF documents"""
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            try:
                filepath = os.path.join(pdf_directory, filename)
                logger.info(f"Processing file: {filename}")
                loader = PyPDFLoader(filepath)
                pages = loader.load_and_split()
                for page in pages:
                    page.metadata["source_file"] = filename
                documents.extend(pages)
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    return documents


def initialize_llama_model():
    """Initialize the local Llama model"""
    return LlamaCpp(
        model_path=LLAMA_MODEL_PATH,
        temperature=0.1,
        max_tokens=2000,
        n_ctx=2048,
        n_gpu_layers=8,
        n_batch=512,
        verbose=False,
    )


def setup_qa_system():
    """Set up the QA system"""
    try:
        # Load and process PDFs
        raw_docs = load_and_process_pdfs(PDF_DIRECTORY)
        if not raw_docs:
            raise ValueError("No documents were loaded. Check your PDF directory.")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs_splits = text_splitter.split_documents(raw_docs)

        # Create vector store (using local embeddings)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_db = FAISS.from_documents(docs_splits, embeddings)

        # Setup QA chain with local Llama
        llm = initialize_llama_model()

        prompt_template = """Answer the question based only on the context below.
        Always include the source file in brackets like [filename.pdf].
        If you don't know the answer, say you don't know. Don't make up answers.
        
        Context: {context}
        Question: {question}
        
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise


def main():
    try:
        print("\nInitializing fully local RAG system with Llama.cpp...")
        qa_chain = setup_qa_system()
        print("\nSystem ready! Type 'exit' to quit.")

        while True:
            query = input("\nYour question: ").strip()
            if query.lower() == "exit":
                break

            if not query:
                print("Please enter a valid question.")
                continue

            try:
                start_time = time.time()
                result = qa_chain.invoke({"query": query})

                print(f"\nAnswer (generated in {time.time() - start_time:.2f}s):")
                print(result["result"])

                if "source_documents" in result and result["source_documents"]:
                    print("\nSources:")
                    unique_sources = {
                        doc.metadata["source_file"]
                        for doc in result["source_documents"]
                    }
                    for source in unique_sources:
                        print(f"- {source}")
                else:
                    print("\nNo sources found for this answer.")

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"An error occurred: {str(e)}")

    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        print(f"Failed to initialize the system: {str(e)}")


if __name__ == "__main__":
    main()
