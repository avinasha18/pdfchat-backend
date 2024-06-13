from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from fastapi.staticfiles import StaticFiles

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mypdfchat.vercel.app"],  # Adjust this to match your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    question: str

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf, filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.post("/upload/")
async def upload(files: list[UploadFile]):
    try:
        logger.info("Starting file upload processing")
        pdf_docs = [await file.read() for file in files]
        logger.info(f"Read {len(pdf_docs)} PDF documents")
        raw_text = get_pdf_text(pdf_docs)
        logger.info(f"Extracted text from PDFs")
        text_chunks = get_text_chunks(raw_text)
        logger.info(f"Split text into {len(text_chunks)} chunks")
        get_vector_store(text_chunks)
        logger.info(f"Vector store created and saved")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/")
async def ask_question(question_request: QuestionRequest):
    try:
        question = question_request.question
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        text_response = response["output_text"]
        return JSONResponse(content={"reply": text_response})
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
