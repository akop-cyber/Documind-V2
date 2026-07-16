from fastapi import HTTPException, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import aiofiles
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from huggingface_hub import InferenceClient
import os
import uuid
import json
from loader import Loader
from chunker import Chunker
from embedder import Embedder
from bm25 import BM25
from vector_store import Vectorstore
from retriever import Retriever

print("done")

embedder = Embedder()
sessions: dict = {}

print("embedder initialized")

MODELS = [
    ("Qwen/Qwen2.5-72B-Instruct"),
    ("meta-llama/Llama-3.2-3B-Instruct"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("mistralai/Mistral-7B-Instruct-v0.3"),
    ("HuggingFaceH4/zephyr-7b-beta"),
]

system_prompt = (
    "You are a helpful study assistant. Answer the user's question based ONLY on the provided context.\n\n"
    "if the answer is not contained within the context, respond with: 'I couldn't find relevant information in the document.'\n\n(STRICT)"
    "FORMATTING RULES (STRICT):\n"
    "You MUST format your entire response using valid Markdown.\n"
    "1. Use `##` for main section headings.\n"
    "2. Use `**bold text**` for subheadings.\n"
    "3. Use `- ` (a hyphen followed by a space) for bullet points.\n"
    "4. CRITICAL: You MUST leave a completely blank line before every heading and bullet point.\n"
    "5. Do not write long paragraphs. Keep points concise."
)

async def cleanup_loop(sessions: dict):
    while True:
        await asyncio.sleep(3600)
        now = datetime.now()
        for sid in list(sessions.keys()):
            if sessions[sid]["expires_at"] < now:
                del sessions[sid]

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cleanup_loop(sessions))
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: list = Field(default_factory=list)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    tmp_path = None
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            await tmp.write(content)
            tmp_path = tmp.name
        
        extracted_text = await asyncio.to_thread(Loader(tmp_path).load)
        chunks = await asyncio.to_thread(Chunker(extracted_text).chunk)
        embedded_chunks = await asyncio.to_thread(embedder.embed, chunks)
        
        vector_store = Vectorstore(embedder)
        await asyncio.to_thread(vector_store.add_vectors, embedded_chunks)
        
        bm25 = BM25()
        await asyncio.to_thread(bm25.add, chunks)
        
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "store": vector_store,
            "bm25": bm25,
            "expires_at": datetime.now() + timedelta(hours=24),
        }
        
        return {"message": "PDF indexed successfully!", "session_id": session_id}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)



@app.post("/chat")
async def chat(chat_req: ChatRequest):
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not configured")
    
    if not chat_req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    session = sessions.get(chat_req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    bm25 = session["bm25"]
    vector_store = session["store"]
    retriever = Retriever(vector_store=vector_store, bm25=bm25)
    
    context_chunks = await asyncio.to_thread(retriever.retrieve, chat_req.message)
    
    if not context_chunks:
        async def empty_stream():
            ymsg = "I couldn't find relevant information in the document."
            yield f"data: {json.dumps({'token': ymsg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")
    
    context_text = "\n\n".join(context_chunks)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_req.history)
    messages.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {chat_req.message}"})

    def stream_chat():
        success = False  
        
        for model in MODELS:
            try:
                client = InferenceClient(model, token=hf_token)
                for texts in client.chat_completion(messages, max_tokens=512, stream=True):
                    text = texts.choices[0].delta.content
                    if text:
                        success = True
                        
                        yield f"data: {json.dumps({'token': text})}\n\n"
        
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
              
                print(f"Model {model} failed: {e}")
                continue  

        error_msg = "Sorry, all models are currently unavailable. Try again later."
        yield f"data: {json.dumps({'token': error_msg})}\n\n"
        yield "data: [DONE]\n\n"


    return StreamingResponse(stream_chat(), media_type="text/event-stream")
            



    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)