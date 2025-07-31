 # app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from generator import qa_chain  # assuming generator.py has a working qa_chain

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "RAG Assistant is running "}

@app.post("/ask")
def ask_question(payload: QueryInput):
    print(f"🔍 Received query: {payload.query}")
    try:
        response = qa_chain.invoke(payload.query)
        print("✅ Response:", response)
        return {"answer": response}
    except Exception as e:
        print("❌ Error in /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

