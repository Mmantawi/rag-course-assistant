import uvicorn
import os
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    print("==================================================")
    print(f"Starting Local RAG API Backend Server...")
    print(f"URL: http://{host}:{port}")
    print(f"API Docs: http://{host}:{port}/docs")
    print("==================================================")
    
    # Run uvicorn server pointing to the api app in backend.api
    uvicorn.run("backend.api:app", host=host, port=port, reload=False)
