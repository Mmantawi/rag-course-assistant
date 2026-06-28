import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()
model_name = os.getenv("LLM_MODEL", "llama3.2")

print(f"Initializing connection to Ollama model: '{model_name}'...")

try:
    # Initialize ChatOllama LLM
    llm = ChatOllama(model=model_name, temperature=0)
    
    # Query the LLM
    prompt = "Hello! Please tell me your model name and answer: what is 2 + 2?"
    print(f"Sending prompt: \"{prompt}\"")
    
    response = llm.invoke(prompt)
    
    print("\n--- LLM Response ---")
    print(response.content)
    print("--------------------")
    print("\nSuccess! The LLM is functional.")
except Exception as e:
    print(f"\n[ERROR] Failed to query Ollama model: {e}")
    print("Please ensure Ollama is running and the model is fully pulled.")
