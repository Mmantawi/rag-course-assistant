import os
import sys
from dotenv import load_dotenv

# Load configuration
load_dotenv()

try:
    from backend.prompts import RAG_PROMPT
except ModuleNotFoundError:
    from prompts import RAG_PROMPT

def get_llm(model_choice: str):
    """
    Returns the appropriate LangChain ChatModel instance based on model_choice.
    """
    try:
        from backend.config import (
            LLM_MODEL,
            GEMINI_API_KEY,
            GEMINI_MODEL,
            GROQ_API_KEY,
            GROQ_MODEL
        )
    except ModuleNotFoundError:
        from config import (
            LLM_MODEL,
            GEMINI_API_KEY,
            GEMINI_MODEL,
            GROQ_API_KEY,
            GROQ_MODEL
        )

    choice = model_choice.lower() if model_choice else "local"

    if "gemini" in choice:
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if not api_key or not api_key.strip():
            raise ValueError(
                "Gemini API key is missing. Please add your GEMINI_API_KEY in the .env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, temperature=0)
        
    elif "groq" in choice:
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key or not api_key.strip():
            raise ValueError(
                "Groq API key is missing. Please add your GROQ_API_KEY in the .env file."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, groq_api_key=api_key, temperature=0)
        
    else:  # local (Ollama)
        ollama_model = LLM_MODEL
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=ollama_model, temperature=0)

def generate_answer(question, context, llm_model):
    """
    Formulates a prompt using the context and question, queries the selected LLM,
    and returns the generated answer.
    """
    llm = get_llm(llm_model)
    
    # Format the prompt with inputs
    formatted_prompt = RAG_PROMPT.format(context=context, question=question)
    
    # Invoke the model
    response = llm.invoke(formatted_prompt)
    return response.content.strip()

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    model_name = os.getenv("LLM_MODEL", "llama3.2")
    
    # Mock test data
    test_context = (
        "Grading Distribution for the ML course:\n"
        "- Midterm: 15%\n"
        "- Year Work (Lab + Assignments): 35%\n"
        "- Final Exam: 50%"
    )
    test_question = "What is the percentage of the final exam?"
    
    print(f"Testing generator with model: '{model_name}'...")
    print(f"Mock Context:\n{test_context}")
    print(f"Mock Question: {test_question}\n")
    
    try:
        answer = generate_answer(test_question, test_context, model_name)
        print("--- Generated Answer ---")
        print(answer)
        print("------------------------")
        
        # Test "I don't know" fallback
        unknown_question = "What is the capital of Egypt?"
        print(f"\nMock Question 2 (Not in context): {unknown_question}")
        answer2 = generate_answer(unknown_question, test_context, model_name)
        print("--- Generated Answer 2 ---")
        print(answer2)
        print("--------------------------")
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}", file=sys.stderr)
        sys.exit(1)
