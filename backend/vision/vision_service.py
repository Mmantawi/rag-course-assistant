import os
import base64
from langchain_core.messages import HumanMessage

class VisionService:
    """
    Modular Vision-Language Model Service wrapper.
    Supports local execution (Ollama) and cloud APIs (Gemini, Groq).
    """
    def __init__(self, provider=None, model=None):
        self.provider = (provider or os.getenv("VISION_PROVIDER", "ollama")).lower()
        self.model = model or os.getenv("VISION_MODEL", "qwen2.5-vl")
        
    def _encode_image_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def generate_caption(self, image_path, prompt):
        """
        Sends the image and prompt to the VLM and returns the detailed description content.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
            
        base64_image = self._encode_image_base64(image_path)
        
        # Detect mime type from file extension
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "gif", "webp"] else "image/jpeg"
        
        # Build Standard LangChain Multimodal content payload
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ]
        )
        
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment variables.")
            # Fallback model naming
            model_name = self.model if self.model != "qwen2.5-vl" else "gemini-2.5-flash"
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.1)
            
        elif self.provider == "groq":
            from langchain_groq import ChatGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set in environment variables.")
            # Fallback model naming
            model_name = self.model if self.model != "qwen2.5-vl" else "llama-3.2-11b-vision-preview"
            llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.1)
            
        else:  # default local ollama provider
            from langchain_ollama import ChatOllama
            model_name = self.model
            llm = ChatOllama(model=model_name, temperature=0.1)
            
        response = llm.invoke([message])
        return response.content.strip()
