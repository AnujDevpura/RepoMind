import os
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
from src.config import LLM_MODEL_NAME, LLM_PROVIDER

# Load environment variables (API Keys)
load_dotenv()

class LLMEngine:
    def __init__(self, provider: str = None, model_name: str = None):
        self.provider = (provider or LLM_PROVIDER).lower()
        self.model_name = model_name or LLM_MODEL_NAME
        
        print(f"🧠 Initializing LLM Engine with provider: '{self.provider}', model: '{self.model_name}'...")
        
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("❌ GROQ_API_KEY not found in .env file. Please add it!")
            self.llm = Groq(model=self.model_name, api_key=api_key)
            
        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ OPENAI_API_KEY not found in .env file. Please add it!")
            self.llm = OpenAI(model=self.model_name, api_key=api_key)
            
        elif self.provider == "ollama":
            # Ollama runs locally, no API key needed
            self.llm = Ollama(model=self.model_name, request_timeout=120.0)
            
        else:
            raise ValueError(f"❌ Unsupported LLM provider: {self.provider}")

    def stream_chat(self, user_query: str, context_nodes: list, history: list):
        """
        Generates a streaming response using the LLM, memory, and code context.
        Enforces a Chain-of-Thought scratchpad to eliminate code hallucination.
        """
        if not user_query or not user_query.strip():
            yield "Please provide a valid question."
            return
        
        # 1. Construct the Context String (The "Evidence")
        context_str = ""
        if not context_nodes:
            context_str = "No code context was retrieved. Please try a different query."
        else:
            for i, node in enumerate(context_nodes, 1):
                metadata = getattr(node, 'metadata', {})
                file_path = metadata.get('file_path', 'Unknown File')
                
                try:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                except Exception as e:
                    content = f"[Error retrieving content: {e}]"
                
                context_str += f"\n=== Source {i}: {file_path} ===\n{content}\n"

        # 2. Construct the Upgraded System Prompt
        system_prompt = (
            "You are RepoMind, an elite AI Architect and Senior Software Engineer. You excel at explaining complex codebases.\n\n"
            "MANDATORY EXECUTION PROTOCOL:\n"
            "1. Information Synthesis: Synthesize the provided Code Context to formulate a comprehensive, high-quality answer. Do not just blindly quote code; explain how it works together.\n"
            "2. Anchoring & Hallucination Prevention: If the provided Code Context does NOT contain the answer, politely state that you do not have enough context. DO NOT invent or hallucinate code, logic, or file names.\n"
            "3. Citations: You MUST frequently cite your sources. When mentioning a function, class, or logic, specify the exact file path (e.g., 'In `src/app.py`...').\n"
            "4. Formatting: Your output MUST be beautifully formatted in Markdown. Use headings (`###`), bullet points, tables, and fenced code blocks (`python`) to make your explanation highly readable and structured.\n"
            "5. Tone: Be professional, insightful, and concise."
        )

        messages = [ChatMessage(role="system", content=system_prompt)]
        
        # 3. Inject Conversational Memory (History)
        for interaction in history:
            if len(interaction) == 2:
                user_msg, bot_msg = interaction
                if user_msg: messages.append(ChatMessage(role="user", content=user_msg))
                if bot_msg: messages.append(ChatMessage(role="assistant", content=bot_msg))

        # 4. Construct the User Prompt
        user_prompt = f"Here is the Code Context you discovered:\n{context_str}\n\nUser Question: {user_query}"
        messages.append(ChatMessage(role="user", content=user_prompt))

        # 5. Generate Streamed Response
        try:
            response_stream = self.llm.stream_chat(messages)
            for chunk in response_stream:
                if chunk.delta:
                    yield chunk.delta
        except Exception as e:
            yield f"\n❌ Error generating response: {e}"

if __name__ == "__main__":
    # Sanity Check
    try:
        llm = LLMEngine()
        print("✅ LLM Connected! Testing streaming pipeline...")
        
        # Test stream without context
        test_stream = llm.stream_chat("What is your mandatory execution protocol?", [], [])
        print("🤖 Response: ", end="")
        for token in test_stream:
            print(token, end="", flush=True)
        print("\n✅ Streaming test complete.")
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")