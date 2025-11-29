import os
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
from src.config import LLM_MODEL_NAME

# Load environment variables (API Keys)
load_dotenv()

class LLMEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file. Please add it!")
            
        print(f"🧠 Initializing LLM: {LLM_MODEL_NAME}...")
        self.llm = Groq(model=LLM_MODEL_NAME, api_key=api_key)

    def chat(self, user_query: str, context_nodes: list) -> str:
        """
        Generates a response using the LLM and the retrieved code context.
        
        Args:
            user_query: The user's question
            context_nodes: List of retrieved nodes from the vector store
        
        Returns:
            LLM-generated response string
        """
        if not user_query or not user_query.strip():
            return "Please provide a valid question."
        
        # 1. Construct the Context String (The "Evidence")
        context_str = ""
        if not context_nodes:
            context_str = "No code context was retrieved. Please try a different query."
        else:
            for i, node in enumerate(context_nodes, 1):
                # Safe access to metadata
                metadata = getattr(node, 'metadata', {})
                file_path = metadata.get('file_path', 'Unknown File')
                
                # Get content safely
                try:
                    content = node.get_content() if hasattr(node, 'get_content') else str(node)
                except Exception as e:
                    content = f"[Error retrieving content: {e}]"
                
                # Format: File Path + Code Content
                context_str += f"\n=== Source {i}: {file_path} ===\n"
                context_str += f"{content}\n"

        # 2. Construct the System Prompt (The "Personality")
        system_prompt = (
            "You are RepoMind, an expert Senior Software Architect assisting a developer.\n"
            "Your Goal: Answer the user's question based STRICTLY on the provided Code Context.\n\n"
            "Rules:\n"
            "1. Citations: When you explain logic, mention the file name (e.g., 'In `auth.ts`...').\n"
            "2. Hallucination: If the answer is NOT in the context, say 'I cannot find that logic in the retrieved files.'\n"
            "3. Format: Use Markdown formatting for code blocks (```python, ```typescript, etc.).\n"
            "4. Tone: Be concise, technical, and direct."
        )

        # 3. Construct the User Prompt
        user_prompt = (
            f"Here is the Code Context you discovered:\n{context_str}\n\n"
            f"User Question: {user_query}"
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        # 4. Generate Response
        try:
            response = self.llm.chat(messages)
            return response.message.content if hasattr(response, 'message') else str(response)
        except Exception as e:
            return f"❌ Error generating response: {e}"

if __name__ == "__main__":
    # Sanity Check
    try:
        llm = LLMEngine()
        print("✅ LLM Connected! Testing simple query...")
        response = llm.chat("Hello!", []) # Empty context test
        print(f"🤖 Response: {response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")