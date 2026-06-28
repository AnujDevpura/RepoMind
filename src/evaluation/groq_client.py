import os
import re
import time
import asyncio
import logging
from groq import AsyncGroq, RateLimitError, APIConnectionError, APIStatusError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GroqClient")

def parse_groq_time(time_str: str) -> float:
    """Parses Groq's wait time format (e.g. '14h23m12s', '4.3s') into total seconds."""
    total_seconds = 0.0
    
    h_match = re.search(r'([0-9]+)h', time_str)
    if h_match:
        total_seconds += float(h_match.group(1)) * 3600
        
    m_match = re.search(r'([0-9]+)m', time_str)
    if m_match:
        total_seconds += float(m_match.group(1)) * 60
        
    s_match = re.search(r'([0-9]+(?:\.[0-9]+)?)s', time_str)
    if s_match:
        total_seconds += float(s_match.group(1))
        
    return total_seconds

class RobustGroqClient:
    """
    A specialized Groq client that dynamically intercepts rate limits, 
    calculates the EXACT requested wait time, waits, and automatically resumes.
    """
    def __init__(self, model_name: str, max_retries: int = 5, base_delay: float = 2.0):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables.")
            
        self.client = AsyncGroq(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def acomplete(self, prompt: str):
        """
        Executes a completion call, trapping exact rate limit wait times, 
        and guaranteeing a response even if it takes hours.
        """
        attempt = 0
        while True:
            try:
                # 1. Execute the actual completion request
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                content = response.choices[0].message.content
                if content is None:
                    content = ""
                    
                # Mock the res.text behavior that LlamaIndex expects
                class MockResponse:
                    def __init__(self, text):
                        self.text = text
                
                return MockResponse(content)

            except RateLimitError as e:
                # 2. Dynamic Rate Limit Parsing
                error_message = str(e)
                
                # Groq exact format: "Please try again in 14h23m12.5s." or "Please try again in 8.5s."
                wait_match = re.search(r'Please try again in ((?:[0-9]+h)?(?:[0-9]+m)?(?:[0-9]+(?:\.[0-9]+)?s))', error_message)
                
                if wait_match:
                    raw_time_str = wait_match.group(1)
                    wait_time = parse_groq_time(raw_time_str)
                    
                    logger.warning(
                        f"⚠️ GROQ RATE LIMIT HIT! Exact wait time requested by API: {raw_time_str}. "
                        f"Sleeping for exactly {wait_time:.3f} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                    # We do NOT increment `attempt` here because rate limits are intentional and we want to retry indefinitely after waiting.
                    continue
                else:
                    # Fallback if Groq changes their error format
                    attempt += 1
                    if attempt > self.max_retries:
                        logger.error(f"❌ Max retries ({self.max_retries}) exhausted due to unparseable rate limit.")
                        raise
                        
                    wait_time = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Rate limit hit, but no exact wait time found. "
                        f"Falling back to exponential backoff: {wait_time:.2f}s before attempt {attempt}..."
                    )
                    await asyncio.sleep(wait_time)

            except APIConnectionError as e:
                # 3. Connection Drops
                attempt += 1
                if attempt > self.max_retries:
                    raise
                wait_time = self.base_delay * (2 ** attempt)
                logger.warning(f"🔌 Connection error. Exponential backoff: {wait_time:.2f}s before attempt {attempt}...")
                await asyncio.sleep(wait_time)
                
            except APIStatusError as e:
                # 4. HTTP Status Errors
                status_code = getattr(e, 'status_code', None)
                
                # Fail immediately without retrying on fatal errors (400, 401, 403, 413, etc)
                if status_code in [400, 401, 403, 413]:
                    logger.error(f"❌ Fatal HTTP Error ({status_code}) encountered. Aborting retries: {str(e)}")
                    raise
                
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"❌ Max retries ({self.max_retries}) exhausted due to server errors.")
                    raise
                
                wait_time = self.base_delay * (2 ** attempt)
                logger.warning(f"🔌 Server error ({status_code}). Exponential backoff: {wait_time:.2f}s before attempt {attempt}...")
                await asyncio.sleep(wait_time)
