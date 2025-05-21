import asyncio
import random
import time
from typing import List, Dict, Any
from openai import AsyncOpenAI

class RateLimiter:
    """Rate limiter for OpenAI GPT API"""
    def __init__(self, rpm_limit=10000, tpm_limit=2000000):
        # OpenAI API limits (adjust based on your tier)
        # Tier 1: 500 RPM, 10,000 TPM
        # Tier 2: 5,000 RPM, 450,000 TPM  
        # Tier 3: 10,000 RPM, 2,000,000 TPM
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_timestamps = []
        self.token_counts = []
        # Conservative concurrent request limit
        self.semaphore = asyncio.Semaphore(50)
    
    async def wait_if_needed(self, estimated_tokens=1000):
        now = time.time()
        # Clean up old timestamps (older than 60 seconds)
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]
        
        # Check RPM limit
        rpm_current = len(self.request_timestamps)
        if rpm_current >= self.rpm_limit:
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.wait_if_needed(estimated_tokens)
        
        # Check TPM limit
        total_tokens = sum(self.token_counts)
        if total_tokens + estimated_tokens >= self.tpm_limit:
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.wait_if_needed(estimated_tokens)
        
        # Update tracking
        self.request_timestamps.append(now)
        self.token_counts.append(estimated_tokens)

def run_gpt_request(prompts: List[str], config) -> List[Dict[str, Any]]:
    """
    Process prompts with OpenAI GPT API, handling rate limits.
    
    Args:
        prompts: List of prompt strings to process
        config: Config object that supports config.get() method
    
    Returns:
        List of dictionaries with results for each prompt
    """
    # Process in batches if needed
    batch_size = config.get("batch_size", 10)
    if len(prompts) <= batch_size:
        return _process_batch(prompts, config)
    
    # For larger sets, process in batches
    all_results = []
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    
    for i, batch in enumerate(batches):
        if len(batches) > 1:
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} prompts)")
        batch_results = _process_batch(batch, config)
        all_results.extend(batch_results)
        if i < len(batches) - 1:
            time.sleep(1)  # Slightly longer delay between batches
    
    return all_results

def _process_batch(prompts: List[str], config) -> List[Dict[str, Any]]:
    """Process a single batch with rate limiting"""
    async def _async_batch_completions():
        async_client = AsyncOpenAI(
            api_key=config.get("api_key") or None  # Uses OPENAI_API_KEY env var if None
        )
        rate_limiter = RateLimiter(
            rpm_limit=config.get("rpm_limit", 70),
            tpm_limit=config.get("tpm_limit", 4000)
        )
        
        results = [{"response": "", "success": False, "retries": 0, "error": None} for _ in prompts]
        
        async def process_prompt(prompt: str, index: int) -> None:
            retries = 0
            # Estimate tokens (1 token ≈ 4 chars for English text)
            estimated_prompt_tokens = len(prompt) // 4
            estimated_completion_tokens = config.get("max_tokens", 1500)
            total_estimated_tokens = estimated_prompt_tokens + estimated_completion_tokens
            
            while retries <= config.get("max_retries", 3):
                try:
                    async with rate_limiter.semaphore:
                        await rate_limiter.wait_if_needed(total_estimated_tokens)
                        
                        # Use chat completions for GPT models
                        response = await async_client.chat.completions.create(
                            model=config.get("model", "gpt-4.1-nano"),
                            messages=[
                                {"role": "system", "content": config.get("system_message", "You are a helpful assistant.")},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=config.get("temperature", 0.1),
                            max_tokens=estimated_completion_tokens,
                            timeout=config.get("request_timeout", 120)
                        )
                        
                        results[index] = {
                            "response": response.choices[0].message.content,
                            "success": True,
                            "retries": retries,
                            "error": None,
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        }
                        return
                        
                except Exception as e:
                    error_str = str(e)
                    retries += 1
                    
                    # Handle different types of errors
                    if "rate_limit" in error_str.lower() or "rate limit" in error_str.lower():
                        # Exponential backoff for rate limit errors
                        backoff_time = config.get("retry_delay", 2) * (2 ** (retries - 1))
                        backoff_time += random.uniform(0, 1)  # Add jitter
                        backoff_time = min(backoff_time, 60)  # Cap at 60s
                        print(f"Rate limit hit, waiting {backoff_time:.1f}s before retry {retries}")
                        await asyncio.sleep(backoff_time)
                    elif "timeout" in error_str.lower():
                        print(f"Timeout error on attempt {retries}, retrying...")
                        await asyncio.sleep(config.get("retry_delay", 2))
                    elif retries <= config.get("max_retries", 3):
                        await asyncio.sleep(config.get("retry_delay", 2))
                    else:
                        results[index] = {
                            "response": f"Error after {retries} attempts: {error_str}",
                            "success": False,
                            "retries": retries,
                            "error": error_str,
                            "usage": None
                        }
                        return
        
        tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=config.get("batch_timeout", 300)  # 5 minutes for the entire batch
            )
        except asyncio.TimeoutError:
            print("Batch processing timed out")
        
        return results
    
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(_async_batch_completions())
                loop.close()
            else:
                results = loop.run_until_complete(_async_batch_completions())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_async_batch_completions())
            loop.close()
    except Exception as e:
        print(f"Global error: {str(e)}")
        return [{"response": f"Global error: {str(e)}", "success": False, "retries": 0, "error": str(e), "usage": None} 
                for _ in prompts]
    
    return results

# Example usage
if __name__ == "__main__":
    # Configuration example
    config = {
        "model": "gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
        "api_key": None,  # Will use OPENAI_API_KEY environment variable
        "system_message": "You are a helpful assistant.",
        "temperature": 0.1,
        "max_tokens": 500,
        "max_retries": 3,
        "retry_delay": 2,
        "batch_size": 10,
        "rpm_limit": 70,  # Requests per minute
        "tpm_limit": 4000,  # Tokens per minute
        "request_timeout": 120,  # Individual request timeout
        "batch_timeout": 300  # Batch timeout
    }
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI."
    ]
    
    results = run_gpt_request(test_prompts, config)
    
    for i, result in enumerate(results):
        print(f"Prompt {i+1}: {'✓' if result['success'] else '✗'}")
        if result['success']:
            print(f"Response: {result['response'][:100]}...")
            if result.get('usage'):
                print(f"Tokens used: {result['usage']['total_tokens']}")
        else:
            print(f"Error: {result['error']}")
        print("-" * 50)