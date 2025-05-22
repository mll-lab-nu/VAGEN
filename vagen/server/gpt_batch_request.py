import asyncio
import random
import time
from typing import List, Dict, Any
from openai import AsyncOpenAI

class RateLimiter:
    """Rate limiter for OpenAI GPT API"""
    def __init__(self, qps_limit=70, rpm_limit=4000, tps_limit=15000):
        self.qps_limit = qps_limit
        self.rpm_limit = rpm_limit
        self.tps_limit = tps_limit
        self.request_timestamps = []
        self.token_counts = []
        self.semaphore = asyncio.Semaphore(qps_limit)
    
    async def wait_if_needed(self, estimated_tokens=500):
        now = time.time()
        # Clean up old timestamps (older than 60 seconds)
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]
        
        # Check RPM limit
        rpm_current = len(self.request_timestamps)
        if rpm_current >= self.rpm_limit:
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.wait_if_needed(estimated_tokens)
        
        # Check QPS limit (last 1 second)
        recent_requests = sum(1 for ts in self.request_timestamps if now - ts < 1)
        if recent_requests >= self.qps_limit:
            await asyncio.sleep(0.1)
            return await self.wait_if_needed(estimated_tokens)
        
        # Check TPS limit (last 1 second)
        recent_tokens = sum(tokens for ts, tokens in zip(self.request_timestamps, self.token_counts) if now - ts < 1)
        if recent_tokens + estimated_tokens >= self.tps_limit:
            await asyncio.sleep(0.2)
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
    batch_size = config.get("batch_size", 20)
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
            time.sleep(0.5)
    
    return all_results

def _process_batch(prompts: List[str], config) -> List[Dict[str, Any]]:
    """Process a single batch with rate limiting"""
    async def _async_batch_completions():
        async_client = AsyncOpenAI()
        rate_limiter = RateLimiter(
            qps_limit=config.get("qps_limit", 70),
            rpm_limit=config.get("rpm_limit", 4000),
            tps_limit=config.get("tps_limit", 15000)
        )
        
        results = [{"response": "", "success": False, "retries": 0, "error": None} for _ in prompts]
        
        async def process_prompt(prompt: str, index: int) -> None:
            retries = 0
            # Estimate tokens (1 token ≈ 4 chars)
            estimated_prompt_tokens = len(prompt) // 4
            estimated_completion_tokens = config.get("max_tokens", 500)
            total_estimated_tokens = estimated_prompt_tokens + estimated_completion_tokens
            
            while retries <= config.get("max_retries", 3):
                try:
                    async with rate_limiter.semaphore:
                        await rate_limiter.wait_if_needed(total_estimated_tokens)
                        
                        response = await async_client.chat.completions.create(
                            model=config.get("name", "gpt-4.1-nano"),
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=config.get("temperature", 0.1),
                            max_tokens=estimated_completion_tokens
                        )
                        
                        results[index] = {
                            "response": response.choices[0].message.content,
                            "success": True,
                            "retries": retries,
                            "error": None
                        }
                        return
                except Exception as e:
                    error_str = str(e)
                    retries += 1
                    
                    # Exponential backoff for rate limit errors
                    if "rate_limit" in error_str.lower():
                        backoff_time = config.get("retry_delay", 1) * (2 ** (retries - 1))
                        backoff_time += random.uniform(0, 1)  # Add jitter
                        backoff_time = min(backoff_time, 30)  # Cap at 30s
                        await asyncio.sleep(backoff_time)
                    elif retries <= config.get("max_retries", 3):
                        await asyncio.sleep(config.get("retry_delay", 1))
                    else:
                        results[index] = {
                            "response": f"Error after {retries} attempts",
                            "success": False,
                            "retries": retries,
                            "error": error_str
                        }
                        return
        
        tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=config.get("request_timeout", 120)
            )
        except asyncio.TimeoutError:
            pass
        
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
        return [{"response": f"Global error: {str(e)}", "success": False, "retries": 0, "error": str(e)} 
                for _ in prompts]
    
    return results