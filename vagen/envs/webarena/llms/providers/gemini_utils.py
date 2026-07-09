"""Tools to generate from Gemini prompts."""

import random
import time
from typing import Any

from google.api_core.exceptions import InvalidArgument
from vertexai.preview.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Image,
)

model = GenerativeModel("gemini-pro-vision")


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple[Any] = (InvalidArgument,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def generate_from_gemini_completion(
    prompt: list[str | Image],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> str:
    del engine
    safety_config = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    response = model.generate_content(
        prompt,
        generation_config=dict(
            candidate_count=1,
            max_output_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        ),
        safety_settings=safety_config,
    )
    answer = response.text
    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_gemini_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
