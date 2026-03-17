# All comments are in English.
# Importing these modules triggers @register_client / @register_adapter decorators.

import vagen.evaluate.clients  

import vagen.evaluate.adapters.openai_adapter  # noqa: F401  (registers: openai, azure)
import vagen.evaluate.adapters.openai_responses_adapter  # noqa: F401  (registers: openai_responses, azure_responses)
import vagen.evaluate.adapters.sglang_adapter  # noqa: F401
import vagen.evaluate.adapters.vllm_adapter  # noqa: F401
import vagen.evaluate.adapters.together_adapter  # noqa: F401
import vagen.evaluate.adapters.claude_adapter  # noqa: F401
import vagen.evaluate.adapters.gemini_adapter  # noqa: F401
