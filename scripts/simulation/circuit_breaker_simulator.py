import os
import json

from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

# Initialize the client. It will use GEMINI_API_KEY from the environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file or environment.")

client = genai.Client(api_key=GEMINI_API_KEY)


def print_request_preview(prompt: str, model_id: str, total_tokens: int, token_limit: int) -> None:
    """Print a safe preview of the outgoing generateContent HTTP request."""
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    max_preview_chars = 400
    prompt_preview = prompt
    if len(prompt) > max_preview_chars:
        prompt_preview = (
            f"{prompt[:max_preview_chars]}... "
            f"[truncated, original_length={len(prompt)} chars]"
        )

    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_preview}],
            }
        ]
    }

    print("\n--- REQUEST PREVIEW (What will be sent) ---")
    print("Method: POST")
    print(f"Endpoint: {endpoint}")
    print(f"Input Tokens (estimated): {total_tokens}")
    print(f"Token Limit: {token_limit}")
    print("Body:")
    print(json.dumps(request_payload, indent=2))
    print("--- END REQUEST PREVIEW ---")


def print_actual_usage(response: object) -> None:
    """Print exact token usage returned by the Gemini API response metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        print("Actual Usage: unavailable (no usage_metadata in response).")
        return

    prompt_tokens = getattr(usage, "prompt_token_count", 0)
    output_tokens = getattr(usage, "candidates_token_count", 0)
    total_tokens = getattr(usage, "total_token_count", 0)
    
    # Check for the hidden tokens
    cached_tokens = getattr(usage, "cached_content_token_count", 0)

    print("\n--- ACTUAL API TOKEN USAGE ---")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Cached Tokens: {cached_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print(f"(Math Check: {prompt_tokens} + {output_tokens} + {cached_tokens} = {prompt_tokens + output_tokens + cached_tokens})")
    print("--- END ACTUAL API TOKEN USAGE ---")


def find_fallback_model(*, exclude: str | None = None) -> str | None:
    """Pick a likely available Gemini model from the account's model list."""
    preferred = [
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    try:
        available_model_names: list[str] = []
        for model in client.models.list():
            name = getattr(model, "name", "")
            if isinstance(name, str) and name:
                available_model_names.append(name.removeprefix("models/"))

        for candidate in preferred:
            if candidate == exclude:
                continue
            if candidate in available_model_names:
                return candidate

        for name in available_model_names:
            if name == exclude:
                continue
            if "flash" in name:
                return name

        if not available_model_names:
            return None

        if available_model_names[0] == exclude and len(available_model_names) > 1:
            return available_model_names[1]

        return available_model_names[0]
    except Exception:
        return None

def safe_agent_query(prompt: str, token_limit: int = 500) -> str:
    """
    Acts as a simple agent wrapper that implements a pre-flight circuit breaker.
    """
    print("\n--- PHASE 1: The Pre-Flight Circuit Breaker ---")
    
    try:
        # 1. Count tokens (Free, local/pre-flight operation)
        token_info = client.models.count_tokens(
            model=MODEL_ID, 
            contents=prompt
        )
        total_tokens = token_info.total_tokens or 0
        print(f"Pre-flight check: This call will cost {total_tokens} input tokens.")
        print_request_preview(prompt, MODEL_ID, total_tokens, token_limit)
        
        # 2. Circuit Breaker Logic
        if total_tokens > token_limit:
            print("ACTION: Token budget exceeded! Tripping orchestration circuit breaker to protect resources.")
            return "ERROR: Circuit breaker tripped. Prompt too large."
            
        print("ACTION: Token count is within budget. Proceeding to Generation...")
        
        # 3. Safe Generation Call
        print("\n--- PHASE 2: Generating Response ---")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        print_actual_usage(response)
        
        return response.text or ""

    except errors.APIError as e:
        print(f"API Error encountered: {e}")
        error_text = str(e)

        if any(code in error_text for code in ("NOT_FOUND", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
            fallback_model = find_fallback_model(exclude=MODEL_ID)
            if fallback_model and fallback_model != MODEL_ID:
                print(f"ACTION: Retrying with discovered fallback model '{fallback_model}'.")
                try:
                    response = client.models.generate_content(
                        model=fallback_model,
                        contents=prompt,
                    )
                    print_actual_usage(response)
                    return response.text or ""
                except errors.APIError as retry_error:
                    print(f"Fallback model API error: {retry_error}")

            print(
                "Hint: The configured model may be unavailable or temporarily throttled. "
                "Set GEMINI_MODEL in .env to a currently supported model."
            )

        return "ERROR: API failed."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "ERROR: Execution failed."

# ==========================================
# Testing the Agent
# ==========================================
if __name__ == "__main__":
    print("Initializing Circuit Breaker Agent Test...\n")
    
    # Test 1: A safe, normal-sized prompt
    print(">>> TEST 1: Normal Prompt")
    safe_prompt = "Explain the concept of a circuit breaker in software engineering in two sentences."
    result = safe_agent_query(safe_prompt, token_limit=500)
    print(f"Agent Response:\n{result}\n")
    
    print("-" * 50)
    
    # Test 2: Simulating the runaway loop / massive context
    print(">>> TEST 2: Runaway Prompt (Triggering the Breaker)")
    # We multiply a string to artificially inflate the token count well beyond 500
    runaway_prompt = "Simulated runaway reasoning loop data... " * 100 
    result = safe_agent_query(runaway_prompt, token_limit=500)
    print(f"Agent Response:\n{result}\n")