import time
import sys

from google import genai
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()

# Initialize the GenAI SDK
# The client automatically picks up the GEMINI_API_KEY environment variable.
# Ensure you have run: export GEMINI_API_KEY="your-api-key"
REQUEST_TIMEOUT_MS = 120000
MAX_ATTEMPTS = 3
client = genai.Client(http_options={"timeout": REQUEST_TIMEOUT_MS})

# Define our tiered models
MANAGER_MODEL = 'gemini-flash-latest'
WORKER_MODEL = 'gemini-3.1-pro-preview'

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130


def token_usage_or_zero(response) -> tuple[int, int, int, int]:
    """Return prompt, output, cached, and total token counts from usage metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return (0, 0, 0, 0)

    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    cached_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    return (prompt_tokens, output_tokens, cached_tokens, total_tokens)


def print_usage_line(label: str, response) -> tuple[int, int, int, int]:
    """Print token usage for a single model call and return parsed counts."""
    prompt_tokens, output_tokens, cached_tokens, total_tokens = token_usage_or_zero(response)
    if response is None:
        print(f"-> Tokens [{label}]: unavailable (request failed)")
        return (0, 0, 0, 0)

    print(
        f"-> Tokens [{label}]: "
        f"prompt={prompt_tokens}, output={output_tokens}, cached={cached_tokens}, total={total_tokens}"
    )
    return (prompt_tokens, output_tokens, cached_tokens, total_tokens)


def should_retry_api_error(exc: errors.APIError) -> bool:
    """Return True for API errors that are likely transient."""
    status_code = getattr(exc, "code", None)
    if status_code == 503:
        return True
    if status_code == 429:
        # Quota limit 0 indicates retries are unlikely to succeed.
        return "limit: 0" not in str(exc)
    return False


def generate_with_retry(model: str, contents: str, max_attempts: int = MAX_ATTEMPTS):
    """Call Gemini with simple retry/backoff for transient overload errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except errors.ServerError as exc:
            if attempt == max_attempts:
                print(f"-> ERROR: {model} failed after {max_attempts} attempts: {exc}")
                return None
            delay_seconds = 2 ** (attempt - 1)
            print(
                f"-> WARN: {model} temporarily unavailable "
                f"(attempt {attempt}/{max_attempts}). Retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)
        except errors.APIError as exc:
            status_code = getattr(exc, "code", None)
            should_retry = should_retry_api_error(exc)

            if attempt == max_attempts or not should_retry:
                print(f"-> ERROR: {model} request failed: {exc}")
                return None

            delay_seconds = 2 ** (attempt - 1)
            print(
                f"-> WARN: {model} API throttled/unavailable "
                f"(status {status_code}, attempt {attempt}/{max_attempts}). "
                f"Retrying in {delay_seconds}s..."
            )
            time.sleep(delay_seconds)


def response_text_or_empty(response) -> str:
    """Return model text content safely even when the SDK returns None."""
    if response is None or response.text is None:
        return ""
    return response.text.strip()

def run_routing_demo():
    print("\n--- Manager-Worker Tiered Routing Demo ---")
    print(f"Manager Model: {MANAGER_MODEL} (Fast, Low Cost)")
    print(f"Worker Model:  {WORKER_MODEL} (High Reasoning, Higher Cost)\n")
    print(f"Request timeout: {REQUEST_TIMEOUT_MS}ms | Max attempts: {MAX_ATTEMPTS}\n")
    
    # We define two tasks: one obviously simple, one obviously complex
    tasks = [
        "Sort this list of names alphabetically: Alice, Charlie, Bob.",
        "Explain the architectural differences between monolithic and microservice architectures, specifically focusing on data consistency and transaction management."
    ]
    
    # The prompt that instructs the Manager how to triage incoming requests
    router_prompt = (
        "You are a task routing manager. Evaluate the following task and determine its complexity. "
        "If it is a basic formatting, sorting, or simple factual task, reply with exactly the word 'SIMPLE'. "
        "If it involves complex logic, system architecture, or heavy reasoning, reply with exactly the word 'COMPLEX'.\n\n"
        "Task: {task}"
    )
    
    for i, task in enumerate(tasks):
        print(f"==================================================")
        print(f"Scenario {i+1}: {task}")
        print(f"==================================================")
        
        # 1. The Manager Evaluates the Task
        print(f"-> Manager ({MANAGER_MODEL}) is evaluating complexity...")
        route_response = generate_with_retry(
            model=MANAGER_MODEL,
            contents=router_prompt.format(task=task),
        )
        route_prompt, route_output, route_cached, route_total = print_usage_line(
            "Manager Route Check",
            route_response,
        )
        
        # Clean up the output to ensure we just get the keyword
        decision_text = response_text_or_empty(route_response)
        if decision_text:
            decision = decision_text.upper()
        else:
            decision = "COMPLEX"
            print("-> WARN: No manager decision received. Falling back to COMPLEX routing.")
        print(f"-> Manager Decision: {decision}")
        
        # 2. The Routing Logic executes based on the decision
        if "COMPLEX" in decision:
            print(f"-> ACTION: Escalating task to Worker ({WORKER_MODEL})...")
            response = generate_with_retry(model=WORKER_MODEL, contents=task)
            exec_prompt, exec_output, exec_cached, exec_total = print_usage_line(
                "Worker Execution",
                response,
            )
            
            full_response = response_text_or_empty(response)
            if full_response:
                print(f"\n[Worker Output]: {full_response}\n")
            else:
                print("\n[Worker Output]: <no response text>\n")
        else:
            print(f"-> ACTION: Resolving task directly with Manager ({MANAGER_MODEL})...")
            response = generate_with_retry(model=MANAGER_MODEL, contents=task)
            exec_prompt, exec_output, exec_cached, exec_total = print_usage_line(
                "Manager Execution",
                response,
            )
            
            full_response = response_text_or_empty(response)
            if full_response:
                print(f"\n[Manager Output]: {full_response}\n")
            else:
                print("\n[Manager Output]: <no response text>\n")

        print("-> Scenario Token Summary:")
        print(
            "   "
            f"prompt={route_prompt + exec_prompt}, "
            f"output={route_output + exec_output}, "
            f"cached={route_cached + exec_cached}, "
            f"total={route_total + exec_total}"
        )
            
    print("Takeaway: Filter and route with a fast, cheap model first to save your budget!")


def main() -> int:
    """Run the routing demo with top-level interrupt and error handling."""
    try:
        run_routing_demo()
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:  # pragma: no cover - script-level guard
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

if __name__ == "__main__":
    sys.exit(main())