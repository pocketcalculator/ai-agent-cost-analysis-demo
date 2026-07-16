import os
from google import genai

# Ensure your GEMINI_API_KEY is set in your environment
client = genai.Client()

print("Models available to your API key:\n" + "-"*30)

# Iterate through the models and print ones that support generation
for model_info in client.models.list():
    # We only care about models that support the generateContent method
    if "generateContent" in model_info.supported_generation_methods:
        print(f"- {model_info.name}")