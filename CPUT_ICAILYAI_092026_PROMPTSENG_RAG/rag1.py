import os
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# 1. Load the environment variables from the .env file
load_dotenv(find_dotenv(usecwd=True))   # read LLM_PROVIDER and your API key from the .env file   load_dotenv()

# 2. Retrieve the API key securely from your environment
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY. Please check your .env file.")

# 3. Initialize the OpenAI client with OpenRouter's base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# 4. Make a request using a Groq-hosted model
# Note: OpenRouter uses the format 'provider/model-name' or 'meta-llama/...'
# You can check exact slugs at openrouter.ai/models
completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "https://your-site-url.com", # Optional, for OpenRouter rankings
        "X-Title": "My Python App",                 # Optional, for OpenRouter rankings
    },
    model="meta-llama/llama-3-70b-instruct", # OpenRouter automatically routes to top providers like Groq
    messages=[
        {
            "role": "user",
            "content": "Explain quantum computing in one short sentence."
        }
    ]
)

# 5. Print the model's response
print(completion.choices[0].message.content)



