# This script provides a wrapper for LLM calls using environment variables.
# It is compatible with Gemini and other providers as required by Adobe's evaluation.
import os
import sys
import json
from typing import List

def chat_with_llm(messages: List[dict], model: str = None) -> str:
    """
    Call the LLM using the environment variables and return the response.
    Compatible with Gemini and other providers.
    """
    # Use the sample script logic from https://github.com/rbabbar-adobe/sample-repo/blob/main/chat_with_llm.py
    # This is a stub; real implementation should use the provider specified in env vars.
    # For Gemini, use google-generativeai; for others, use their respective SDKs.
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'gemini')
    GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
    if LLM_PROVIDER == 'gemini':
        import google.generativeai as genai
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY not set')
        genai.configure(api_key=api_key)
        model = model or GEMINI_MODEL
        model_gemini = genai.GenerativeModel(model)
        prompt = messages[-1]['content'] if messages else ''
        response = model_gemini.generate_content(prompt)
        return response.text
    else:
        raise NotImplementedError('Only Gemini LLM is supported in this sample.')

if __name__ == "__main__":
    # Example usage: python chat_with_llm.py '{"messages": [{"role": "user", "content": "Hello!"}]}'
    if len(sys.argv) < 2:
        print("Usage: python chat_with_llm.py '<json-args>'")
        sys.exit(1)
    args = json.loads(sys.argv[1])
    messages = args.get('messages', [])
    model = args.get('model')
    print(chat_with_llm(messages, model))
