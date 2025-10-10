import requests
import json

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"  # or "llama2", "neural-chat", etc.

def summarize_text(text, max_length=150):
    """
    Summarize text using Ollama and a local OSS model.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of summary in words (approximate)
    
    Returns:
        The summarized text
    """
    
    # Create a prompt for summarization
    prompt = f"""Please summarize the following text in approximately {max_length} words. 
    Provide only the summary without any additional explanation.

Text to summarize:
{text}

Summary:"""
    
    try:
        # Make request to Ollama
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        summary = result.get("response", "").strip()
        
        return summary
    
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure it's running on localhost:11434"
    except Exception as e:
        return f"Error: {str(e)}"


def summarize_with_streaming(text, max_length=150):
    """
    Summarize text with streaming output (real-time token generation).
    """
    
    prompt = f"""Please summarize the following text in approximately {max_length} words:

Text:
{text}

Summary:"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": True,
                "temperature": 0.7,
            },
            stream=True,
            timeout=120
        )
        
        response.raise_for_status()
        
        print("Summary (streaming): ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
        
        print()  # New line after streaming completes
    
    except Exception as e:
        print(f"Error: {str(e)}")


# Example usage
if __name__ == "__main__":
    sample_text = """
    Artificial intelligence has revolutionized multiple industries over the past decade.
    From healthcare to finance, AI systems are now being deployed to solve complex problems,
    make predictions, and automate routine tasks. Machine learning models can now analyze
    medical images with accuracy comparable to human radiologists. In finance, AI-powered
    algorithms trade billions of dollars daily. However, concerns about bias, privacy, and
    job displacement continue to shape the AI policy landscape. Researchers are working on
    making AI systems more transparent and interpretable to build public trust.
    """
    
    print("=== Text Summarization with Ollama ===\n")
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Method 1: Regular summarization
    print("Method 1: Regular Summarization")
    summary = summarize_text(sample_text, max_length=80)
    print(f"Summary:\n{summary}\n")
    
    print("="*50 + "\n")
    
    # Method 2: Streaming summarization
    print("Method 2: Streaming Summarization")
    summarize_with_streaming(sample_text, max_length=80)