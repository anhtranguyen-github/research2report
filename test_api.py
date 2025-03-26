import requests
import os
import time
import argparse

# Default API URL
API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoints"""
    print("\n=== Testing Health Endpoints ===")
    
    # Test main health endpoint
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health Check: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Failed to check health: {str(e)}")
    
    # Test Ollama health
    try:
        response = requests.get(f"{API_URL}/health/ollama")
        print(f"\nOllama Health: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Failed to check Ollama health: {str(e)}")
    
    # Test Qdrant health
    try:
        response = requests.get(f"{API_URL}/health/qdrant")
        print(f"\nQdrant Health: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Failed to check Qdrant health: {str(e)}")

def test_query(queries=None, model="qwen2.5", use_web_search=True):
    """Test query endpoint with optional web search"""
    print("\n=== Testing Query Endpoint ===")
    
    if queries is None:
        queries = [
            "What are the key components of a RAG system?",
            "How does LangGraph work with Ollama?",
            "What is the difference between RAG and traditional search?"
        ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        payload = {
            "query": query,
            "model_name": model,
            "use_web_search": use_web_search
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/query", json=payload)
            elapsed_time = time.time() - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"Time Taken: {elapsed_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response: {data.get('response')[:200]}...")  # Print first 200 chars
                
                if data.get('query_analysis'):
                    print(f"Query Analysis: {data.get('query_analysis')}")
                
                if data.get('relevance_assessment'):
                    print(f"Relevance Assessment: {data.get('relevance_assessment')}")
                    
                if data.get('error'):
                    print(f"Error: {data.get('error')}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Failed to send query: {str(e)}")

def test_upload_file(file_path):
    """Test file upload endpoint"""
    print("\n=== Testing File Upload ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Uploading file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            data = {
                'chunk_size': '1000',
                'chunk_overlap': '200'
            }
            
            response = requests.post(f"{API_URL}/upload", files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"Failed to upload file: {str(e)}")

def main():
    global API_URL  # Move global declaration to the beginning of the function
    
    parser = argparse.ArgumentParser(description="Test Agentic RAG API")
    parser.add_argument("--url", default=API_URL, help="API URL")
    parser.add_argument("--query", help="Single query to test")
    parser.add_argument("--model", default="qwen2.5", help="Model to use for queries")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search")
    parser.add_argument("--file", help="File to upload for testing ingestion")
    parser.add_argument("--skip-health", action="store_true", help="Skip health checks")
    parser.add_argument("--skip-query", action="store_true", help="Skip query tests")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload tests")
    
    args = parser.parse_args()
    
    # Update API_URL if provided in command line
    if args.url != API_URL:
        API_URL = args.url  # No need for global declaration here
    
    if not args.skip_health:
        test_health()
    
    if not args.skip_query:
        queries = [args.query] if args.query else None
        test_query(queries, args.model, not args.no_web_search)
    
    if not args.skip_upload and args.file:
        test_upload_file(args.file)

if __name__ == "__main__":
    main() 