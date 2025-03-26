import requests
import os
import time
import argparse
import json
from tabulate import tabulate

# Default API URL
API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoints"""
    print("\n=== Testing Health Endpoints ===")
    
    # Test main health endpoint
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health Check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to check health: {str(e)}")
    
    # Test Ollama health
    try:
        response = requests.get(f"{API_URL}/health/ollama")
        print(f"\nOllama Health: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to check Ollama health: {str(e)}")
    
    # Test Qdrant health
    try:
        response = requests.get(f"{API_URL}/health/qdrant")
        print(f"\nQdrant Health: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to check Qdrant health: {str(e)}")

def test_models():
    """Test model configuration endpoints"""
    print("\n=== Testing Model Configuration Endpoints ===")
    
    # Test list models endpoint
    try:
        response = requests.get(f"{API_URL}/models")
        print(f"List Models: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            # Create a table of available models
            table_data = []
            for model in models:
                table_data.append([model.get("name"), model.get("description")])
            
            print("\nAvailable Models:")
            print(tabulate(table_data, headers=["Model Name", "Description"]))
            print(f"\nDefault Model: {data.get('default_model')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to list models: {str(e)}")
    
    # Test model config endpoint for default model
    try:
        # Get default model first
        response = requests.get(f"{API_URL}/models")
        if response.status_code == 200:
            default_model = response.json().get("default_model")
            
            # Get config for default model
            response = requests.get(f"{API_URL}/models/{default_model}/config")
            print(f"\nModel Config ({default_model}): {response.status_code}")
            
            if response.status_code == 200:
                config = response.json().get("config", {})
                
                # Create a table of config parameters
                table_data = []
                for key, value in config.items():
                    if key != "description":
                        table_data.append([key, value])
                
                print("\nModel Configuration:")
                print(tabulate(table_data, headers=["Parameter", "Value"]))
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to get model config: {str(e)}")

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
                    print("\nQuery Analysis:")
                    query_analysis = data.get('query_analysis')
                    
                    # Create a table for query analysis
                    table_data = []
                    table_data.append(["Intent", query_analysis.get("intent", "")])
                    table_data.append(["Questions", ", ".join(query_analysis.get("questions", []))])
                    table_data.append(["Search Terms", ", ".join(query_analysis.get("search_terms", []))])
                    table_data.append(["Requires Context", query_analysis.get("requires_context", "")])
                    
                    print(tabulate(table_data))
                
                if data.get('relevance_assessment'):
                    print(f"\nRelevance Assessment: {data.get('relevance_assessment')}")
                    
                if data.get('error'):
                    print(f"\nError: {data.get('error')}")
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
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to upload file: {str(e)}")

def test_with_different_models(query, models=None):
    """Test the same query with different models"""
    print("\n=== Testing Query with Different Models ===")
    
    if models is None:
        # Get available models from API
        try:
            response = requests.get(f"{API_URL}/models")
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])][:3]  # Limit to first 3
            else:
                models = ["qwen2.5", "llama3", "mistral"]  # Fallback
        except:
            models = ["qwen2.5", "llama3", "mistral"]  # Fallback
    
    print(f"Testing query with models: {', '.join(models)}")
    print(f"Query: {query}")
    
    results = []
    
    for model in models:
        try:
            print(f"\nTesting with model: {model}")
            payload = {
                "query": query,
                "model_name": model,
                "use_web_search": True
            }
            
            start_time = time.time()
            response = requests.post(f"{API_URL}/query", json=payload)
            elapsed_time = time.time() - start_time
            
            status = response.status_code
            
            if status == 200:
                data = response.json()
                response_text = data.get('response', '')
                truncated_response = response_text[:100] + "..." if len(response_text) > 100 else response_text
                
                results.append({
                    "model": model,
                    "status": status,
                    "time": elapsed_time,
                    "response": truncated_response,
                    "error": data.get('error', '')
                })
            else:
                results.append({
                    "model": model,
                    "status": status,
                    "time": elapsed_time,
                    "response": "",
                    "error": response.text
                })
        except Exception as e:
            results.append({
                "model": model,
                "status": "Error",
                "time": 0,
                "response": "",
                "error": str(e)
            })
    
    # Display results table
    table_data = []
    for result in results:
        table_data.append([
            result["model"],
            result["status"],
            f"{result['time']:.2f}s",
            result["response"],
            result["error"]
        ])
    
    print("\nModel Comparison Results:")
    print(tabulate(table_data, headers=["Model", "Status", "Time", "Response", "Error"]))

def main():
    global API_URL  # Declare global variable at the beginning of the function
    
    parser = argparse.ArgumentParser(description="Test Agentic RAG API")
    parser.add_argument("--url", default=API_URL, help="API URL")
    parser.add_argument("--query", help="Single query to test")
    parser.add_argument("--model", default="qwen2.5", help="Model to use for queries")
    parser.add_argument("--models", help="Comma-separated list of models to test the query with")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web search")
    parser.add_argument("--file", help="File to upload for testing ingestion")
    parser.add_argument("--skip-health", action="store_true", help="Skip health checks")
    parser.add_argument("--skip-query", action="store_true", help="Skip query tests")
    parser.add_argument("--skip-upload", action="store_true", help="Skip upload tests")
    parser.add_argument("--skip-models", action="store_true", help="Skip model tests")
    parser.add_argument("--compare-models", action="store_true", help="Compare different models on the same query")
    
    args = parser.parse_args()
    
    # Update API_URL if provided in command line
    if args.url != API_URL:
        API_URL = args.url
    
    if not args.skip_health:
        test_health()
    
    if not args.skip_models:
        test_models()
    
    if not args.skip_query:
        queries = [args.query] if args.query else None
        test_query(queries, args.model, not args.no_web_search)
    
    if args.compare_models and args.query:
        models = args.models.split(",") if args.models else None
        test_with_different_models(args.query, models)
    
    if not args.skip_upload and args.file:
        test_upload_file(args.file)

if __name__ == "__main__":
    main() 