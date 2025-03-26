import requests
import os
import time
import argparse
import json
from tabulate import tabulate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

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

def test_embeddings():
    """Test embedding model configuration endpoints"""
    print("\n=== Testing Embedding Model Configuration Endpoints ===")
    
    # Test list embeddings endpoint
    try:
        response = requests.get(f"{API_URL}/embeddings")
        print(f"List Embedding Models: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            # Create a table of available embedding models
            table_data = []
            for model in models:
                table_data.append([
                    model.get("name"), 
                    model.get("dimensions"),
                    model.get("type"),
                    model.get("description")
                ])
            
            print("\nAvailable Embedding Models:")
            print(tabulate(table_data, headers=["Model Name", "Dimensions", "Type", "Description"]))
            print(f"\nDefault Model: {data.get('default_model')}")
            print(f"Current Model: {data.get('current_model')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to list embedding models: {str(e)}")
    
    # Test embedding model config endpoint for default model
    try:
        # Get default model first
        response = requests.get(f"{API_URL}/embeddings")
        if response.status_code == 200:
            default_model = response.json().get("default_model")
            
            # Get config for default model
            response = requests.get(f"{API_URL}/embeddings/{default_model}/config")
            print(f"\nEmbedding Config ({default_model}): {response.status_code}")
            
            if response.status_code == 200:
                config = response.json().get("config", {})
                
                # Create a table of config parameters
                table_data = []
                for key, value in config.items():
                    if key != "description":
                        table_data.append([key, value])
                
                print("\nEmbedding Model Configuration:")
                print(tabulate(table_data, headers=["Parameter", "Value"]))
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to get embedding config: {str(e)}")

def test_collections():
    """Test Qdrant collections endpoints"""
    print("\n=== Testing Collections Endpoints ===")
    
    # List collections
    try:
        response = requests.get(f"{API_URL}/collections")
        print(f"List Collections: {response.status_code}")
        
        if response.status_code == 200:
            collections = response.json()
            
            if collections:
                # Create a table of collections
                table_data = []
                for collection in collections:
                    table_data.append([
                        collection.get("name"),
                        collection.get("vectors_count", "N/A"),
                        collection.get("vector_size", "N/A"),
                        collection.get("distance", "N/A")
                    ])
                
                print("\nAvailable Collections:")
                print(tabulate(table_data, headers=["Name", "Vectors Count", "Vector Size", "Distance"]))
            else:
                print("No collections found.")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to list collections: {str(e)}")

def test_change_embedding(embedding_model=None, create_new_collection=False):
    """Test changing the active embedding model"""
    if not embedding_model:
        print("No embedding model specified for change test, skipping.")
        return
    
    print(f"\n=== Testing Change Embedding Model to {embedding_model} ===")
    
    # Get current embedding first
    try:
        response = requests.get(f"{API_URL}/embeddings")
        current_model = "unknown"
        if response.status_code == 200:
            current_model = response.json().get("current_model")
            print(f"Current model before change: {current_model}")
        
        # Skip if already using the requested model
        if current_model == embedding_model:
            print(f"Already using {embedding_model}, skipping change test.")
            return
        
        # Change embedding model
        collection_name = f"test_{embedding_model}" if create_new_collection else None
        
        payload = {
            "model_name": embedding_model,
            "create_new_collection": create_new_collection,
            "collection_name": collection_name
        }
        
        print(f"Requesting change to {embedding_model}" + 
              (f" with new collection {collection_name}" if create_new_collection else ""))
        
        response = requests.post(f"{API_URL}/embeddings/change", json=payload)
        print(f"Change Embedding: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
            
            # Verify the change
            verify_response = requests.get(f"{API_URL}/embeddings")
            if verify_response.status_code == 200:
                new_current = verify_response.json().get("current_model")
                print(f"Verified current model after change: {new_current}")
                
                if new_current == embedding_model:
                    print("✅ Embedding model successfully changed")
                else:
                    print("❌ Embedding model change failed")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to change embedding model: {str(e)}")

def test_query(queries=None, model="qwen2.5", use_web_search=True):
    """Test query endpoint with optional web search"""
    print("\n=== Testing Query Endpoint ===")
    
    if queries is None:
        queries = [
            "What is Vibe Coding",
            "What is MCP",
            "What is Cursor AI"
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
                    print(f"Query Analysis: {json.dumps(data.get('query_analysis'), indent=2)}")
                
                if data.get('retrieval_result'):
                    print(f"Retrieval Results: {len(data.get('retrieval_result'))} documents found")
                    
                if data.get('web_search_results'):
                    print(f"Web Search Results: {len(data.get('web_search_results'))} results found")
                    
                if data.get('error'):
                    print(f"Error: {data.get('error')}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Failed to send query: {str(e)}")

def test_query_analysis(query="What is RAG in artificial intelligence?", model="qwen2.5"):
    """Test the query analysis component specifically"""
    print(f"\n=== Testing Query Analysis for: '{query}' ===")
    
    payload = {
        "query": query,
        "model_name": model,
        "use_web_search": True
    }
    
    try:
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            analysis = data.get('query_analysis')
            
            if analysis:
                print("Query Analysis Results:")
                print(f"Intent: {analysis.get('intent', 'N/A')}")
                print(f"Requires Retrieval: {analysis.get('requires_retrieval', False)}")
                print(f"Requires Web Search: {analysis.get('requires_web_search', False)}")
                
                if 'specific_questions' in analysis and analysis['specific_questions']:
                    print("\nSpecific Questions:")
                    for i, question in enumerate(analysis['specific_questions'], 1):
                        print(f"  {i}. {question}")
                
                if 'context_requirements' in analysis and analysis['context_requirements']:
                    print("\nContext Requirements:")
                    for key, value in analysis['context_requirements'].items():
                        print(f"  {key}: {value}")
                
                print(f"\nFull Analysis: {json.dumps(analysis, indent=2)}")
            else:
                print("No query analysis returned in the response")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed to test query analysis: {str(e)}")

def test_upload_file(file_path, embedding_model=None):
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
            
            # Add embedding model if specified
            if embedding_model:
                data['embedding_model'] = embedding_model
                print(f"Using embedding model: {embedding_model}")
            
            response = requests.post(f"{API_URL}/upload", files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to upload file: {str(e)}")

def test_with_different_models(query, models=None):
    """Test the same query with different models for comparison"""
    if not models:
        # Try to get available models from API
        try:
            response = requests.get(f"{API_URL}/models")
            if response.status_code == 200:
                models_data = response.json().get("models", [])
                models = [model.get("name") for model in models_data]
            else:
                # Default models if can't fetch from API
                models = ["qwen2.5", "phi3"]
        except:
            # Default models if API fails
            models = ["qwen2.5", "phi3"]
    
    print(f"\n=== Testing With Different Models: '{query}' ===")
    
    results = []
    
    for model in models:
        print(f"\nTesting with model: {model}")
        payload = {
            "query": query,
            "model_name": model,
            "use_web_search": True
        }
        
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/query", json=payload)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").strip()
                
                # Get the first 50 characters for the table
                short_response = response_text[:50] + "..." if len(response_text) > 50 else response_text
                
                results.append({
                    "model": model,
                    "time": elapsed_time,
                    "response": short_response,
                    "full_response": response_text
                })
                
                print(f"Status: ✅ ({elapsed_time:.2f}s)")
            else:
                print(f"Status: ❌ Error: {response.text}")
                results.append({
                    "model": model,
                    "time": elapsed_time,
                    "response": f"Error: {response.status_code}",
                    "full_response": response.text
                })
        except Exception as e:
            print(f"Failed: {str(e)}")
            results.append({
                "model": model,
                "time": 0,
                "response": f"Failed: {str(e)}",
                "full_response": str(e)
            })
    
    # Print comparison table
    if results:
        print("\nComparison of Model Responses:")
        table_data = [
            [r["model"], f"{r['time']:.2f}s", r["response"]]
            for r in results
        ]
        print(tabulate(table_data, headers=["Model", "Time", "Response"]))
        
        # Print full responses
        for r in results:
            print(f"\n--- Full Response from {r['model']} ---")
            print(r["full_response"])
    
    return results

def test_web_search():
    """Test if web search is working through the API."""
    print("\nTesting web search functionality...")
    
    # Test query with web search enabled
    query_data = {
        "query": "What is artificial intelligence?",
        "model_name": "qwen2.5",
        "use_web_search": True
    }
    
    try:
        # Make the API request
        print(f"Sending request to {API_URL}/query")
        response = requests.post(
            f"{API_URL}/query",
            json=query_data
        )
        response.raise_for_status()
        result = response.json()
        
        # Check if web search results are present
        if "web_search_results" in result:
            web_results = result["web_search_results"]
            print(f"\nFound {len(web_results)} web search results")
            
            # Print web search results
            for i, doc in enumerate(web_results, 1):
                print(f"\nWeb Search Result {i}:")
                print(f"Title: {doc.get('title', 'N/A')}")
                print(f"URL: {doc.get('url', 'N/A')}")
                print(f"Snippet: {doc.get('snippet', '')[:200]}...")  # Print first 200 chars
                print(f"Score: {doc.get('relevance_score', 'N/A')}")
        else:
            print("\nNo web search results found in response")
        
        # Print the final response
        print("\nFinal Response:")
        print(result.get("response", "No response generated"))
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        return False
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

def test_web_search_disabled():
    """Test query with web search disabled."""
    print("\nTesting query with web search disabled...")
    
    query_data = {
        "query": "What is artificial intelligence?",
        "model_name": "qwen2.5",
        "use_web_search": False
    }
    
    try:
        response = requests.post(
            f"{API_URL}/query",
            json=query_data
        )
        response.raise_for_status()
        result = response.json()
        
        # Verify no web search results
        if "web_search_results" in result and result["web_search_results"]:
            print("Warning: Web search results found when web search was disabled")
        else:
            print("Success: No web search results when web search was disabled")
        
        print("\nResponse without web search:")
        print(result.get("response", "No response generated"))
        
        return True
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

def test_streaming():
    """Test streaming response functionality."""
    print("\nTesting streaming response...")
    
    query_data = {
        "query": "Explain the concept of RAG in simple terms",
        "model_name": "qwen2.5",
        "use_web_search": True
    }
    
    try:
        # Send request to the streaming endpoint
        print(f"Connecting to {API_URL}/query/stream")
        response = requests.post(
            f"{API_URL}/query/stream",
            json=query_data,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        response.raise_for_status()
        
        print("\nReceiving streamed response:")
        complete_response = ""
        
        # Process the SSE stream
        for line in response.iter_lines():
            if not line:
                continue
                
            line = line.decode('utf-8')
            
            # Check for SSE format "data: [content]"
            if line.startswith('data:'):
                data = line[5:].strip()
                
                # Check for completion marker
                if data == "[DONE]":
                    print("\nStream complete")
                    break
                
                # Print token and add to complete response
                print(data, end="", flush=True)
                complete_response += data
        
        print("\n\nFull streamed response:")
        print(complete_response)
        
        return True
        
    except Exception as e:
        print(f"Error during streaming test: {str(e)}")
        return False

def test_basic_query(query="What is Vibe Coding", model="qwen2.5"):
    """Test basic query functionality"""
    print(f"\n=== Testing Basic Query with model {model} ===")
    
    payload = {
        "query": query,
        "model_name": model,
        "use_web_search": True
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
                print(f"Query Analysis: {json.dumps(data.get('query_analysis'), indent=2)}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Failed to send query: {str(e)}")
        return False

def main():
    global API_URL
    
    parser = argparse.ArgumentParser(description="Test Agentic RAG API")
    parser.add_argument("--url", default=API_URL, help="API URL")
    parser.add_argument("--query", default="What is Vibe Coding", help="Query to test")
    parser.add_argument("--model", default="qwen2.5", help="Model to use")
    
    args = parser.parse_args()
    
    # Update API_URL if provided in command line
    if args.url != API_URL:
        API_URL = args.url
    
    print(f"Testing API at: {API_URL}")
    
    # Run health check
    test_health()
    
    # Run basic query test
    success = test_basic_query(args.query, args.model)
    print(f"\nBasic query test {'PASSED' if success else 'FAILED'}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main() 