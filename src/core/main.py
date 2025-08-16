# main.py
import uvicorn
import os
import asyncio
import concurrent.futures
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
import json
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from dotenv import load_dotenv
import logging
import time
from typing import List, Optional, Union, Dict, Any
import httpx
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore
import ast
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import collections
import threading
from PIL import Image
import io

# Load environment variables for security
load_dotenv()

# --- Initialize Firestore for Persistent Conversation History ---
db = None
FIREBASE_CREDENTIALS_FILE = "firebase-credentials.json"
if os.path.exists(FIREBASE_CREDENTIALS_FILE):
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_FILE)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logging.info("Firestore initialized successfully.")
    except Exception as e:
        logging.critical(f"FATAL ERROR: Failed to initialize Firestore from file: {e}")
else:
    logging.warning(f"Firestore credentials file '{FIREBASE_CREDENTIALS_FILE}' not found. Firestore will not be used.")

# --- Pydantic models for request and response validation ---
class Message(BaseModel):
    role: str
    content: str

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str
    user_id: str
    collect_data: bool = True
    conversation_history: Optional[List[Message]] = []
    image_data: Optional[str] = None

class FinalResponse(BaseModel):
    final_response: str
    agent_steps: List[str]
    sources: Optional[List[dict]] = None

# --- FastAPI App Initialization and Middleware Configuration ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global router_pipeline, specialized_pipelines, reranker_model, embedding_model, multimodal_model, multimodal_processor
    try:
        logging.info("Loading models on application startup. This may take a few minutes...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        router_pipeline = pipeline("text-generation", model="microsoft/Phi-3-mini-128k-instruct", 
                                   quantization_config=quantization_config, device_map="auto")
        logging.info("Router model loaded.")

        model_id = "microsoft/Phi-3.5-vision-instruct"
        multimodal_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            _attn_implementation="eager"
        )
        multimodal_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        logging.info("Multimodal model and processor loaded.")

        specialized_pipelines = {
            'mixtral-8x22b': pipeline("text-generation", model="mistralai/Mixtral-8x22B-Instruct-v0.1", 
                                quantization_config=quantization_config, device_map="auto"),
            'starcoder2': pipeline("text-generation", model="bigcode/starcoder2-3b", 
                                   quantization_config=quantization_config, device_map="auto"),
            'deepseek-r1': pipeline("text-generation", model="deepseek-ai/deepseek-math-7b-instruct", 
                                    quantization_config=quantization_config, device_map="auto"),
            'mistral-7b': pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3", 
                                   quantization_config=quantization_config, device_map="auto"),
        }
        logging.info("Specialized model pipelines loaded.")


        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        logging.info("Reranker and Embedding models loaded.")
        
        _populate_chroma_db()
        yield
    except Exception as e:
        logging.critical(f"FATAL ERROR: Failed to load one or more models: {e}")
        router_pipeline = None
        specialized_pipelines = {}
        multimodal_model = None
        multimodal_processor = None
        yield
        
app = FastAPI(title="Orion AI: A Multi-Model LLM", lifespan=lifespan)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
if not allowed_origins_str:
    logging.warning("ALLOWED_ORIGINS not set in .env. Using a development default.")
    origins = ["http://localhost:3000"]
else:
    origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path not in ["/docs", "/openapi.json"]:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key is not configured.")
        
        provided_key = request.headers.get("x-api-key")
        if provided_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
            
    response = await call_next(request)
    return response

# --- External Web Search API Client ---
class GoogleSearchAPI:
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    async def search(self, query: str):
        if not self.api_key or not self.cse_id:
            logging.error("Google Search API key or CSE ID is not configured.")
            return []

        params = {"key": self.api_key, "cx": self.cse_id, "q": query, "num": 5}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                results = []
                if "items" in data:
                    for item in data["items"]:
                        results.append({"title": item.get("title", ""), "url": item.get("link", "#"), "snippet": item.get("snippet", "")})
                return results
        except httpx.HTTPError as exc:
            logging.error(f"HTTP error for Google Search: {exc}")
            return []
        except Exception as exc:
            logging.error(f"An unexpected error occurred during Google Search: {exc}")
            return []

# --- Caching Layer for Responses and Rate Limiting ---
cache = collections.OrderedDict()
CACHE_TTL = 3600 # 1 hour
user_request_counts = collections.defaultdict(int)
RATE_LIMIT = 10 # 10 requests per minute
rate_limit_lock = threading.Lock()

def check_rate_limit(user_id: str) -> bool:
    with rate_limit_lock:
        if user_request_counts[user_id] >= RATE_LIMIT:
            return False
        user_request_counts[user_id] += 1
        return True

def reset_rate_limits():
    while True:
        time.sleep(60)
        with rate_limit_lock:
            user_request_counts.clear()

# Start a background thread to reset rate limits
rate_limit_thread = threading.Thread(target=reset_rate_limits, daemon=True)
rate_limit_thread.start()

# --- Tool Definitions for Orchestration ---
specialized_models = {
    'mixtral-8x22b': {'name': "mistralai/Mixtral-8x22B-Instruct-v0.1", 'specialty': 'General Purpose Conversational AI'},
    'starcoder2': {'name': "bigcode/starcoder2-3b", 'specialty': 'Code Generation'},
    'deepseek-r1': {'name': "deepseek-ai/deepseek-math-7b-instruct", 'specialty': 'Advanced Reasoning'},
    'mistral-7b': {'name': "mistralai/Mistral-7B-Instruct-v0.3", 'specialty': 'High-Performance Generalist'},
}

multimodal_models = {
    'phi-3-vision': {'name': "microsoft/Phi-3.5-vision-instruct", 'specialty': 'Image and Document Analysis'}
}

tools = [
    {
        "name": "web_search",
        "description": "Performs a real-time web search for information. Use this for questions that require current knowledge or external research. Input is a single string search query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}
    },
    {
        "name": "deep_research",
        "description": "Performs a multi-step, in-depth web search to answer complex questions that require synthesizing information from multiple sources. Use this for nuanced or controversial topics. Input is a single string search query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The complex research query."}}, "required": ["query"]}
    },
    {
        "name": "calculator",
        "description": "A tool that can perform mathematical calculations. Use this for any calculation or mathematical expression. Input is a single string expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate."}}, "required": ["expression"]}
    },
    {
        "name": "image_analyzer",
        "description": "Analyzes an image to extract text, describe its contents, or identify key objects. Use this when the user has provided an image with their query. Input is a base64 encoded image string.",
        "parameters": {"type": "object", "properties": {"image_data": {"type": "string", "description": "The base64 encoded image string."}}, "required": ["image_data"]}
    },
]

# --- This section loads the models on application startup ---
router_pipeline = None
specialized_pipelines = {}
multimodal_model = None
multimodal_processor = None
reranker_model = None
embedding_model = None
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
db_client = chromadb.Client()
documents_collection = db_client.get_or_create_collection(name="documents")

DOCUMENTS = [
    "Orion AI is a multi-model LLM designed for advanced reasoning and a unified user experience.",
    "The core of Orion AI's architecture is an intelligent router that selects specialized expert models.",
    "Orion AI's RAG system can perform multi-step search and post-retrieval reranking to provide high-quality, relevant answers."
]
def _populate_chroma_db():
    if not documents_collection.count():
        documents_collection.add(
            documents=DOCUMENTS,
            ids=[f"doc_{i}" for i in range(len(DOCUMENTS))]
        )
        logging.info("ChromaDB populated with demonstration data.")


def run_model_sync(pipe, prompt, max_tokens):
    output = pipe(prompt, max_new_tokens=max_tokens, return_full_text=False)
    return output[0]['generated_text']

def log_user_data(data):
    log_file = "user_data_log.jsonl"
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        logging.info("User query and response logged.")
    except Exception as e:
        logging.error(f"Failed to log user data: {e}")

def safe_json_parse(json_str: str, max_retries: int = 3) -> Optional[dict]:
    cleaned_str = json_str.strip()
    if cleaned_str.startswith("```json"): cleaned_str = cleaned_str[7:].strip()
    if cleaned_str.endswith("```"): cleaned_str = cleaned_str[:-3].strip()

    for attempt in range(max_retries):
        try: return json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}.")
            last_brace_index = cleaned_str.rfind('}')
            if last_brace_index != -1: cleaned_str = cleaned_str[:last_brace_index + 1]
            else: return None
    return None

async def execute_tool(tool_call: ToolCall, query: str, image_data: Optional[str]) -> str:
    tool_name = tool_call.name
    args = tool_call.arguments
    logging.info(f"Executing tool: {tool_name} with args: {args}")
    loop = asyncio.get_event_loop()

    if tool_name == "web_search":
        search_api = GoogleSearchAPI(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cse_id=os.getenv("GOOGLE_CSE_ID"))
        search_query = args.get("query", query)
        try:
            results = await search_api.search(query=search_query)
            if not results: return "Web search returned no results."
            pairs = [(search_query, res['snippet']) for res in results]
            scores = reranker_model.predict(pairs)
            reranked_results = [res for _, res in sorted(zip(scores, results), reverse=True)]
            return json.dumps(reranked_results)
        except Exception as e:
            return f"Tool execution failed: web_search failed with error: {e}"
    
    elif tool_name == "deep_research":
        search_api = GoogleSearchAPI(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"), cse_id=os.getenv("GOOGLE_CSE_ID"))
        search_query = args.get("query", query)
        
        multi_query_prompt = f"""
        You are a research assistant. Your task is to generate up to 3 alternative search queries for the user's request to ensure a comprehensive search.

        Original Query: "{search_query}"

        Queries (as a JSON array of strings):
        """
        multi_query_output = await loop.run_in_executor(
            executor, run_model_sync, specialized_pipelines['mixtral-8x22b'], multi_query_prompt, 100
        )
        sub_queries = safe_json_parse(multi_query_output)
        if not sub_queries:
            sub_queries = [search_query]
        
        all_results = []
        for q in sub_queries:
            results = await search_api.search(query=q)
            all_results.extend(results)
        
        if not all_results:
            return "Deep research returned no results."
        
        pairs = [(search_query, res['snippet']) for res in all_results]
        scores = reranker_model.predict(pairs)
        reranked_results = [res for _, res in sorted(zip(scores, all_results), reverse=True)]
        
        return json.dumps(reranked_results[:10])

    elif tool_name == "calculator":
        try:
            expression = args.get("expression")
            node = ast.parse(expression, mode='eval')
            result = eval(compile(node, '<string>', 'eval'))
            return f"The result of the calculation is: {result}"
        except Exception as e:
            return f"Tool execution failed: calculator failed with error: {e}"
    
    elif tool_name == "image_analyzer":
        if not image_data:
            return "No image data provided."
        if not multimodal_model or not multimodal_processor:
            return "Multimodal model is not loaded."
        
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))

            user_prompt = args.get("prompt", "Analyze this image in detail.")
            
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{user_prompt}"}
            ]
            
            prompt = multimodal_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = multimodal_processor(prompt, [image], return_tensors="pt").to(multimodal_model.device)

            generate_ids = multimodal_model.generate(**inputs, max_new_tokens=500, eos_token_id=multimodal_processor.tokenizer.eos_token_id)

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response_text = multimodal_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return response_text
        except Exception as e:
            logging.error(f"Image analysis failed: {e}")
            return f"Tool execution failed: image_analyzer failed with error: {e}"
    
    return f"Error: Tool '{tool_name}' not found."

async def _summarize_history(history: List[Message], loop) -> str:
    summarizer_pipeline = specialized_pipelines.get('mistral-7b')
    if not summarizer_pipeline: return "Unable to summarize conversation history."

    history_text = "\n".join([f"{m.role}: {m.content}" for m in history])
    prompt = f"Summarize the following conversation history:\n\n{history_text}\n\nSummary:"
    summary = await loop.run_in_executor(executor, run_model_sync, summarizer_pipeline, prompt, 150)
    return summary

@app.post("/route_query", response_model=FinalResponse)
async def route_query_endpoint(request: QueryRequest):
    if not router_pipeline or not specialized_pipelines:
        raise HTTPException(status_code=503, detail="Service is currently unavailable.")

    # --- Implement Caching ---
    cache_key = json.dumps({"query": request.query, "history": [h.dict() for h in request.conversation_history]})
    if cache_key in cache:
        cached_response, timestamp = cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            logging.info("Serving response from cache.")
            return cached_response
        else:
            del cache[cache_key] # Expire cache

    # --- Implement Rate Limiting ---
    if not check_rate_limit(request.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

    try:
        logging.info(f"Received query: '{request.query}' from user '{request.user_id}' with image: {bool(request.image_data)}")
        agent_steps = []
        conversation_history = request.conversation_history
        
        system_message = {"role": "system", "content": "You are Orion AI, a helpful and highly knowledgeable assistant. Your goal is to provide accurate and concise responses."}
        full_conversation = [Message(**system_message)] + conversation_history

        image_prompt_part = ""
        if request.image_data:
            image_prompt_part = "The user has provided an image. Use the image_analyzer tool to describe it or answer questions about it. If the user's query is just the image, you must call the image_analyzer tool with a descriptive prompt."

        router_prompt = f"""
        You are Orion AI, a highly intelligent and helpful agent. You have access to the following tools:
        {json.dumps(tools, indent=2)}

        {image_prompt_part}

        You can either respond directly to the user or call one of the tools. When calling a tool, your output must be a JSON object with 'tool_call' and 'response_text' keys. The 'tool_call' should be a JSON object with 'name' and 'arguments' keys.

        Conversation History:
        {json.dumps([h.dict() for h in full_conversation], indent=2)}

        User's Query: "{request.query}"
        
        Final Answer or Tool Call:
        """
        
        loop = asyncio.get_event_loop()
        router_text_response = await loop.run_in_executor(executor, run_model_sync, router_pipeline, router_prompt, 250)
        agent_steps.append(f"Router's raw response: {router_text_response}")
        
        parsed_decision = safe_json_parse(router_text_response)
        sources = None

        final_response_obj = None

        if parsed_decision and "tool_call" in parsed_decision and parsed_decision['tool_call']:
            tool_call_data = parsed_decision['tool_call']
            try:
                tool_call = ToolCall(**tool_call_data)
                tool_output = await execute_tool(tool_call, request.query, request.image_data)
                agent_steps.append(f"Called tool '{tool_call.name}'. Output: {tool_output}")
            
                synthesis_prompt = f"""
                You are Orion AI. A tool was just executed. The tool's output is provided below. Use this information, along with the original conversation, to form a final, helpful response to the user.

                Conversation History:
                {json.dumps([h.dict() for h in full_conversation], indent=2)}

                User's Query: "{request.query}"

                Tool Name: {tool_call.name}
                Tool Output: {tool_output}

                Unified Response:
                """
                synthesis_pipeline = specialized_pipelines.get('mixtral-8x22b')
                final_response_text = await loop.run_in_executor(executor, run_model_sync, synthesis_pipeline, synthesis_prompt, 400)
                agent_steps.append("Synthesized a final response using the tool output.")
            
                try:
                    if tool_call.name in ["web_search", "deep_research"]:
                        raw_sources = json.loads(tool_output)
                        sources = raw_sources
                except Exception:
                    sources = None

                final_response_obj = FinalResponse(
                    final_response=final_response_text,
                    agent_steps=agent_steps,
                    sources=sources
                )
            except ValidationError as e:
                logging.error(f"Router returned invalid tool call: {tool_call_data} - Error: {e}")
                synthesis_pipeline = specialized_pipelines.get('mixtral-8x22b')
                final_response_text = await loop.run_in_executor(executor, run_model_sync, synthesis_pipeline, "The AI agent tried to use a tool but failed due to an invalid format. Please try again.", 200)
                final_response_obj = FinalResponse(final_response=final_response_text, agent_steps=agent_steps)
        else:
            synthesis_pipeline = specialized_pipelines.get('mixtral-8x22b')
            synthesis_prompt = f"""
            You are Orion AI. Respond directly and helpfully to the user's query.

            Conversation History:
            {json.dumps([h.dict() for h in full_conversation], indent=2)}

            User's Query: "{request.query}"

            Unified Response:
            """
            final_response_text = await loop.run_in_executor(executor, run_model_sync, synthesis_pipeline, synthesis_prompt, 400)

            final_response_obj = FinalResponse(
                final_response=final_response_text,
                agent_steps=agent_steps,
                sources=sources
            )

        if request.collect_data:
            log_user_data({"timestamp": time.time(), "query": request.query, "response": final_response_obj.final_response})
        
        # Store response in cache
        cache[cache_key] = (final_response_obj, time.time())

        return final_response_obj

    except (HTTPException, ValidationError) as e:
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return FinalResponse(
            final_response=f"I'm sorry, I encountered an unexpected error while processing your request: {str(e)}. Please try again later.",
            agent_steps=[],
            sources=None
        )
