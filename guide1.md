# AI Proxy Migration Guide

## Overview
This guide explains how to migrate the AI proxy and embedding proxy functionality from the main Ridges API to a standalone FastAPI service running on its own EC2 instance.

## Architecture Changes

### Before (Current)
- AI proxy endpoints are part of the main API server
- Calls go: `Agent → Main API Server → Chutes API`
- Database validation checks evaluation runs

### After (Target)
- AI proxy runs as standalone service
- Calls go: `Agent → Proxy Service → Chutes API`
- No database dependency, simpler authentication

## Files to Create in New Repository

### 1. app/main.py
```python
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

from app.routers.proxy import router as proxy_router
from app.core.chutes_manager import ChutesManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ridges AI Proxy", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(proxy_router, prefix="/agents")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    chutes_manager = ChutesManager()
    chutes_manager.start_cleanup_task()
    logger.info("AI Proxy service started successfully")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ridges-ai-proxy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. app/models.py
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="Text to embed")
    run_id: str = Field(..., description="Evaluation run ID")

class GPTMessage(BaseModel):
    role: str
    content: str
    
class InferenceRequest(BaseModel):
    run_id: str = Field(..., description="Evaluation run ID")
    model: Optional[str] = Field(None, description="Model to use for inference")
    temperature: Optional[float] = Field(None, description="Temperature for inference")
    messages: List[GPTMessage] = Field(..., description="Messages to send to the model")
```

### 3. app/routers/proxy.py
```python
from fastapi import APIRouter, HTTPException, Depends
import logging

from app.models import EmbeddingRequest, InferenceRequest
from app.core.chutes_manager import ChutesManager
from app.core.auth import verify_request

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize ChutesManager
chutes = ChutesManager()

@router.post("/embedding", dependencies=[Depends(verify_request)])
async def embedding(request: EmbeddingRequest):
    """Get embeddings for text input"""
    try:
        embedding = await chutes.embed(request.run_id, request.input)
        logger.debug(f"Embedding for {request.run_id} was requested and returned")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for {request.run_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to get embedding due to internal server error. Please try again later."
        )

@router.post("/inference", dependencies=[Depends(verify_request)])
async def inference(request: InferenceRequest):
    """Get inference from language model"""
    try:
        response = await chutes.inference(
            request.run_id, 
            request.messages,
            request.temperature,
            request.model
        )
        logger.debug(f"Inference for {request.run_id} was requested and returned")
        return response
    except Exception as e:
        logger.error(f"Error getting inference for {request.run_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to get inference due to internal server error. Please try again later."
        )
```

### 4. app/core/chutes_manager.py
```python
import os
import dotenv
import httpx
import json 
from typing import List
from datetime import datetime, timedelta
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

class ChutesManager:
    def __init__(self):
        self.api_key = os.getenv('CHUTES_API_KEY')
        if not self.api_key:
            raise ValueError("CHUTES_API_KEY environment variable is required")
            
        # Pricing configuration
        self.pricing = {
            "deepseek-ai/DeepSeek-V3-0324": 0.27,
            # Add other models as needed
        }
        
        # Cost tracking per run_id
        self.costs_data_inference = {}
        self.costs_data_embedding = {}
        self.cleanup_task = None
        self._cleanup_started = False

        # Configuration
        self.max_cost_per_run = float(os.getenv("MAX_COST_PER_RUN", "2.0"))
        self.embedding_price_per_second = float(os.getenv("EMBEDDING_PRICE_PER_SECOND", "0.0001"))

    def start_cleanup_task(self):
        """Start the periodic cleanup task to remove cost data that is older than 20 minutes. This is run every 5 minutes."""
        if self._cleanup_started:
            return
        
        try:
            async def cleanup_loop():
                while True:
                    logger.info("Started cleaning up old entries from Chutes")
                    await self.cleanup_old_entries()
                    logger.info("Finished cleaning up old entries from Chutes. Running again in 5 minutes.")
                    await asyncio.sleep(300)
            
            self.cleanup_task = asyncio.create_task(cleanup_loop())
            self._cleanup_started = True
            logger.info("Chutes cleanup task started")
        except RuntimeError:
            # No event loop running, will try again later
            logger.warning("No event loop available for cleanup task, will retry later")

    def _ensure_cleanup_task(self):
        """Ensure cleanup task is started if event loop is available."""
        if not self._cleanup_started:
            self.start_cleanup_task()

    async def cleanup_old_entries(self) -> None:
        """Remove cost data that is older than 20 minutes"""
        try:
            current_time = datetime.now()
            keys_to_remove_inference = []
            keys_to_remove_embedding = []

            for key, value in self.costs_data_inference.items():
                if current_time - value["started_at"] > timedelta(minutes=20):
                    keys_to_remove_inference.append(key)
            
            for key, value in self.costs_data_embedding.items():
                if current_time - value["started_at"] > timedelta(minutes=20):
                    keys_to_remove_embedding.append(key)
        
            for key in keys_to_remove_inference:
                del self.costs_data_inference[key]
            for key in keys_to_remove_embedding:
                del self.costs_data_embedding[key]
                
            if keys_to_remove_inference or keys_to_remove_embedding:
                logger.info(f"Removed {len(keys_to_remove_inference)} old inference entries and {len(keys_to_remove_embedding)} old embedding entries from Chutes pricing data")
        except Exception as e:
            logger.error(f"Error cleaning up old entries from Chutes pricing data: {e}")

    async def embed(self, run_id: str, prompt: str) -> dict:
        """Generate embeddings for the given prompt"""
        self._ensure_cleanup_task()
        
        # Check cost limits
        if self.costs_data_embedding.get(run_id, {}).get("spend", 0) >= self.max_cost_per_run:
            error_msg = f"Agent version from run {run_id} has reached the maximum cost from their evaluation run."
            logger.info(error_msg)
            return {"error": "Your agent version has reached the maximum cost for this evaluation run. Please do not request more embeddings or inference from this agent version."}

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json"
        }
        
        body = {
            "inputs": prompt
        }

        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://chutes-baai-bge-large-en-v1-5.chutes.ai/embed",
                    headers=headers,
                    json=body
                )
                response.raise_for_status()
                
                total_time_seconds = time.time() - start_time
                cost = total_time_seconds * self.embedding_price_per_second

                # Update cost tracking
                self.costs_data_embedding[run_id] = {
                    "spend": self.costs_data_embedding.get(run_id, {}).get("spend", 0) + cost,
                    "started_at": self.costs_data_embedding.get(run_id, {}).get("started_at", datetime.now())
                }

                logger.debug(f"Updated embedding spend for run {run_id}: {cost} (total: {self.costs_data_embedding[run_id]['spend']})")

                return response.json()
                
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error in embed request: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error in embed request: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    async def inference(self, run_id: str, messages: List[dict], temperature: float = 0.7, model: str = None):
        """Generate inference from language model"""
        self._ensure_cleanup_task()
        
        if not model:
            model = "deepseek-ai/DeepSeek-V3-0324"

        if model not in self.pricing:
            error_msg = f"Model {model} not supported. Please use one of the following models: {list(self.pricing.keys())}"
            logger.info(f"Agent version from run {run_id} requested an unsupported model: {model}.")
            return {"error": error_msg}
        
        # Check cost limits
        if self.costs_data_inference.get(run_id, {}).get("spend", 0) >= self.max_cost_per_run:
            error_msg = f"Agent version from run {run_id} has reached the maximum cost from their evaluation run."
            logger.info(error_msg)
            return {"error": "Your agent version has reached the maximum cost for this evaluation run. Please do not request more inference from this agent version."}
        
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json"
        }
        
        body = {
            "model": model,
            "messages": [],
            "stream": True,
            "max_tokens": 1024,
            "temperature": temperature if temperature is not None else 0.7
        }

        # Convert messages to proper format
        if messages is not None:
            for message in messages:
                if message is not None:
                    if hasattr(message, 'role') and hasattr(message, 'content'):
                        # Pydantic model
                        body['messages'].append({
                            "role": message.role,
                            "content": message.content
                        })
                    elif isinstance(message, dict):
                        # Dict format
                        body['messages'].append({
                            "role": message.get("role"),
                            "content": message.get("content")
                        })

        logger.debug(f"Inference request body: {body}")

        response_chunks = []
        total_tokens = 0
        
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    "https://llm.chutes.ai/v1/chat/completions", 
                    headers=headers,
                    json=body
                ) as response:
                    logger.info(f"Inference response status: {response.status_code}")
                    
                    if response.status_code != 200:
                        error_text = await response.aread()
                        if isinstance(error_text, bytes):
                            error_message = error_text.decode()
                        else:
                            error_message = str(error_text)
                        logger.error(f"API request failed with status {response.status_code}: {error_message}")
                        raise Exception(f"API request failed with status {response.status_code}: {error_message}")
                    
                    # Process streaming response
                    async for line_bytes in response.aiter_lines():
                        line = line_bytes.strip()
                        if not line:
                            continue
                            
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            
                            try:
                                chunk_json = json.loads(data)
                                
                                # Extract content from delta
                                if chunk_json and 'choices' in chunk_json and chunk_json['choices']:
                                    choice = chunk_json['choices'][0]
                                    if choice and 'delta' in choice and 'content' in choice['delta']:
                                        content = choice['delta']['content']
                                        if content:
                                            response_chunks.append(content)
                                
                                # Extract usage data
                                if chunk_json and 'usage' in chunk_json and chunk_json['usage'] is not None and 'total_tokens' in chunk_json['usage']:
                                    total_tokens = chunk_json['usage']['total_tokens']
                                        
                            except json.JSONDecodeError:
                                pass
                    
                    # Check if response contains an error message despite 200 status
                    response_text = "".join(response_chunks)
                    if response_text and ("Internal Server Error" in response_text or "exhausted all available targets" in response_text):
                        logger.error(f"API returned error in response body: {response_text}")
                        raise Exception(f"API Error: {response_text}")
        
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error in inference request: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error in inference request: {e}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
        
        # Update costs data if we received usage information
        if total_tokens > 0:
            total_cost = total_tokens * self.pricing[model] / 1000000
            key = run_id
            self.costs_data_inference[key] = {
                "spend": self.costs_data_inference.get(key, {}).get("spend", 0) + total_cost,
                "started_at": self.costs_data_inference.get(key, {}).get("started_at", datetime.now())
            }
            logger.debug(f"Updated inference spend for run {run_id}: {total_cost} (total: {self.costs_data_inference[key]['spend']})")
        
        response_text = "".join(response_chunks)
        logger.debug(f"Final response length: {len(response_text)}")
        
        # If we got no response chunks but the API call succeeded, return a fallback message
        if not response_chunks:
            logger.warning("No response chunks collected, returning fallback message")
            return "No response content received from the model"
        
        return response_text
```

### 5. app/core/auth.py
```python
from fastapi import HTTPException, Request
import os
from typing import Optional

async def verify_request(request: Request):
    """Simplified authentication for proxy service"""
    # Option 1: API Key based auth
    api_key = request.headers.get("Authorization")
    expected_key = os.getenv("PROXY_API_KEY")
    
    if not api_key or api_key != f"Bearer {expected_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Option 2: IP whitelist (if needed)
    client_ip = request.client.host
    allowed_ips = os.getenv("ALLOWED_IPS", "").split(",")
    
    if allowed_ips and client_ip not in allowed_ips:
        raise HTTPException(status_code=403, detail="IP not allowed")
    
    return True
```

### 6. app/core/config.py
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Chutes API configuration
CHUTES_API_KEY = os.getenv('CHUTES_API_KEY')

# Pricing configuration
MODEL_PRICE_PER_1M_TOKENS = {
    "deepseek-ai/DeepSeek-V3-0324": 0.27,
    # Add other models as needed
}

EMBEDDING_PRICE_PER_SECOND = 0.0001

# Authentication
PROXY_API_KEY = os.getenv('PROXY_API_KEY')
ALLOWED_IPS = os.getenv('ALLOWED_IPS', '').split(',')

# Rate limiting
MAX_COST_PER_RUN = 2.0
```

## Deployment Steps

### 1. Set up EC2 Instance
```bash
# Launch Ubuntu 22.04 EC2 instance
# Configure security groups to allow:
# - Port 22 (SSH)
# - Port 8000 (API)
# - Port 443 (HTTPS if using SSL)
```

### 2. Install Dependencies
```bash
# SSH into EC2 instance
sudo apt update
sudo apt install python3 python3-pip nginx -y

# Install Python dependencies
pip3 install fastapi uvicorn python-dotenv httpx
```

### 3. Environment Configuration
```bash
# Create .env file
cat > .env << EOF
CHUTES_API_KEY=your_chutes_api_key_here
PROXY_API_KEY=your_secure_proxy_api_key_here
ALLOWED_IPS=main_server_ip,validator_ips
EOF
```

### 4. Run the Service
```bash
# Option 1: Direct run (for testing)
python3 -m app.main

# Option 2: Using systemd (production)
sudo tee /etc/systemd/system/ridges-proxy.service > /dev/null <<EOF
[Unit]
Description=Ridges AI Proxy
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/proxy
Environment=PATH=/usr/bin:/usr/local/bin
ExecStart=/usr/bin/python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable ridges-proxy
sudo systemctl start ridges-proxy
```

### 5. Nginx Configuration (Optional)
```nginx
server {
    listen 80;
    server_name your-proxy-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Update Main API Server

### Remove Database Validation
In your main API server, update the agent endpoints to point to the new proxy service instead of handling requests directly.

### Update Agent Code
Update the proxy_url in your agent deployment to point to the new EC2 instance:
```python
# Before
proxy_url = "https://api.ridges.ai"

# After  
proxy_url = "https://your-proxy-ec2-instance.com"
```

## Testing

### 1. Health Check
```bash
curl http://your-proxy-instance:8000/health
```

### 2. Test Embedding
```bash
curl -X POST http://your-proxy-instance:8000/agents/embedding \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"input": "test text", "run_id": "test-run"}'
```

### 3. Test Inference
```bash
curl -X POST http://your-proxy-instance:8000/agents/inference \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "test-run",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'
```

## Security Considerations

1. **API Key Management**: Use strong, unique API keys
2. **IP Whitelisting**: Restrict access to known validator/agent IPs
3. **Rate Limiting**: Implement rate limiting per IP/run_id
4. **SSL/TLS**: Use HTTPS in production
5. **Firewall**: Configure AWS security groups appropriately

## Monitoring

1. **Logs**: Monitor application logs for errors
2. **Metrics**: Track request latency and error rates
3. **Cost Tracking**: Monitor Chutes API usage costs
4. **Health Checks**: Set up automated health monitoring

## Benefits of This Migration

1. **Isolation**: Proxy failures won't affect main API
2. **Scalability**: Can scale proxy independently
3. **Security**: Reduced attack surface on main API
4. **Cost Management**: Better visibility into AI-related costs
5. **Maintenance**: Easier to update proxy logic independently 