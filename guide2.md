# Ridges AI Proxy Migration

This guide will help you migrate the AI proxy and embedding proxy functionality from your main API to a standalone service.

## Quick Overview

**Current**: Agent → Main API → Chutes API  
**Target**: Agent → Standalone Proxy → Chutes API

## Step 1: Set Up the New Repository Structure

In your `https://github.com/ridgesai/proxy` repository, create this structure:

```
proxy/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chutes_manager.py
│   │   └── auth.py
│   └── routers/
│       ├── __init__.py
│       └── proxy.py
├── requirements.txt
├── .env.example
└── README.md
```

## Step 2: Copy the Files I Created

1. **Copy `chutes_manager_for_proxy.py`** → `app/core/chutes_manager.py`
2. **Copy `proxy_requirements.txt`** → `requirements.txt`
3. **Copy the code from proxy_migration_guide.md** for:
   - `app/main.py`
   - `app/models.py` 
   - `app/routers/proxy.py`
   - `app/core/auth.py`

## Step 3: Deploy to EC2

### Launch EC2 Instance
1. Create Ubuntu 22.04 LTS EC2 instance
2. Configure security groups:
   - SSH (port 22) from your IP
   - Custom TCP (port 8000) from your main server IP

### Set Up the Server
```bash
# SSH into EC2 instance
ssh ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip git nginx -y

# Clone your proxy repository
git clone https://github.com/ridgesai/proxy.git
cd proxy

# Install Python dependencies
pip3 install -r requirements.txt
```

### Configure Environment
```bash
# Create .env file
cp .env.example .env
nano .env
```

Add your configuration:
```bash
CHUTES_API_KEY=your_chutes_api_key_here
PROXY_API_KEY=generate_a_secure_random_key
MAX_COST_PER_RUN=2.0
EMBEDDING_PRICE_PER_SECOND=0.0001
ALLOWED_IPS=your_main_server_ip,validator_ips
```

### Test the Service
```bash
# Run the service
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test in another terminal
curl http://localhost:8000/health
```

### Set Up as System Service
```bash
# Create systemd service
sudo nano /etc/systemd/system/ridges-proxy.service
```

Add this content:
```ini
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
```

```bash
# Enable and start service
sudo systemctl enable ridges-proxy
sudo systemctl start ridges-proxy
sudo systemctl status ridges-proxy
```

## Step 4: Update Your Main API

### Option A: Route to Proxy (Minimal Changes)
Update your current `api/src/endpoints/agents.py` to proxy requests:

```python
import httpx
from fastapi import APIRouter, HTTPException, Depends

async def embedding(request: EmbeddingRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://your-proxy-ec2-ip:8000/agents/embedding",
            json=request.dict(),
            headers={"Authorization": f"Bearer {PROXY_API_KEY}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

async def inference(request: InferenceRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://your-proxy-ec2-ip:8000/agents/inference", 
            json=request.dict(),
            headers={"Authorization": f"Bearer {PROXY_API_KEY}"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
```

### Option B: Remove Endpoints (Clean Separation)
1. Remove the `/agents/embedding` and `/agents/inference` endpoints from your main API
2. Update agent configuration to point directly to the proxy

## Step 5: Update Agent Configuration

In your agent deployment/sandbox configuration, change the proxy URL:

```python
# Before
proxy_url = "https://api.ridges.ai"

# After
proxy_url = "http://your-proxy-ec2-ip:8000"
```

## Step 6: Test End-to-End

### Test with the Provided Script
```bash
# Copy test_proxy.py to your EC2 instance
python3 test_proxy.py --host "http://your-proxy-ec2-ip:8000" --api-key "your_api_key"
```

### Or Test Each Endpoint Manually
```bash
# Health check
curl http://your-proxy-ec2-ip:8000/health

# Embedding
curl -X POST http://your-proxy-ec2-ip:8000/agents/embedding \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test text", "run_id": "test-run-123"}'

# Inference  
curl -X POST http://your-proxy-ec2-ip:8000/agents/inference \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "test-run-123",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "temperature": 0.7,
    "model": "deepseek-ai/DeepSeek-V3-0324"
  }'
```

## Step 7: Monitor and Maintain

### Check Logs
```bash
# Service logs
sudo journalctl -u ridges-proxy -f

# Application logs (if you add file logging)
tail -f /home/ubuntu/proxy/logs/app.log
```

### Monitor Costs
The proxy automatically tracks costs per run_id and enforces limits. Check the logs for cost information.

### Health Monitoring
Set up a cron job or monitoring service to check:
```bash
curl http://your-proxy-ec2-ip:8000/health
```

## Benefits After Migration

✅ **Isolation**: Proxy failures won't affect your main API  
✅ **Scalability**: Scale proxy independently based on AI usage  
✅ **Security**: Reduced attack surface on main API  
✅ **Cost Management**: Better visibility into AI-related costs  
✅ **Flexibility**: Easier to update AI logic without touching main API

## Troubleshooting

### Common Issues
1. **"CHUTES_API_KEY not found"**: Check your .env file
2. **"Connection refused"**: Verify the service is running and ports are open
3. **"Invalid API key"**: Check PROXY_API_KEY matches between proxy and main API
4. **Cost limit errors**: Adjust MAX_COST_PER_RUN if needed

### Service Commands
```bash
# Restart service
sudo systemctl restart ridges-proxy

# Check status
sudo systemctl status ridges-proxy

# View logs
sudo journalctl -u ridges-proxy -f
```

## Need Help?

If you run into issues:
1. Check the service logs: `sudo journalctl -u ridges-proxy -f`
2. Verify your .env configuration
3. Test with curl commands to isolate the issue
4. Check security groups allow traffic on port 8000 