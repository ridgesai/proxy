import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import argparse

from app.routers.proxy import router as proxy_router
from app.core.chutes_manager import ChutesManager
from app.db.operations import DatabaseManager

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
    # Initialize database
    db = DatabaseManager()
    await db.init()
    logger.info("Database initialized successfully")
    
    # Initialize chutes manager
    chutes_manager = ChutesManager()
    chutes_manager.start_cleanup_task()
    logger.info("AI Proxy service started successfully")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ridges-ai-proxy"}

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ridges AI Proxy Service')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the service on')
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)