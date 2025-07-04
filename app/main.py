import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import argparse

from app.routers.proxy import router as proxy_router
from app.core.chutes_manager import ChutesManager
from app.db.operations import DatabaseManager
from app.socket.websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ridges AI Proxy", version="1.0.0")

# Configure CORS - removed invalid allow_websockets parameter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(proxy_router, prefix="/agents")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for validator communication"""
    try:
        # Accept the connection immediately
        await websocket.accept()
        validator_key = websocket.headers.get("validator-key", str(id(websocket)))
        logger.info(f"New WebSocket connection accepted: {validator_key}")
        
        try:
            while True:
                # Wait for messages from the validator
                message = await websocket.receive_json()
                logger.info(f"Received message from validator {validator_key}: {message}")
                
                # Echo back the message
                await websocket.send_json({
                    "status": "received",
                    "message": message
                })
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {validator_key}")
        except Exception as e:
            logger.error(f"Error handling message for {validator_key}: {e}")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        try:
            WebSocketManager.get_instance().disconnect(validator_key)
            logger.info(f"Cleaned up connection for {validator_key}")
        except Exception as e:
            logger.error(f"Error during connection cleanup: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        db = DatabaseManager()
        await db.init()
        logger.info("Database initialized successfully")
        
        # Initialize chutes manager
        chutes_manager = ChutesManager()
        chutes_manager.start_cleanup_task()
        logger.info("AI Proxy service started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ridges-ai-proxy"}

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ridges AI Proxy Service')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the service on')
    args = parser.parse_args()
    
    # Run with WebSocket support
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        ws_ping_timeout=None,
        ws_max_size=10 * 1024 * 1024  # 10MB max message size
    ) 