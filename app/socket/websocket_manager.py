from fastapi import WebSocket
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

class WebSocketManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.active_connections: Dict[str, WebSocket] = {}
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def connect(self, websocket: WebSocket, validator_key: str):
        await websocket.accept()
        self.active_connections[validator_key] = websocket
        logger.info(f"Validator {validator_key} connected")
    
    def disconnect(self, validator_key: str):
        if validator_key in self.active_connections:
            del self.active_connections[validator_key]
            logger.info(f"Validator {validator_key} disconnected")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")