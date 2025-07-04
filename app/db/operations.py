import os
import logging
from typing import Optional
import threading
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from app.models import EmbeddingRequest, InferenceRequest, GPTMessage
from .sqlalchemy_models import Base, EvaluationRun

load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    AsyncSessionLocal = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    async def init(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.engine = None
                    self.AsyncSessionLocal = None
                    self._initialize_engine()
                    await self._init_tables()
                    self._initialized = True
    
    def _initialize_engine(self):
        """Initialize the async SQLAlchemy engine and session maker"""
        try:
            # Use asyncpg as the async driver
            database_url = f"postgresql+asyncpg://{os.getenv('AWS_MASTER_USERNAME')}:{os.getenv('AWS_MASTER_PASSWORD')}@{os.getenv('AWS_RDS_PLATFORM_ENDPOINT')}/{os.getenv('AWS_RDS_PLATFORM_DB_NAME')}"
            
            self.engine = create_async_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                autoflush=False,
                autocommit=False
            )
            logger.info("Async SQLAlchemy engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize async SQLAlchemy engine: {str(e)}")
            raise

    async def _init_tables(self):
        """
        Check if required tables exist and create them if they don't.
        """
        try:
            # Create tables using SQLAlchemy
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {str(e)}")
            raise

    async def close(self):
        """Close the database engine."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine closed")

    async def get_evaluation_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Get an evaluation run by ID"""
        async with self.AsyncSessionLocal() as session:
            try:
                stmt = select(EvaluationRun).where(EvaluationRun.run_id == run_id)
                result = await session.execute(stmt)
                evaluation_run = result.scalar_one_or_none()
                return evaluation_run
            except Exception as e:
                logger.error(f"Error getting evaluation run {run_id}: {str(e)}")
                return None
    
