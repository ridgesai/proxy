import os
import logging
from typing import Optional, Dict, Any, List
import threading
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text, inspect
import uuid

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
                    logger.info("Database manager initialized successfully")
    
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

    async def get_db_info(self) -> Dict[str, Any]:
        """Get information about the database structure"""
        if not self._initialized:
            await self.init()
            
        try:
            info = {
                "database_name": os.getenv('AWS_RDS_PLATFORM_DB_NAME'),
                "tables": [],
                "has_evaluation_runs_table": False,
                "evaluation_runs_columns": []
            }
            
            async with self.AsyncSessionLocal() as session:
                # Get list of tables
                tables_result = await session.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'"))
                tables = tables_result.fetchall()
                info["tables"] = [table[0] for table in tables]
                
                # Check if evaluation_runs table exists
                if 'evaluation_runs' in info["tables"]:
                    info["has_evaluation_runs_table"] = True
                    
                    # Get columns of evaluation_runs table
                    columns_result = await session.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'evaluation_runs'"))
                    columns = columns_result.fetchall()
                    info["evaluation_runs_columns"] = [{"name": col[0], "type": col[1]} for col in columns]
                    
                    # Try to get a sample row count
                    count_result = await session.execute(text("SELECT COUNT(*) FROM evaluation_runs"))
                    row_count = count_result.scalar()
                    info["evaluation_runs_count"] = row_count
                    
                    # Try to get a sample row
                    sample_result = await session.execute(text("SELECT run_id FROM evaluation_runs LIMIT 1"))
                    sample_row = sample_result.first()
                    if sample_row:
                        info["sample_run_id"] = str(sample_row[0])
            
            return info
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {"error": str(e)}

    async def get_evaluation_run(self, run_id_str: str) -> Optional[EvaluationRun]:
        """Get an evaluation run by ID"""
        # Ensure database is initialized before proceeding
        if not self._initialized:
            logger.info("Database not initialized, initializing now")
            await self.init()
        
        # Validate and convert run_id string to UUID object
        try:
            run_id = uuid.UUID(run_id_str)
            logger.info(f"Valid UUID format: {run_id}")
        except ValueError as e:
            logger.error(f"Invalid UUID format for run_id: {run_id_str}")
            raise ValueError(f"Invalid UUID format: {run_id_str}") from e
            
        async with self.AsyncSessionLocal() as session:
            try:
                # Use the UUID object for the query
                logger.info(f"Querying database for run_id UUID: {run_id}")
                stmt = select(EvaluationRun).where(EvaluationRun.run_id == run_id)
                result = await session.execute(stmt)
                evaluation_run = result.scalar_one_or_none()
                
                if evaluation_run:
                    logger.info(f"Found evaluation run with ID {run_id_str}")
                else:
                    logger.info(f"No evaluation run found with ID {run_id_str}")
                    
                return evaluation_run
            except Exception as e:
                logger.error(f"Error getting evaluation run {run_id_str}: {e}")
                raise
    
