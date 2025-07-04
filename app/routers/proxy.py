from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
import uuid
import os

from app.core.auth import verify_request
from app.core.chutes_manager import ChutesManager
from app.db.operations import DatabaseManager
from app.models import EmbeddingRequest, InferenceRequest

import logging

db = DatabaseManager()

logger = logging.getLogger(__name__)
chutes = ChutesManager()
router = APIRouter()

# Remove testing mode line

@router.post("/embedding")
async def embedding(request: EmbeddingRequest):
    # Validate run_id format before proceeding
    try:
        # Attempt to parse as UUID to validate format
        uuid_obj = uuid.UUID(request.run_id)
    except ValueError:
        logger.error(f"Invalid run_id format: {request.run_id}")
        raise HTTPException(status_code=400, detail="Invalid run_id format")
        
    # Add database validation
    try:
        # Ensure DB is initialized
        await db.init()
        
        evaluation_run = await db.get_evaluation_run(request.run_id)
        if not evaluation_run:
            logger.info(f"Run {request.run_id} not found in database")
            raise HTTPException(status_code=404, detail="Evaluation run not found")
                
        # Status check - required for all requests
        if evaluation_run.status != "sandbox_created":
            logger.info(f"Embedding for {request.run_id} was requested but evaluation run is not in sandbox_created state")
            raise HTTPException(status_code=400, detail="Evaluation run is not in the sandbox_created state")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except SQLAlchemyError as e:
        # Handle database errors specifically
        logger.error(f"Database error for run_id {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error for run_id {request.run_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # For all other exceptions, log and stop execution
        logger.error(f"Error during validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during validation")
    
    # Rest of the embedding function
    try:
        embedding = await chutes.embed(request.run_id, request.input)
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get embedding due to internal server error")

@router.post("/inference") 
async def inference(request: InferenceRequest):
    # Validate run_id format before proceeding
    try:
        # Attempt to parse as UUID to validate format
        uuid_obj = uuid.UUID(request.run_id)
    except ValueError:
        logger.error(f"Invalid run_id format: {request.run_id}")
        raise HTTPException(status_code=400, detail="Invalid run_id format")
        
    # Add database validation
    try:
        # Ensure DB is initialized
        await db.init()
        
        evaluation_run = await db.get_evaluation_run(request.run_id)
        if not evaluation_run:
            logger.info(f"Run {request.run_id} not found in database")
            raise HTTPException(status_code=404, detail="Evaluation run not found")
                
        # Status check - required for all requests
        if evaluation_run.status != "sandbox_created":
            logger.info(f"Inference for {request.run_id} was requested but evaluation run is not in sandbox_created state")
            raise HTTPException(status_code=400, detail="Evaluation run is not in the sandbox_created state")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except SQLAlchemyError as e:
        # Handle database errors specifically
        logger.error(f"Database error for run_id {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error for run_id {request.run_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # For all other exceptions, log and stop execution
        logger.error(f"Error during validation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during validation")
    
    # Rest of the inference function
    try:
        response = await chutes.inference(
            request.run_id,
            request.messages,
            request.temperature,
            request.model
        )
        return response
    except Exception as e:
        logger.error(f"Error getting inference for {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get inference due to internal server error")

router = APIRouter()

routes = [
    ("/embedding", embedding),
    ("/inference", inference),
]

for path, endpoint in routes:
    router.add_api_route(
        path,
        endpoint,
        tags=["agents"],
        dependencies=[Depends(verify_request)],
        methods=["POST"]
    )
