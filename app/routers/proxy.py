from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import verify_request
from app.core.chutes_manager import ChutesManager
from app.db.operations import DatabaseManager
from app.models import EmbeddingRequest, InferenceRequest

import logging

db = DatabaseManager()

logger = logging.getLogger(__name__)

chutes = ChutesManager()

async def embedding(request: EmbeddingRequest):
    try:
        # Check if this run_id is valid
        evaluation_run = None
        try:
            evaluation_run = await db.get_evaluation_run(request.run_id)
        except Exception as e:
            logger.warning(f"Database error when fetching evaluation run: {e}")
            # For testing purposes, proceed without database validation
            if request.run_id == "test-run-id":
                logger.info(f"Allowing test run ID: {request.run_id}")
            else:
                logger.error(f"Could not validate run_id: {request.run_id}, error: {e}")
                raise HTTPException(status_code=500, detail="Database error: Could not validate run_id")
        
        if evaluation_run is None and request.run_id != "test-run-id":
            logger.info(f"Embedding for {request.run_id} was requested but no such evaluation run was found in our database")
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        
        if evaluation_run and evaluation_run.status != "sandbox_created":
            logger.info(f"Embedding for {request.run_id} was requested but the evaluation run is not in the sandbox_created state")
            raise HTTPException(status_code=400, detail="Evaluation run is not in the sandbox_created state")

        embedding = await chutes.embed(request.run_id, request.input)
        logger.debug(f"Embedding for {request.run_id} was requested and returned")
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding for {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding: {str(e)}")

async def inference(request: InferenceRequest):
    try:
        # For testing purposes, allow test run ID without database validation
        if request.run_id != "test-run-id":
            try:
                evaluation_run = await db.get_evaluation_run(request.run_id)
                if not evaluation_run:
                    logger.info(f"Inference for {request.run_id} was requested but no such evaluation run was found in our database")
                    raise HTTPException(status_code=404, detail="Evaluation run not found")
                
                if evaluation_run.status != "sandbox_created":
                    logger.info(f"Inference for {request.run_id} was requested but the evaluation run is not in the sandbox_created state")
                    raise HTTPException(status_code=400, detail="Evaluation run is not in the sandbox_created state")
            except Exception as e:
                logger.warning(f"Database error when fetching evaluation run: {e}")
                # Only allow test run IDs to proceed when database validation fails
                if request.run_id != "test-run-id":
                    raise HTTPException(status_code=500, detail="Database error: Could not validate run_id")

        response = await chutes.inference(
            request.run_id, 
            request.messages,
            request.temperature,
            request.model
        )
        logger.debug(f"Inference for {request.run_id} was requested and returned \"{response}\"")
        return response
    except Exception as e:
        logger.error(f"Error getting inference for {request.run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get inference: {str(e)}")

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
