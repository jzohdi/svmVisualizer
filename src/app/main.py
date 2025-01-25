import hashlib
import json
from fastapi import FastAPI, Depends, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .database import engine, Base
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import get_db
from .models import SvmResult
from .svm import train_svm_from_data_then_update_db, ModelMethod
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os

# Initialize logging (optional)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

# Configure logging to write to a file
# Configure rotating file handler
handler = RotatingFileHandler(
    "error.log", maxBytes=5 * 1024 * 1024, backupCount=3  # 5 MB per file, keep 3 backups
)
logging.basicConfig(
    level=logging.ERROR,
    handlers=[handler],
    format="%(asctime)s - %(levelname)s - %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown: any cleanup can go here

app = FastAPI(lifespan=lifespan)

# CORS settings
origins = [
    "https://svm-visualizer.fly.dev",  # Replace with the URL serving your HTML page
    "http://localhost",     # Allow local testing
    "http://localhost:8000" # For local backend access
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Specify allowed origins
    allow_credentials=True,          # Allow cookies or authentication
    allow_methods=["*"],             # Allow all HTTP methods
    allow_headers=["*"],             # Allow all headers
)

# Dynamically determine the absolute path to the `static` directory
current_file_path = os.path.dirname(__file__)  # Directory of `main.py`
static_directory = os.path.join(current_file_path, "static")
templates_directory = os.path.join(current_file_path, "templates")

if not os.path.exists(static_directory):
    raise RuntimeError(f"Directory '{static_directory}' does not exist")

# Mount the static directory
app.mount("/static", StaticFiles(directory=static_directory), name="static")

@app.exception_handler(Exception)
async def internal_server_error_handler(request: Request, exc: Exception):
    """
    Custom handler for internal server errors (HTTP 500).
    Logs the error and returns a custom JSON response.
    """
    logger.error(f"Unhandled exception: {exc}")
    logger.error("500 error")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path},
    )

@app.get("/", response_class=HTMLResponse)
async def read_html():
    index_file_path = os.path.join(templates_directory, "index.html")
    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Template not found: {index_file_path}")
    
    with open(index_file_path, "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/about", response_class=HTMLResponse)
async def read_html():
    index_file_path = os.path.join(templates_directory, "about.html")
    if not os.path.exists(index_file_path):
        raise FileNotFoundError(f"Template not found: {index_file_path}")
    
    with open(index_file_path, "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

testing_description ="""
data to test training model against.
must be the same number of dimensions as training data.
will return results as a list of labels the same length as training data.
"""


class TrainingRequest(BaseModel):
    training_data: List[List[float]] = Field(..., min_length=1)
    labels: List[str] = Field(..., min_length=1)
    testing_data: Optional[List[List[float]]] = Field(
        default=None, description=testing_description, min_length=1)
    method: ModelMethod

class TrainingStatus(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"

class SvmResultResponse(BaseModel):
    id: str
    method: str
    testing_data: str
    result: str
    confidence: Optional[str]
    score: Optional[float]
    params: Optional[str]

class TrainingResponse(BaseModel):
    id: str
    status: TrainingStatus
    result: Optional[SvmResultResponse] = None


def generate_unique_id(training_data, labels, method):
    # Combine inputs into a consistent string representation
    input_str = json.dumps({
        'training_data': training_data, 
        'labels': labels, 
        'method': method
    }, sort_keys=True)
    
    # Create SHA-256 hash
    return hashlib.sha256(input_str.encode()).hexdigest()

@app.get("/api/train_result/{id}", response_model=TrainingResponse)
async def train_model(
    id: str,
    db: AsyncSession = Depends(get_db)
):
    
    # Check if record already exists
    existing_record = await db.execute(
        select(SvmResult).filter(SvmResult.id == id)
    )

    record = existing_record.scalar_one_or_none();

    if record is not None:
        return {
            "id": id,
            "status": "complete",
            "result": {
                "id": record.id,
                "method": record.method,
                "testing_data": record.test_data,
                "result": record.result,
                "confidence": record.confidence,
                "score": record.score,
                "params": record.params
            }
        }
    return {
        "id": id,
        "status": "pending"
    }

@app.post("/api/train_model", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
        # Generate unique ID
    unique_id = generate_unique_id(
        request.training_data, 
        request.labels, 
        request.method
    )
    
    # Check if record already exists
    existing_record = await db.execute(
        select(SvmResult).filter(SvmResult.id == unique_id)
    )

    record = existing_record.scalar_one_or_none();

    if record is not None:
        return {
            "id": unique_id,
            "status": "complete",
            "result": {
                "id": record.id,
                "method": record.method,
                "testing_data": record.test_data,
                "result": record.result,
                "confidence": record.confidence,
                "score": record.score,
                "params": record.params
            }
        }

    # Schedule background task
    background_tasks.add_task(
        train_svm_from_data_then_update_db,
        unique_id=unique_id,
        training_data=request.training_data,
        testing_data=request.testing_data, 
        labels=request.labels,
        method=request.method,
        db_session_factory=get_db
    )


    return {
        "id": unique_id, 
        "status": "pending"
    }

# If using uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)