"""
Web interface for StatWhy.

Provides a FastAPI-based web application with HTML templates and API endpoints
for statistical verification.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Request, HTTPException, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from .core import StatWhyEngine
from .models import TestType, VerificationRequest
from .utils import load_data, validate_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="StatWhy Web Interface",
    description="Formally verified statistical hypothesis testing",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Ensure directories exist
templates_dir.mkdir(parents=True, exist_ok=True)
static_dir.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize StatWhy engine
engine = StatWhyEngine()


# API Models
class VerificationResponse:
    def __init__(
        self,
        success: bool,
        message: str,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        self.success = success
        self.message = message
        self.result = result
        self.error = error

    def dict(self):
        return {
            "success": self.success,
            "message": self.message,
            "result": self.result,
            "error": self.error,
        }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with file upload and test selection."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "test_types": [t.value for t in TestType]}
    )


@app.get("/api/tests")
async def get_supported_tests():
    """Get list of supported statistical tests."""
    try:
        tests = engine.get_supported_tests()
        return {
            "success": True,
            "tests": [
                {
                    "id": t.value,
                    "name": t.value.upper(),
                    "description": f"{t.value.upper()} statistical test",
                }
                for t in tests
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get supported tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify")
async def verify_test(
    test_type: str = Form(...),
    alpha: float = Form(0.05),
    timeout: int = Form(300),
    data_file: UploadFile = File(...),
):
    """
    Verify a statistical procedure.

    Args:
        test_type: Type of statistical test
        alpha: Significance level
        timeout: Verification timeout in seconds
        data_file: Uploaded data file

    Returns:
        Verification result
    """
    try:
        # Validate test type
        try:
            test_enum = TestType(test_type)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Unsupported test type: {test_type}"
            )

        # Save uploaded file temporarily
        temp_file = Path(f"/tmp/statwhy_{uuid.uuid4()}_{data_file.filename}")
        try:
            with open(temp_file, "wb") as f:
                content = await data_file.read()
                f.write(content)

            # Load and validate data
            data_df = load_data(temp_file)
            validate_data(data_df, test_enum)

            # Create verification request
            request = VerificationRequest(
                test_type=test_enum, data=data_df, alpha=alpha, timeout=timeout
            )

            # Perform verification
            result = engine.verify(request)

            # Convert result to dict for JSON response
            result_dict = result.dict()

            return VerificationResponse(
                success=result.is_verified,
                message=(
                    "Verification completed successfully"
                    if result.is_verified
                    else "Verification failed"
                ),
                result=result_dict,
            )

        finally:
            # Clean up temporary file
            if temp_file.exists():
                temp_file.unlink()

    except Exception as e:
        logger.error(f"Verification error: {e}")
        return VerificationResponse(
            success=False, message="Verification failed", error=str(e)
        )


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "supported_tests": len(engine.get_supported_tests()),
    }


@app.post("/api/clear-cache")
async def clear_cache():
    """Clear the verification cache."""
    try:
        engine.cache.clear()
        return {"success": True, "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples")
async def examples_page(request: Request):
    """Examples and tutorials page."""
    return templates.TemplateResponse("examples.html", {"request": request})


@app.get("/docs")
async def docs_page(request: Request):
    """Documentation page."""
    return templates.TemplateResponse("docs.html", {"request": request})


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
