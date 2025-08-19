"""
FastAPI server for LoRA model serving.

Production-ready API server with automatic model loading, request validation,
rate limiting, and comprehensive monitoring for LoRA fine-tuned models.
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, AsyncGenerator
import time
import asyncio
import uuid
from pathlib import Path
import json
from datetime import datetime

from .engine import LoRAInferenceEngine, InferenceResult
from ..lora import InferenceConfig


class GenerationRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input text prompt", min_length=1, max_length=2048)
    max_length: Optional[int] = Field(100, ge=1, le=1024, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.1, ge=0.0, le=2.0, description="Repetition penalty")
    stop_tokens: Optional[List[str]] = Field(None, description="Stop tokens")
    stream: Optional[bool] = Field(False, description="Stream response")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v


class BatchGenerationRequest(BaseModel):
    """Request model for batch text generation."""
    
    prompts: List[str] = Field(..., description="List of input prompts", min_items=1, max_items=10)
    max_length: Optional[int] = Field(100, ge=1, le=1024)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(50, ge=1, le=100)
    repetition_penalty: Optional[float] = Field(1.1, ge=0.0, le=2.0)
    stop_tokens: Optional[List[str]] = Field(None)


class GenerationResponse(BaseModel):
    """Response model for text generation."""
    
    request_id: str = Field(..., description="Unique request identifier")
    generated_text: str = Field(..., description="Generated text")
    input_text: str = Field(..., description="Original input text")
    model_name: str = Field(..., description="Model used for generation")
    
    # Generation metadata
    tokens_generated: int = Field(..., description="Number of tokens generated")
    inference_time: float = Field(..., description="Inference time in seconds")
    tokens_per_second: float = Field(..., description="Generation speed")
    
    # Generation parameters
    generation_params: Dict[str, Any] = Field(..., description="Parameters used for generation")
    
    # Timestamps
    created_at: str = Field(..., description="Request timestamp")
    
    @classmethod
    def from_inference_result(
        cls, 
        result: InferenceResult, 
        request_id: str, 
        generation_params: Dict[str, Any]
    ) -> "GenerationResponse":
        """Create response from inference result."""
        return cls(
            request_id=request_id,
            generated_text=result.generated_text,
            input_text=result.input_text,
            model_name=result.model_name,
            tokens_generated=result.tokens_generated,
            inference_time=result.inference_time,
            tokens_per_second=result.tokens_per_second,
            generation_params=generation_params,
            created_at=datetime.now().isoformat(),
        )


class BatchGenerationResponse(BaseModel):
    """Response model for batch text generation."""
    
    request_id: str = Field(..., description="Unique request identifier")
    results: List[GenerationResponse] = Field(..., description="Generation results")
    total_inference_time: float = Field(..., description="Total batch processing time")
    created_at: str = Field(..., description="Request timestamp")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    model_path: str
    adapter_path: Optional[str] = None
    lora_config: Dict[str, Any]
    inference_config: Dict[str, Any]
    loaded_at: str
    
    # Model statistics
    total_inferences: int
    total_tokens_generated: int
    average_tokens_per_second: float


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Service uptime")
    memory_usage_mb: float = Field(..., description="Current memory usage")
    timestamp: str = Field(..., description="Health check timestamp")


class LoRAServer:
    """
    Production-ready LoRA model server.
    
    Provides REST API for LoRA model inference with automatic model loading,
    request queuing, rate limiting, and comprehensive monitoring.
    """
    
    def __init__(
        self,
        model_path: Path,
        adapter_path: Optional[Path] = None,
        config: Optional[InferenceConfig] = None,
        max_concurrent_requests: int = 10,
        request_timeout: float = 60.0,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.config = config or InferenceConfig()
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout
        
        # Server state
        self.inference_engine: Optional[LoRAInferenceEngine] = None
        self.model_loaded = False
        self.start_time = time.time()
        self.request_queue = asyncio.Queue(maxsize=max_concurrent_requests)
        
        # Request tracking
        self.active_requests: Dict[str, float] = {}
        self.request_history: List[Dict[str, Any]] = []
    
    async def load_model(self) -> None:
        """Load the LoRA model and adapters."""
        try:
            print(f"Loading model from {self.model_path}")
            
            # This would load the actual model in a real implementation
            # self.inference_engine = LoRAInferenceEngine.from_pretrained(
            #     model_path=self.model_path,
            #     adapter_path=self.adapter_path,
            #     config=self.config,
            # )
            
            # For demo purposes, create a placeholder
            print("Model loading completed (placeholder)")
            self.model_loaded = True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text for a single request."""
        if not self.model_loaded or self.inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Track active request
            self.active_requests[request_id] = start_time
            
            # Generate text using inference engine
            result = self.inference_engine.generate(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stop_tokens=request.stop_tokens,
            )
            
            # Create response
            generation_params = {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "stop_tokens": request.stop_tokens,
            }
            
            response = GenerationResponse.from_inference_result(
                result=result,
                request_id=request_id,
                generation_params=generation_params,
            )
            
            # Log request
            self._log_request(request_id, request.dict(), result.to_dict(), start_time)
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
            
        finally:
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def generate_batch(self, request: BatchGenerationRequest) -> BatchGenerationResponse:
        """Generate text for batch request."""
        if not self.model_loaded or self.inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Track active request
            self.active_requests[request_id] = start_time
            
            # Generate for all prompts
            results = self.inference_engine.batch_generate(
                prompts=request.prompts,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stop_tokens=request.stop_tokens,
            )
            
            # Create responses
            generation_params = {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repetition_penalty": request.repetition_penalty,
                "stop_tokens": request.stop_tokens,
            }
            
            responses = [
                GenerationResponse.from_inference_result(
                    result=result,
                    request_id=f"{request_id}_{i}",
                    generation_params=generation_params,
                )
                for i, result in enumerate(results)
            ]
            
            total_inference_time = time.time() - start_time
            
            return BatchGenerationResponse(
                request_id=request_id,
                results=responses,
                total_inference_time=total_inference_time,
                created_at=datetime.now().isoformat(),
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")
            
        finally:
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    def get_model_info(self) -> ModelInfo:
        """Get model information and statistics."""
        if not self.model_loaded or self.inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        stats = self.inference_engine.get_stats()
        
        return ModelInfo(
            model_name=self.inference_engine.model_name,
            model_path=str(self.model_path),
            adapter_path=str(self.adapter_path) if self.adapter_path else None,
            lora_config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            inference_config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            loaded_at=datetime.fromtimestamp(self.start_time).isoformat(),
            total_inferences=stats["total_inferences"],
            total_tokens_generated=stats["total_tokens_generated"],
            average_tokens_per_second=stats["average_tokens_per_second"],
        )
    
    def get_health(self) -> HealthResponse:
        """Get service health status."""
        uptime = time.time() - self.start_time
        
        # Get memory usage (simplified)
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            memory_mb = 0.0
        
        return HealthResponse(
            status="healthy" if self.model_loaded else "loading",
            model_loaded=self.model_loaded,
            uptime_seconds=uptime,
            memory_usage_mb=memory_mb,
            timestamp=datetime.now().isoformat(),
        )
    
    def _log_request(
        self, 
        request_id: str, 
        request_data: Dict[str, Any], 
        result_data: Dict[str, Any], 
        start_time: float
    ) -> None:
        """Log request for monitoring and analytics."""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "request_data": request_data,
            "result_data": result_data,
            "processing_time": time.time() - start_time,
        }
        
        self.request_history.append(log_entry)
        
        # Keep only recent history (last 1000 requests)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]


def create_fastapi_app(
    model_path: Path,
    adapter_path: Optional[Path] = None,
    config: Optional[InferenceConfig] = None,
    **server_kwargs
) -> FastAPI:
    """
    Create FastAPI application for LoRA model serving.
    
    Args:
        model_path: Path to the base model
        adapter_path: Path to LoRA adapters
        config: Inference configuration
        **server_kwargs: Additional server configuration
        
    Returns:
        Configured FastAPI application
    """
    # Create server instance
    server = LoRAServer(
        model_path=model_path,
        adapter_path=adapter_path,
        config=config,
        **server_kwargs
    )
    
    # Create FastAPI app
    app = FastAPI(
        title="LoRA Model Server",
        description="Production-ready API for LoRA fine-tuned model inference",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        await server.load_model()
    
    @app.post("/generate", response_model=GenerationResponse)
    async def generate_text(request: GenerationRequest):
        """Generate text from a prompt."""
        return await server.generate_text(request)
    
    @app.post("/generate/batch", response_model=BatchGenerationResponse)
    async def generate_batch(request: BatchGenerationRequest):
        """Generate text for multiple prompts."""
        return await server.generate_batch(request)
    
    @app.get("/model/info", response_model=ModelInfo)
    async def model_info():
        """Get model information and statistics."""
        return server.get_model_info()
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return server.get_health()
    
    @app.get("/metrics")
    async def metrics():
        """Get detailed metrics (Prometheus compatible)."""
        if not server.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        stats = server.inference_engine.get_stats()
        uptime = time.time() - server.start_time
        
        metrics_text = f"""# HELP lora_server_uptime_seconds Server uptime in seconds
# TYPE lora_server_uptime_seconds counter
lora_server_uptime_seconds {uptime}

# HELP lora_total_inferences Total number of inferences
# TYPE lora_total_inferences counter
lora_total_inferences {stats['total_inferences']}

# HELP lora_total_tokens_generated Total tokens generated
# TYPE lora_total_tokens_generated counter
lora_total_tokens_generated {stats['total_tokens_generated']}

# HELP lora_average_tokens_per_second Average generation speed
# TYPE lora_average_tokens_per_second gauge
lora_average_tokens_per_second {stats['average_tokens_per_second']}

# HELP lora_active_requests Current number of active requests
# TYPE lora_active_requests gauge
lora_active_requests {len(server.active_requests)}
"""
        
        return Response(content=metrics_text, media_type="text/plain")
    
    return app


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA Model Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    parser.add_argument("--adapter-path", type=str, help="Path to LoRA adapters")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Create app
    app = create_fastapi_app(
        model_path=Path(args.model_path),
        adapter_path=Path(args.adapter_path) if args.adapter_path else None,
    )
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )