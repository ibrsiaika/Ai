"""
REST API for India's GPT model inference.
Provides endpoints for text generation, classification, and embeddings.
"""

import torch
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from models.gpt_model import GPTForCausalLM, GPTConfig
from inference.generator import TextGenerator, GenerationConfig
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt")
    max_length: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    do_sample: bool = Field(default=True)
    num_samples: int = Field(default=1, ge=1, le=5)


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    prompt: str
    generated_texts: List[str]
    model_info: Dict


class EmbeddingRequest(BaseModel):
    """Request model for embeddings."""
    texts: List[str] = Field(..., description="List of texts")


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    embeddings: List[List[float]]
    model_info: Dict


class SimilarityRequest(BaseModel):
    """Request model for similarity."""
    text1: str
    text2: str


class SimilarityResponse(BaseModel):
    """Response model for similarity."""
    text1: str
    text2: str
    similarity: float
    model_info: Dict


class ModelAPI:
    """API server for India's GPT model."""
    
    def __init__(self,
                 model_path: str,
                 tokenizer_name: str = 'gpt2',
                 device: str = 'cuda'):
        """
        Initialize API.
        
        Args:
            model_path: Path to saved model
            tokenizer_name: Tokenizer name
            device: Device to use
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.config = GPTConfig()
        self.model = GPTForCausalLM(self.config).to(self.device)
        
        try:
            self.model.load_state_dict(
                torch.load(f"{model_path}/model.pt", map_location=self.device)
            )
        except:
            logger.warning("Could not load model weights, using randomly initialized model")
        
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.generator = TextGenerator(self.model, self.tokenizer, device=str(self.device))
        
        logger.info("Model loaded successfully")
    
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text."""
        try:
            config = GenerationConfig(
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=request.do_sample
            )
            
            generated_texts = []
            for _ in range(request.num_samples):
                text = self.generator.generate(request.prompt, config)
                generated_texts.append(text)
            
            return GenerateResponse(
                prompt=request.prompt,
                generated_texts=generated_texts,
                model_info={
                    'device': str(self.device),
                    'model_path': self.model_path
                }
            )
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Get embeddings."""
        try:
            embeddings = []
            
            with torch.no_grad():
                for text in request.texts:
                    input_ids = self.tokenizer.encode(
                        text, return_tensors='pt'
                    ).to(self.device)
                    
                    outputs = self.model(input_ids=input_ids)
                    hidden_states = outputs['hidden_states']
                    
                    # Mean pooling
                    embedding = hidden_states.mean(dim=1).cpu().tolist()[0]
                    embeddings.append(embedding)
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model_info={
                    'device': str(self.device),
                    'embedding_dim': len(embeddings[0]) if embeddings else 0
                }
            )
        
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def compute_similarity(self, request: SimilarityRequest) -> SimilarityResponse:
        """Compute similarity between texts."""
        try:
            with torch.no_grad():
                # Get embeddings
                embedding_req = EmbeddingRequest(texts=[request.text1, request.text2])
                embedding_resp = self.get_embeddings(embedding_req)
                
                # Compute cosine similarity
                emb1 = torch.tensor(embedding_resp.embeddings[0])
                emb2 = torch.tensor(embedding_resp.embeddings[1])
                
                # Normalize
                emb1 = emb1 / emb1.norm()
                emb2 = emb2 / emb2.norm()
                
                # Cosine similarity
                similarity = (emb1 * emb2).sum().item()
            
            return SimilarityResponse(
                text1=request.text1,
                text2=request.text2,
                similarity=similarity,
                model_info={'device': str(self.device)}
            )
        
        except Exception as e:
            logger.error(f"Similarity error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


def create_app(model_path: str = './outputs/best_model') -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="India's GPT Model API",
        description="REST API for India's first AI model",
        version="1.0.0"
    )
    
    # Initialize model API
    api = ModelAPI(model_path=model_path)
    
    @app.get("/")
    def root():
        """Root endpoint."""
        return {
            "name": "India's GPT Model API",
            "version": "1.0.0",
            "endpoints": [
                "/docs",
                "/generate",
                "/embeddings",
                "/similarity",
                "/health"
            ]
        }
    
    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "device": str(api.device),
            "model_loaded": True
        }
    
    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        """Generate text endpoint."""
        return api.generate(request)
    
    @app.post("/embeddings", response_model=EmbeddingResponse)
    def embeddings(request: EmbeddingRequest):
        """Get embeddings endpoint."""
        return api.get_embeddings(request)
    
    @app.post("/similarity", response_model=SimilarityResponse)
    def similarity(request: SimilarityRequest):
        """Compute similarity endpoint."""
        return api.compute_similarity(request)
    
    return app


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create app
    app = create_app(model_path='./outputs/best_model')
    
    # Run server
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=8000,
        log_level='info'
    )
