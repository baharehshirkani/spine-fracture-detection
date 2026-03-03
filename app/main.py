"""
Spine Fracture Detection API

A FastAPI application for detecting cervical spine fractures (C1-C7) from CT scan DICOM images.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import pydicom
import pylibjpeg  # Required for DICOM decompression
import numpy as np
from io import BytesIO
from pathlib import Path
import os

from .model import load_model, get_model_type_from_path
from .utils import preprocess_single_slice, preprocess_3channel, apply_window

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_3_channel_2.5D.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERTEBRAE = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

# ============================================================
# Initialize FastAPI
# ============================================================
app = FastAPI(
    title="Spine Fracture Detection API",
    description="Detect cervical spine fractures (C1-C7) from CT scan DICOM images",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Load Model on Startup
# ============================================================
model = None
model_type = None

@app.on_event("startup")
async def load_model_on_startup():
    global model, model_type
    
    if not Path(MODEL_PATH).exists():
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}")
        print("  API will run but predictions won't work until model is provided.")
        return
    
    model_type = get_model_type_from_path(MODEL_PATH)
    model = load_model(MODEL_PATH, model_type, str(DEVICE))
    print(f"✓ Model loaded: {MODEL_PATH}")
    print(f"✓ Model type: {model_type}")
    print(f"✓ Device: {DEVICE}")

# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
def root():
    """API information endpoint."""
    return {
        "name": "Spine Fracture Detection API",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": str(DEVICE),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict fractures from a DICOM file.
    
    Upload a single .dcm file and receive fracture predictions for C1-C7 vertebrae.
    
    Returns:
        predictions: Probability for each vertebra (C1-C7)
        fractures_detected: List of vertebrae with probability > 0.5
        patient_overall: Maximum probability across all vertebrae
    """
    # Check model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model file exists."
        )
    
    # Validate file type
    if not file.filename.endswith('.dcm'):
        raise HTTPException(
            status_code=400,
            detail="Only .dcm (DICOM) files are accepted"
        )
    
    try:
        # Read DICOM file
        contents = await file.read()
        dcm = pydicom.dcmread(BytesIO(contents))
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Preprocess based on model type
        if model_type == "single":
            image = preprocess_single_slice(pixel_array)
        else:
            # For 3-channel model with single input, duplicate the slice
            image = preprocess_3channel([pixel_array, pixel_array, pixel_array])
        
        image = image.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            output = model(image)
            probs = torch.sigmoid(output).cpu().numpy()[0]
        
        # Format results
        predictions = {v: float(p) for v, p in zip(VERTEBRAE, probs)}
        fractures = [v for v, p in zip(VERTEBRAE, probs) if p > 0.5]
        
        return JSONResponse(content={
            "predictions": predictions,
            "fractures_detected": fractures,
            "patient_overall": float(probs.max()),
            "model_type": model_type
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing DICOM file: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict fractures from multiple DICOM files (3 slices for 2.5D model).
    
    Upload 3 consecutive .dcm files for better accuracy with the 2.5D model.
    Files should be ordered: [slice_before, middle_slice, slice_after]
    
    Returns:
        predictions: Probability for each vertebra (C1-C7)
        fractures_detected: List of vertebrae with probability > 0.5
        patient_overall: Maximum probability across all vertebrae
    """
    # Check model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model file exists."
        )
    
    if model_type != "3channel":
        raise HTTPException(
            status_code=400,
            detail="Batch prediction only available for 3-channel model"
        )
    
    if len(files) != 3:
        raise HTTPException(
            status_code=400,
            detail="Exactly 3 DICOM files required for 2.5D prediction"
        )
    
    try:
        # Read all DICOM files
        pixel_arrays = []
        for file in files:
            if not file.filename.endswith('.dcm'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only .dcm files accepted. Got: {file.filename}"
                )
            contents = await file.read()
            dcm = pydicom.dcmread(BytesIO(contents))
            pixel_arrays.append(dcm.pixel_array.astype(np.float32))
        
        # Preprocess
        image = preprocess_3channel(pixel_arrays)
        image = image.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            output = model(image)
            probs = torch.sigmoid(output).cpu().numpy()[0]
        
        # Format results
        predictions = {v: float(p) for v, p in zip(VERTEBRAE, probs)}
        fractures = [v for v, p in zip(VERTEBRAE, probs) if p > 0.5]
        
        return JSONResponse(content={
            "predictions": predictions,
            "fractures_detected": fractures,
            "patient_overall": float(probs.max()),
            "model_type": model_type,
            "num_slices": len(files)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing DICOM files: {str(e)}"
        )


# ============================================================
# Run with: uvicorn app.main:app --reload
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
