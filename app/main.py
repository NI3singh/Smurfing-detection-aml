from fastapi import FastAPI
from app.routes.inference import router as inference_router

app = FastAPI(title="Smurf Hunter AI")

app.include_router(inference_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)