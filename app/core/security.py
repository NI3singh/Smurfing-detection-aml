from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

# Define the expected header name. 
# The Bank must send a header like: "X-API-Key: secret-key-12345"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency function to validate the API Key.
    This acts as a gatekeeper for your routes.
    """
    
    # 1. Check if key is missing
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials: Missing API Key"
        )

    # 2. Check if key is correct (Constant time comparison to prevent timing attacks is ideal, 
    # but simple equality is fine for this stage)
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
        
    return api_key