from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 1. API Metadata
    PROJECT_NAME: str = "Smurf Detection Engine"
    API_V1_STR: str = "/api/v1"
    
    # 2. Security
    # In production, this would be a long, secret hash.
    # We use this to prevent unauthorized people from hitting your API.
    API_KEY: str = "secret-key-12345"
    
    # 3. Redis Configuration (The Memory)
    # 'localhost' works when running python main.py locally.
    # 'redis' is the hostname we will use inside Docker (defined in docker-compose).
    REDIS_HOST: str = "localhost" 
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # 4. Model Configuration
    # Path to the trained brain.
    MODEL_PATH: str = "app/models/xgb_smurf_v1.json"
    
    # 5. Business Logic Thresholds (Tunable parameters)
    # Probability above which we auto-block
    BLOCK_THRESHOLD: float = 0.90  
    # Probability above which we flag for review
    REVIEW_THRESHOLD: float = 0.65 

    class Config:
        # This tells Pydantic to read from a .env file if it exists
        case_sensitive = True

# Create a single instance to be imported elsewhere
settings = Settings()