# Roadmap to Production

Here is the exact Roadmap to Production and the Code for the critical missing pieces.

## 1. 🏗️ The "Production Gap" Checklist

To go from MVP to Enterprise Service, you need to fix these 4 layers:

| Layer | Current MVP Status | Production Requirement |
|-------|-------------------|------------------------|
| Data Layer | Reads static transactions.csv (Slow, Stale). | PostgreSQL / TimescaleDB (Live, Indexed). |
| Security | Open Endpoint (Anyone can call). | API Key Authentication (Strict Access). |
| Performance | Logic runs on every request (High Latency). | Feature Store (Redis) (Pre-computed features). |
| Deployment | Running via python main.py terminal. | Docker Container + Gunicorn (Reliable). |

## 2. 💻 Code Implementation: Closing the Gap

We will tackle the two most critical upgrades: Connecting to a Real Database and Dockerizing.

### Upgrade 1: Replace CSV with PostgreSQL

In production, transactions live in a SQL database. We need to swap your "Mock Loader" for a real SQL connection.

#### New File: `modules/Smurfing_detection/app/database.py`

This handles the connection pool.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Get DB URL from env or use default (Production Requirement)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/aml_db")

# Create Connection Pool (Critical for high concurrency)
engine = create_engine(
    DATABASE_URL,
    pool_size=20,      # Handle 20 concurrent DB connections
    max_overflow=10,
    pool_pre_ping=True # Auto-reconnect if DB connection drops
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for DB Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

#### Update File: `modules/Smurfing_detection/app/routes/inference.py`

Update the get_user_history function to query SQL instead of reading CSV.

```python
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from app.database import engine # Import the engine we just made
import pandas as pd

# ... (Previous imports remain)

# --- PRODUCTION DATA LOADER ---
def get_user_history_sql(user_id: str):
    """
    Fetches raw transaction history from PostgreSQL.
    Optimized to fetch only necessary columns for feature engineering.
    """
    query = text("""
        SELECT 
            transaction_id, user_id, timestamp, amount, 
            type, device_id, ip_address, related_user_id
        FROM transactions 
        WHERE user_id = :uid 
        AND timestamp > NOW() - INTERVAL '60 days' -- Optimization: Only fetch relevant window
        ORDER BY timestamp ASC
    """)
    
    try:
        # pd.read_sql is convenient but blocking. 
        # For <10k rows it's fine. For massive scale, use asyncpg.
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"uid": user_id})
        
        if df.empty:
            return None
            
        # Type enforcement
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = df['amount'].astype(float)
        return df
        
    except Exception as e:
        print(f"DB Error: {e}")
        raise e

# Update the endpoint to use this new function
@router.post("/analyze_user")
async def analyze_user(request: AnalysisRequest):
    # Swap 'get_user_history_mock' with 'get_user_history_sql'
    history_df = get_user_history_sql(request.user_id)
    # ... (Rest of the logic remains exactly the same)
```

### Upgrade 2: Containerization (Docker)

You cannot ask your client to "install python and pip install requirements". You give them a Docker Image.

#### New File: `modules/Smurfing_detection/Dockerfile`

This effectively "freezes" your environment so it runs exactly the same on your server as it does on your laptop.

```dockerfile
# Use a lightweight, official Python runtime
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# 1. Install System Dependencies (Needed for numpy/scipy)
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# 2. Copy Requirements first (Docker Layer Caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the Source Code
# Copy the entire module structure
COPY . .

# 4. Set Environment Variables
ENV PYTHONPATH=/app
ENV MODULE_NAME=app.main
ENV VARIABLE_NAME=app

# 5. The Command to Run (Production Server)
# We use Gunicorn with Uvicorn workers for concurrency
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
```

## 3. 🚀 How to "Go Live"

### Database Setup

You need a real Postgres database running.

1. Run the schema migration (create the transactions table in Postgres).
2. Load your transactions.csv into that table using a simple SQL script.

### Build Docker

```bash
docker build -t smurf-hunter:v1 modules/Smurfing_detection/
```

### Run Docker

```bash
docker run -p 8000:8000 --env DATABASE_URL="postgresql://..." smurf-hunter:v1
```