# 1. Base Image: Use a lightweight Python version
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /code

# 3. Install Dependencies
# We copy requirements first to leverage Docker Cache (faster builds)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copy the Application Code
COPY ./app /code/app

# 5. Command to run the application
# host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]