FROM python:3.13-slim

WORKDIR /app

# Install uv for fast resolution
RUN pip install uv

# Copy the app files
COPY . /app

# Setup venv and install dependencies
RUN uv venv && uv pip install -r requirements.txt

# We need to expose the port that Cloud Run expects (8080)
EXPOSE 8080

ENV PORT=8080
ENV HOST=0.0.0.0

# Command to run the application
CMD ["/app/.venv/bin/uvicorn", "task_tripletex.service:app", "--host", "0.0.0.0", "--port", "8080"]
