# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements from root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code (needed by Streamlit)
COPY agent /app/agent

# Copy frontend code
COPY frontend/app.py .

# Expose Streamlit port
ENV PORT=8501
EXPOSE 8501

# Run Streamlit
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0
