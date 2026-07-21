FROM python:3.11-slim

# Install system dependencies, including Nginx and compiler tools
RUN apt-get update && apt-get install -y \
    build-essential \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Copy nginx config template to standard nginx configuration path
COPY nginx.conf /etc/nginx/nginx.conf

# Make startup script executable
RUN chmod +x start.sh

# Expose the default port (Render will override or set $PORT)
EXPOSE 80

# Run the startup script
CMD ["./start.sh"]
