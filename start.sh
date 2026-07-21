#!/bin/bash

# Replace PORT_PLACEHOLDER with the port Render provides
sed -i "s/PORT_PLACEHOLDER/${PORT:-80}/g" /etc/nginx/nginx.conf

# Start FastAPI backend (sets Host to 127.0.0.1 and Port to 8000)
export HOST=127.0.0.1
export PORT=8000
python main.py &

# Start Streamlit frontend
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 &

# Start Nginx in foreground to keep container running
nginx -g "daemon off;"
