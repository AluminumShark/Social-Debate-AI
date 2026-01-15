# Social Debate AI Deployment Guide

*English | [](#chinese-version)*

This guide explains how to deploy Social Debate AI to production environments.

## Deployment Options

### 1. Local Deployment
- Suitable for personal use and testing
- Simplest deployment method

### 2. Cloud Deployment
- AWS EC2
- Google Cloud Platform
- Azure
- Heroku

### 3. Docker Deployment
- Containerized deployment
- Easy to scale

## Local Production Deployment

### 1. Environment Setup

```bash
# Create production environment
python -m venv venv_prod
source venv_prod/bin/activate  # Linux/Mac
# or
venv_prod\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install gunicorn  # Production server
```

### 2. Environment Variables Configuration

Create `.env.production` file:

```bash
# API Keys
OPENAI_API_KEY=your-production-key

# Flask configuration
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key

# Security settings
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax
```

### 3. Running with Gunicorn

```bash
# Basic run
gunicorn -w 4 -b 0.0.0.0:5000 ui.app:app

# Recommended configuration
gunicorn \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  ui.app:app
```

### 4. Using Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (future feature)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Static files
    location /static {
        alias /path/to/Social_Debate_AI/ui/static;
        expires 1d;
    }
}
```

## AWS EC2 Deployment

### 1. Create EC2 Instance

```bash
# Recommended configuration
- Instance type: t3.medium (minimum) or t3.large (recommended)
- OS: Ubuntu 20.04 LTS
- Storage: 20GB SSD
- Security group: Open ports 80, 443, 22
```

### 2. Initialize Server

```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install necessary software
sudo apt install python3.8 python3-pip nginx git -y
```

### 3. Deploy Application

```bash
# Clone project
git clone https://github.com/your-username/Social_Debate_AI.git
cd Social_Debate_AI

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Setup environment variables
cp .env.example .env
nano .env  # Edit configuration
```

### 4. Setup Systemd Service

Create `/etc/systemd/system/social-debate.service`:

```ini
[Unit]
Description=Social Debate AI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Social_Debate_AI
Environment="PATH=/home/ubuntu/Social_Debate_AI/venv/bin"
ExecStart=/home/ubuntu/Social_Debate_AI/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 ui.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Start service:

```bash
sudo systemctl start social-debate
sudo systemctl enable social-debate
sudo systemctl status social-debate
```

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run command
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ui.app:app"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

### 3. Build and Run

```bash
# Build image
docker-compose build

# Run containers
docker-compose up -d

# View logs
docker-compose logs -f
```

##  Security Configuration

### 1. HTTPS Configuration

Using Let's Encrypt free certificates:

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto renewal
sudo certbot renew --dry-run
```

### 2. Firewall Setup

```bash
# UFW firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 3. Environment Variable Security

```bash
# Using python-dotenv
from dotenv import load_dotenv
load_dotenv('.env.production')

# Encrypt sensitive information
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

## Monitoring and Logging

### 1. Application Logging

```python
# Configure logging in app.py
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/social-debate.log',
        maxBytes=10240000,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    app.logger.addHandler(file_handler)
```

### 2. System Monitoring

```bash
# Use htop to monitor resources
sudo apt install htop
htop

# Use netdata for comprehensive monitoring
bash <(curl -Ss https://my-netdata.io/kickstart.sh)
```

### 3. Error Tracking

Consider integrating Sentry:

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

## Performance Optimization

### 1. Cache Configuration

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@cache.cached(timeout=300)
def expensive_function():
    pass
```

### 2. Database Optimization

If using database for storing debate records:

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    pool_recycle=3600
)
```

### 3. CDN Configuration

Use Cloudflare or other CDN services to accelerate static resources.

## Scaling Strategy

### 1. Horizontal Scaling

```nginx
# Nginx load balancing
upstream app_servers {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    location / {
        proxy_pass http://app_servers;
    }
}
```

### 2. Using Redis for Shared State

```python
from redis import Redis
redis_client = Redis(host='localhost', port=6379, db=0)

# Store debate state
redis_client.set(f'debate:{debate_id}', json.dumps(debate_state))
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Increase swap space
   - Optimize model loading
   - Use larger instances

2. **Response Timeout**
   - Increase Gunicorn timeout
   - Optimize model inference
   - Use asynchronous processing

3. **Concurrency Issues**
   - Use Redis locks
   - Implement request queuing
   - Limit concurrent connections

## Deployment Checklist

- [ ] Environment variables configured completely
- [ ] HTTPS certificate installed
- [ ] Firewall rules set up
- [ ] Logging system working properly
- [ ] Backup strategy implemented
- [ ] Monitoring system enabled
- [ ] Error tracking configured
- [ ] Performance testing completed

---

**Tip**: Please validate all configurations in a test environment before deployment!

---

## Chinese Version

# Social Debate AI 

*[English](#social-debate-ai-deployment-guide) | *

 Social Debate AI 

## 

### 1. 
- 
- 

### 2. 
- AWS EC2
- Google Cloud Platform
- Azure
- Heroku

### 3. Docker 
- 
- 

## 

### 1. 

```bash
# 
python -m venv venv_prod
source venv_prod/bin/activate  # Linux/Mac
# 
venv_prod\Scripts\activate  # Windows

# 
pip install -r requirements.txt
pip install gunicorn  # 
```

### 2. 

 `.env.production` 

```bash
# API Keys
OPENAI_API_KEY=your-production-key

# Flask 
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key

# 
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax
```

### 3.  Gunicorn 

```bash
# 
gunicorn -w 4 -b 0.0.0.0:5000 ui.app:app

# 
gunicorn \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  ui.app:app
```

### 4.  Nginx 

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # 
    location /static {
        alias /path/to/Social_Debate_AI/ui/static;
        expires 1d;
    }
}
```

## AWS EC2 

### 1.  EC2 

```bash
# 
- : t3.medium ()  t3.large ()
- : Ubuntu 20.04 LTS
- : 20GB SSD
- :  80, 443, 22 
```

### 2. 

```bash
#  EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# 
sudo apt update && sudo apt upgrade -y

# 
sudo apt install python3.8 python3-pip nginx git -y
```

### 3. 

```bash
# 
git clone https://github.com/your-username/Social_Debate_AI.git
cd Social_Debate_AI

# 
python3 -m venv venv
source venv/bin/activate

# 
pip install -r requirements.txt
pip install gunicorn

# 
cp .env.example .env
nano .env  # 
```

### 4.  Systemd 

 `/etc/systemd/system/social-debate.service`:

```ini
[Unit]
Description=Social Debate AI
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Social_Debate_AI
Environment="PATH=/home/ubuntu/Social_Debate_AI/venv/bin"
ExecStart=/home/ubuntu/Social_Debate_AI/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 ui.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```



```bash
sudo systemctl start social-debate
sudo systemctl enable social-debate
sudo systemctl status social-debate
```

## Docker 

### 1.  Dockerfile

```dockerfile
FROM python:3.8-slim

WORKDIR /app

# 
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# 
COPY . .

# 
EXPOSE 5000

# 
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "ui.app:app"]
```

### 2.  docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
    restart: unless-stopped
```

### 3. 

```bash
# 
docker-compose build

# 
docker-compose up -d

# 
docker-compose logs -f
```

##  

### 1. HTTPS 

 Let's Encrypt 

```bash
#  Certbot
sudo apt install certbot python3-certbot-nginx

# 
sudo certbot --nginx -d your-domain.com

# 
sudo certbot renew --dry-run
```

### 2. 

```bash
# UFW 
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 3. 

```bash
#  python-dotenv
from dotenv import load_dotenv
load_dotenv('.env.production')

# 
from cryptography.fernet import Fernet
key = Fernet.generate_key()
cipher_suite = Fernet(key)
```

## 

### 1. 

```python
#  app.py 
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/social-debate.log',
        maxBytes=10240000,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    app.logger.addHandler(file_handler)
```

### 2. 

```bash
#  htop 
sudo apt install htop
htop

#  netdata 
bash <(curl -Ss https://my-netdata.io/kickstart.sh)
```

### 3. 

 Sentry

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)
```

## 

### 1. 

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@cache.cached(timeout=300)
def expensive_function():
    pass
```

### 2. 



```python
# 
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    pool_recycle=3600
)
```

### 3. CDN 

 Cloudflare  CDN 

## 

### 1. 

```nginx
# Nginx 
upstream app_servers {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    location / {
        proxy_pass http://app_servers;
    }
}
```

### 2.  Redis 

```python
from redis import Redis
redis_client = Redis(host='localhost', port=6379, db=0)

# 
redis_client.set(f'debate:{debate_id}', json.dumps(debate_state))
```

## 

### 

1. ****
   - 
   - 
   - 

2. ****
   -  Gunicorn timeout
   - 
   - 

3. ****
   -  Redis 
   - 
   - 

## 

- [ ] 
- [ ] HTTPS 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 

---

**** 