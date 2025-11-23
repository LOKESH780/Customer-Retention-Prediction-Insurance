#!/bin/bash

# EC2 Deployment Script for Customer Retention Prediction System
# This script helps deploy the Docker container to EC2

echo "🚀 Starting EC2 Deployment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed successfully"
else
    echo "✅ Docker is already installed"
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed successfully"
else
    echo "✅ Docker Compose is already installed"
fi

# Navigate to application directory (adjust path as needed)
APP_DIR="/home/ubuntu/Customer-Retention-Prediction-Insurance"
if [ -d "$APP_DIR" ]; then
    cd $APP_DIR
    echo "📂 Changed to application directory: $APP_DIR"
else
    echo "⚠️  Application directory not found. Please clone your repository first."
    exit 1
fi

# Build and start Docker container
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting application..."
docker-compose up -d

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if container is running
if docker ps | grep -q customer-retention-predictor; then
    echo "✅ Application is running successfully!"
    echo "🌐 Access your application at: http://$(curl -s ifconfig.me):8501"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo "✨ Deployment complete!"

