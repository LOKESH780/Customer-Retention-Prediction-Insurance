# 🚀 EC2 Deployment Guide

This guide will help you deploy the Customer Retention Prediction System to an AWS EC2 instance using Docker.

## Prerequisites

- AWS Account
- EC2 Instance (Ubuntu 22.04 LTS recommended)
- SSH access to your EC2 instance
- Model files (`rf_model.pkl` and `scaler.pkl`) in the project directory

## Step 1: Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance
2. **Choose AMI**: Ubuntu Server 22.04 LTS (Free tier eligible)
3. **Instance Type**: t2.micro or t3.small (minimum recommended)
4. **Key Pair**: Create or select an existing key pair
5. **Security Group**: Configure to allow:
   - SSH (port 22) from your IP
   - Custom TCP (port 8501) from anywhere (0.0.0.0/0) for Streamlit
6. **Launch Instance**

## Step 2: Connect to EC2 Instance

### Using SSH (Linux/Mac)

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Using SSH (Windows - PowerShell)

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Using PuTTY (Windows)

1. Convert `.pem` to `.ppk` using PuTTYgen
2. Use PuTTY to connect with the `.ppk` file

## Step 3: Transfer Files to EC2

### Option A: Using Git (Recommended)

```bash
# On EC2 instance
cd ~
git clone <your-repository-url>
cd Customer-Retention-Prediction-Insurance
```

### Option B: Using SCP

```bash
# From your local machine
scp -i your-key.pem -r . ubuntu@your-ec2-public-ip:/home/ubuntu/Customer-Retention-Prediction-Insurance
```

### Option C: Using AWS Systems Manager Session Manager

If you have SSM configured, you can use the AWS CLI to transfer files.

## Step 4: Install Docker on EC2

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and log back in for group changes to take effect
exit
```

Reconnect to your EC2 instance after logging out.

## Step 5: Verify Docker Installation

```bash
docker --version
docker-compose --version
```

## Step 6: Deploy Application

### Using Docker Compose (Recommended)

```bash
# Navigate to project directory
cd ~/Customer-Retention-Prediction-Insurance

# Build and start the container
docker-compose build
docker-compose up -d

# Check if container is running
docker ps

# View logs
docker-compose logs -f
```

### Using Docker CLI

```bash
# Build the image
docker build -t retention-predictor .

# Run the container
docker run -d \
  --name retention-predictor \
  -p 8501:8501 \
  --restart unless-stopped \
  retention-predictor
```

## Step 7: Access Your Application

1. **Get your EC2 public IP** from AWS Console
2. **Open browser** and navigate to: `http://your-ec2-public-ip:8501`
3. **Login** with credentials from `credentials.py`

## Step 8: Configure Auto-Start (Optional)

To ensure the application starts automatically after EC2 reboots:

```bash
# Create a systemd service
sudo nano /etc/systemd/system/retention-predictor.service
```

Add the following content:

```ini
[Unit]
Description=Customer Retention Prediction System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/Customer-Retention-Prediction-Insurance
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable retention-predictor.service
sudo systemctl start retention-predictor.service
```

## Step 9: Set Up Domain Name (Optional)

### Using Route 53

1. Create an A record pointing to your EC2 public IP
2. Update security group to allow HTTPS (port 443) if using SSL

### Using Nginx Reverse Proxy (Recommended for Production)

```bash
# Install Nginx
sudo apt-get install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/retention-predictor
```

Add configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/retention-predictor /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs

# Check if port is already in use
sudo netstat -tulpn | grep 8501

# Restart container
docker-compose restart
```

### Can't access application

1. **Check Security Group**: Ensure port 8501 is open
2. **Check Firewall**: 
   ```bash
   sudo ufw allow 8501
   ```
3. **Check Container Status**:
   ```bash
   docker ps
   docker logs retention-predictor
   ```

### Application crashes

```bash
# View detailed logs
docker-compose logs -f retention-predictor

# Check if model files exist
ls -la rf_model.pkl scaler.pkl
```

### Update Application

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

## Security Best Practices

1. **Change Default Credentials**: Update `credentials.py` before deployment
2. **Use Environment Variables**: Consider using AWS Secrets Manager for credentials
3. **Enable HTTPS**: Use Let's Encrypt with Nginx for SSL
4. **Restrict Access**: Limit security group to specific IPs if possible
5. **Regular Updates**: Keep system and Docker updated
6. **Backup**: Regularly backup model files

## Cost Optimization

- Use EC2 Spot Instances for development
- Stop instance when not in use
- Use t2.micro for testing (free tier eligible)
- Monitor CloudWatch for usage

## Monitoring

### View Application Logs

```bash
docker-compose logs -f
```

### Monitor Resource Usage

```bash
# CPU and Memory usage
docker stats retention-predictor

# System resources
htop
```

### Set Up CloudWatch (Optional)

1. Install CloudWatch agent on EC2
2. Configure log groups
3. Set up alarms for errors

## Next Steps

- Set up automated backups
- Configure CI/CD pipeline
- Add monitoring and alerting
- Set up SSL certificate
- Configure auto-scaling if needed

---

**Need Help?** Check the logs first: `docker-compose logs -f`

