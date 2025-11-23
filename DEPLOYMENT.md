# 🚀 Deployment Guide - Customer Retention Prediction System

Complete guide for deploying the Customer Retention Prediction System to AWS EC2.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Deployment](#quick-deployment)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Post-Deployment](#post-deployment)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

Before deploying, ensure you have:

- ✅ AWS Account with EC2 access
- ✅ AWS CLI installed and configured
- ✅ SSH key pair (or create one during deployment)
- ✅ Docker knowledge (basic)
- ✅ All project files in the repository

## Quick Deployment

### Automated Deployment (Recommended)

If you have an existing EC2 instance:

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Deploy to existing EC2 instance
python scripts/deploy_to_existing_ec2.py <EC2_IP> <KEY_FILE_PATH>
```

### Full Automated Deployment (Create + Deploy)

The deployment process I performed:

1. **Exported AWS credentials**
2. **Created security group** with ports 22 (SSH) and 8501 (Streamlit)
3. **Created key pair** for SSH access
4. **Launched EC2 instance** (Ubuntu 22.04, t2.micro)
5. **Installed Docker** on EC2
6. **Deployed application** using Docker Compose
7. **Verified deployment**

## Step-by-Step Deployment

### Step 1: Set Up AWS Credentials

```bash

# configure AWS CLI
aws configure set aws_access_key_id YOUR_ACCESS_KEY
aws configure set aws_secret_access_key YOUR_SECRET_KEY
aws configure set default.region us-east-1
```

### Step 2: Create Security Group

```bash
# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)

# Create security group
SG_ID=$(aws ec2 create-security-group \
  --group-name retention-predictor-sg \
  --description "Security group for Customer Retention Prediction System" \
  --vpc-id $VPC_ID \
  --query "GroupId" --output text)

# Add SSH rule (port 22)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

# Add Streamlit rule (port 8501)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 8501 --cidr 0.0.0.0/0
```

### Step 3: Create Key Pair

```bash
# Create key pair
aws ec2 create-key-pair \
  --key-name retention-predictor-key \
  --query "KeyMaterial" \
  --output text > retention-predictor-key.pem

# Set permissions (Linux/Mac)
chmod 400 retention-predictor-key.pem
```

### Step 4: Launch EC2 Instance

```bash
# Get latest Ubuntu 22.04 AMI
# Note: 099720109477 is Canonical's AWS account ID for public Ubuntu AMIs
AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
           "Name=state,Values=available" \
  --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
  --output text)

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --count 1 \
  --instance-type t2.micro \
  --key-name retention-predictor-key \
  --security-group-ids $SG_ID \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=Customer-Retention-Predictor}]" \
  --query "Instances[0].InstanceId" \
  --output text)

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text)

echo "Instance IP: $PUBLIC_IP"
```

### Step 5: Deploy Application

Use the automated deployment script:

```bash
python scripts/deploy_to_existing_ec2.py $PUBLIC_IP retention-predictor-key.pem ubuntu
```

Or deploy manually:

```bash
# SSH into instance
ssh -i retention-predictor-key.pem ubuntu@$PUBLIC_IP

# On EC2 instance:
# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Clone or transfer project files
git clone <your-repo-url>
cd Customer-Retention-Prediction-Insurance

# 5. Build and start
sudo docker-compose build
sudo docker-compose up -d
```

## Post-Deployment

### Verify Deployment

1. **Check container status:**
   ```bash
   ssh -i retention-predictor-key.pem ubuntu@$PUBLIC_IP
   sudo docker ps
   ```

2. **Check application logs:**
   ```bash
   cd ~/retention-predictor
   sudo docker-compose logs -f
   ```

3. **Test application:**
   - Open browser: `http://$PUBLIC_IP:8501`
   - Login with credentials from `src/credentials.py`
   - Test predictions

### Access Information

- **Application URL**: `http://<EC2_PUBLIC_IP>:8501`
- **SSH Access**: `ssh -i retention-predictor-key.pem ubuntu@<EC2_PUBLIC_IP>`
- **Default Login**:
  - Username: `admin`
  - Password: `password123`

⚠️ **Change default credentials before production use!**

## Troubleshooting

### Application Not Accessible

**Issue**: Can't access `http://<IP>:8501`

**Solutions**:
1. Check security group allows port 8501
2. Verify EC2 instance is running
3. Check Docker container status:
   ```bash
   sudo docker ps
   ```
4. Check application logs:
   ```bash
   sudo docker-compose logs
   ```

### Container Keeps Restarting

**Issue**: Container status shows "Restarting"

**Solutions**:
1. Check logs for errors:
   ```bash
   sudo docker-compose logs --tail=50
   ```
2. Verify all files are present:
   ```bash
   ls -la ~/retention-predictor/
   ```
3. Rebuild container:
   ```bash
   sudo docker-compose down
   sudo docker-compose build --no-cache
   sudo docker-compose up -d
   ```

### Files Not Found in Container

**Issue**: Error "File does not exist: app.py"

**Solutions**:
1. Verify Dockerfile copies files correctly
2. Check file paths in Dockerfile
3. Rebuild with no cache:
   ```bash
   sudo docker-compose build --no-cache
   ```

### Port Already in Use

**Issue**: Port 8501 already in use

**Solutions**:
1. Find process using port:
   ```bash
   sudo lsof -i :8501
   ```
2. Stop conflicting service
3. Or change port in `docker-compose.yml`

## Maintenance

### Updating the Application

```bash
# SSH into EC2
ssh -i retention-predictor-key.pem ubuntu@<IP>

# Pull latest changes
cd ~/retention-predictor
git pull

# Rebuild and restart
sudo docker-compose down
sudo docker-compose build
sudo docker-compose up -d
```

### Viewing Logs

```bash
# Real-time logs
sudo docker-compose logs -f

# Last 50 lines
sudo docker-compose logs --tail=50
```

### Restarting Application

```bash
sudo docker-compose restart
```

### Stopping Application

```bash
sudo docker-compose down
```

### Backup

```bash
# Backup model files
scp -i retention-predictor-key.pem \
  ubuntu@<IP>:~/retention-predictor/models/*.pkl \
  ./backup/
```

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│         AWS EC2 Instance                 │
│  ┌───────────────────────────────────┐  │
│  │      Docker Container             │  │
│  │  ┌─────────────────────────────┐ │  │
│  │  │  Streamlit Application      │ │  │
│  │  │  - app.py                   │ │  │
│  │  │  - login.py                 │ │  │
│  │  │  - credentials.py           │ │  │
│  │  └─────────────────────────────┘ │  │
│  │  ┌─────────────────────────────┐ │  │
│  │  │  ML Models                  │ │  │
│  │  │  - rf_model.pkl            │ │  │
│  │  │  - scaler.pkl               │ │  │
│  │  └─────────────────────────────┘ │  │
│  └───────────────────────────────────┘  │
│         Port 8501 (Streamlit)            │
└─────────────────────────────────────────┘
           ↓
    Security Group
    (Ports 22, 8501)
           ↓
      Internet
```

## Cost Optimization

- Use **t2.micro** for testing (free tier eligible)
- Stop instance when not in use
- Use **Spot Instances** for development
- Monitor CloudWatch for usage

## Security Best Practices

1. ✅ Change default credentials
2. ✅ Restrict security group to specific IPs if possible
3. ✅ Use strong passwords
4. ✅ Enable HTTPS with reverse proxy (Nginx)
5. ✅ Regular security updates
6. ✅ Backup model files regularly

## Next Steps

- Set up domain name with Route 53
- Configure Nginx reverse proxy
- Enable SSL with Let's Encrypt
- Set up CloudWatch monitoring
- Configure auto-scaling if needed

## Deployment Summary

**What was deployed:**
- ✅ EC2 Instance: Ubuntu 22.04 LTS (t2.micro)
- ✅ Security Group: Ports 22, 8501
- ✅ Docker & Docker Compose installed
- ✅ Application containerized and running
- ✅ Application accessible at `http://<IP>:8501`

**Deployment Time**: ~10-15 minutes

**Instance Details**:
- Instance ID: `<YOUR_INSTANCE_ID>` (get from AWS Console)
- Public IP: `<YOUR_PUBLIC_IP>` (get from AWS Console, may change on restart)
- Region: `us-east-1` (or your preferred region)

---

For application usage instructions, see [README.md](README.md)

For detailed EC2 deployment steps, see `docs/EC2_DEPLOYMENT.md`

