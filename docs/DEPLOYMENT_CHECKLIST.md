# 🚀 EC2 Deployment Checklist

## ✅ Pre-Deployment Verification

- [x] All required files present
- [x] Dockerfile configured
- [x] docker-compose.yml ready
- [x] Model files (rf_model.pkl, scaler.pkl) included
- [x] Requirements.txt updated
- [x] README.md updated with EC2 instructions

## 📋 EC2 Deployment Steps

### Step 1: Launch EC2 Instance
- [ ] Go to AWS Console → EC2 → Launch Instance
- [ ] Choose: Ubuntu Server 22.04 LTS
- [ ] Instance Type: t2.micro or t3.small
- [ ] Create/Select Key Pair
- [ ] Configure Security Group:
  - [ ] SSH (port 22) from your IP
  - [ ] Custom TCP (port 8501) from 0.0.0.0/0
- [ ] Launch Instance

### Step 2: Connect to EC2
- [ ] Get EC2 Public IP from AWS Console
- [ ] Connect via SSH:
  ```bash
  ssh -i your-key.pem ubuntu@your-ec2-ip
  ```

### Step 3: Transfer Files to EC2

**Option A: Using Git (Recommended)**
- [ ] On EC2: `git clone <your-repo-url>`
- [ ] `cd Customer-Retention-Prediction-Insurance`

**Option B: Using SCP (from local machine)**
- [ ] Run: `scp -i your-key.pem -r . ubuntu@your-ec2-ip:/home/ubuntu/app`

### Step 4: Install Docker on EC2
- [ ] Update system: `sudo apt-get update && sudo apt-get upgrade -y`
- [ ] Install Docker:
  ```bash
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER
  ```
- [ ] Install Docker Compose:
  ```bash
  sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  ```
- [ ] Log out and log back in: `exit`

### Step 5: Deploy Application
- [ ] Navigate to project: `cd Customer-Retention-Prediction-Insurance`
- [ ] Build Docker image: `docker-compose build`
- [ ] Start container: `docker-compose up -d`
- [ ] Verify: `docker ps`
- [ ] Check logs: `docker-compose logs -f`

### Step 6: Access Application
- [ ] Open browser: `http://your-ec2-public-ip:8501`
- [ ] Login with credentials from `credentials.py`
- [ ] Test predictions

### Step 7: Configure Auto-Start (Optional)
- [ ] Create systemd service (see EC2_DEPLOYMENT.md)
- [ ] Enable service: `sudo systemctl enable retention-predictor`

## 🔒 Security Checklist

- [ ] Change default password in `credentials.py` before deployment
- [ ] Verify Security Group only allows necessary ports
- [ ] Consider setting up HTTPS with Nginx
- [ ] Set up firewall rules if needed

## 📊 Post-Deployment

- [ ] Application accessible at EC2 IP:8501
- [ ] Login works correctly
- [ ] Single prediction works
- [ ] Batch CSV upload works
- [ ] Results download works

## 🐛 Troubleshooting

If issues occur:
- [ ] Check Docker logs: `docker-compose logs -f`
- [ ] Verify Security Group allows port 8501
- [ ] Check firewall: `sudo ufw allow 8501`
- [ ] Verify container running: `docker ps`

---

**Ready to deploy!** Follow the steps above to deploy to EC2.

