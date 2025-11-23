"""
Deploy to existing EC2 instance via SSH
Usage: python deploy_to_existing_ec2.py <ec2_ip> <key_file_path>
"""
import paramiko
import sys
import os
from pathlib import Path
import time

def deploy_to_ec2(ec2_ip, key_file_path, username='ubuntu'):
    """Deploy application to EC2 instance via SSH"""
    
    print("=" * 60)
    print("🚀 Deploying to EC2 Instance")
    print("=" * 60)
    print(f"📍 Target: {ec2_ip}")
    print(f"👤 User: {username}")
    print(f"🔑 Key: {key_file_path}")
    print()
    
    # Check if key file exists
    if not os.path.exists(key_file_path):
        print(f"❌ Key file not found: {key_file_path}")
        return False
    
    try:
        # Create SSH client
        print("🔌 Connecting to EC2 instance...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect
        ssh.connect(
            hostname=ec2_ip,
            username=username,
            key_filename=key_file_path,
            timeout=30
        )
        print("✅ Connected successfully!")
        print()
        
        # Commands to execute
        commands = [
            # Update system
            "echo '📦 Updating system packages...'",
            "sudo apt-get update -y",
            
            # Install Docker
            "echo '🐳 Installing Docker...'",
            "if ! command -v docker &> /dev/null; then curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh && sudo usermod -aG docker $USER; fi",
            
            # Install Docker Compose
            "echo '🐳 Installing Docker Compose...'",
            "if ! command -v docker-compose &> /dev/null; then sudo curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose && sudo chmod +x /usr/local/bin/docker-compose; fi",
            
            # Create app directory
            "echo '📁 Creating application directory...'",
            "mkdir -p ~/retention-predictor",
            "cd ~/retention-predictor",
            
            # Create Dockerfile
            "echo '📝 Creating Dockerfile...'",
            '''cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
EOF''',
            
            # Create docker-compose.yml
            "echo '📝 Creating docker-compose.yml...'",
            '''cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  retention-predictor:
    build: .
    container_name: customer-retention-predictor
    ports:
      - "8501:8501"
    restart: unless-stopped
EOF''',
            
            # Create requirements.txt
            "echo '📝 Creating requirements.txt...'",
            '''cat > requirements.txt << 'EOF'
streamlit
joblib
pandas
numpy
scikit-learn
requests
EOF''',
            
            # Create directory structure
            "echo '📁 Creating directory structure...'",
            "mkdir -p src models assets",
        ]
        
        # Execute commands
        for cmd in commands:
            print(f"▶️  Executing: {cmd[:50]}...")
            stdin, stdout, stderr = ssh.exec_command(cmd)
            exit_status = stdout.channel.recv_exit_status()
            
            if exit_status != 0:
                error = stderr.read().decode()
                print(f"⚠️  Warning: {error}")
            else:
                output = stdout.read().decode()
                if output.strip():
                    print(f"   {output.strip()}")
        
        print()
        print("📤 Transferring application files...")
        
        # Transfer files using SCP
        sftp = ssh.open_sftp()
        
        # Transfer source files
        src_files = ['app.py', 'login.py', 'credentials.py']
        for file in src_files:
            src_path = f'src/{file}'
            if os.path.exists(src_path):
                print(f"  📤 Transferring {src_path}...")
                sftp.put(src_path, f'/home/{username}/retention-predictor/src/{file}')
                print(f"  ✅ {file} transferred")
            else:
                print(f"  ⚠️  {src_path} not found, skipping...")
        
        # Transfer model files
        model_files = ['rf_model.pkl', 'scaler.pkl']
        for file in model_files:
            model_path = f'models/{file}'
            if os.path.exists(model_path):
                print(f"  📤 Transferring {model_path}...")
                sftp.put(model_path, f'/home/{username}/retention-predictor/models/{file}')
                print(f"  ✅ {file} transferred")
            else:
                print(f"  ⚠️  {model_path} not found, skipping...")
        
        # Transfer assets
        if os.path.exists('assets/wallpaper.png'):
            print(f"  📤 Transferring assets/wallpaper.png...")
            sftp.put('assets/wallpaper.png', f'/home/{username}/retention-predictor/assets/wallpaper.png')
            print(f"  ✅ wallpaper.png transferred")
        
        sftp.close()
        
        # Build and start Docker container
        print()
        print("🔨 Building Docker image...")
        stdin, stdout, stderr = ssh.exec_command(
            "cd ~/retention-predictor && docker-compose build"
        )
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("✅ Docker image built successfully")
        else:
            error = stderr.read().decode()
            print(f"❌ Build failed: {error}")
            return False
        
        print()
        print("🚀 Starting application...")
        stdin, stdout, stderr = ssh.exec_command(
            "cd ~/retention-predictor && docker-compose up -d"
        )
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            print("✅ Application started successfully")
        else:
            error = stderr.read().decode()
            print(f"❌ Start failed: {error}")
            return False
        
        # Wait a bit for app to start
        print()
        print("⏳ Waiting for application to start...")
        time.sleep(10)
        
        # Check if container is running
        stdin, stdout, stderr = ssh.exec_command("docker ps | grep retention-predictor")
        output = stdout.read().decode()
        if 'retention-predictor' in output:
            print("✅ Container is running!")
        else:
            print("⚠️  Container status unclear, check manually")
        
        print()
        print("=" * 60)
        print("✅ DEPLOYMENT COMPLETE!")
        print("=" * 60)
        print(f"🌐 Access your application at: http://{ec2_ip}:8501")
        print(f"🔐 Login credentials: Check credentials.py")
        print()
        print("📋 Useful commands:")
        print(f"  ssh -i {key_file_path} {username}@{ec2_ip}")
        print("  docker-compose logs -f  # View logs")
        print("  docker-compose restart  # Restart app")
        print("=" * 60)
        
        ssh.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python deploy_to_existing_ec2.py <ec2_ip> <key_file_path> [username]")
        print("Example: python deploy_to_existing_ec2.py 54.123.45.67 C:\\path\\to\\key.pem")
        sys.exit(1)
    
    ec2_ip = sys.argv[1]
    key_file = sys.argv[2]
    username = sys.argv[3] if len(sys.argv) > 3 else 'ubuntu'
    
    success = deploy_to_ec2(ec2_ip, key_file, username)
    sys.exit(0 if success else 1)

