"""
EC2 Deployment Script using boto3
This script will deploy the application to EC2
"""
import boto3
import time
import paramiko
from pathlib import Path
import os

# AWS Credentials - Use environment variables or AWS credentials file
# Never hardcode credentials in scripts!
# Set these as environment variables:
# export AWS_ACCESS_KEY_ID="your-access-key"
# export AWS_SECRET_ACCESS_KEY="your-secret-key"
# export AWS_DEFAULT_REGION="us-east-1"

import os

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError(
        "AWS credentials not found! Please set AWS_ACCESS_KEY_ID and "
        "AWS_SECRET_ACCESS_KEY environment variables, or configure AWS CLI."
    )

# Initialize AWS clients
ec2 = boto3.client(
    'ec2',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION
)

def check_existing_instances():
    """Check for existing EC2 instances"""
    print("🔍 Checking for existing EC2 instances...")
    try:
        response = ec2.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}
            ]
        )
        
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append({
                    'id': instance['InstanceId'],
                    'state': instance['State']['Name'],
                    'ip': instance.get('PublicIpAddress', 'N/A'),
                    'type': instance['InstanceType']
                })
        
        return instances
    except Exception as e:
        print(f"❌ Error checking instances: {e}")
        return []

def create_security_group():
    """Create security group for the application"""
    print("🔒 Creating security group...")
    try:
        # Check if security group exists
        sgs = ec2.describe_security_groups(
            Filters=[{'Name': 'group-name', 'Values': ['retention-predictor-sg']}]
        )
        
        if sgs['SecurityGroups']:
            sg_id = sgs['SecurityGroups'][0]['GroupId']
            print(f"✅ Security group already exists: {sg_id}")
            return sg_id
        
        # Create new security group
        response = ec2.create_security_group(
            GroupName='retention-predictor-sg',
            Description='Security group for Customer Retention Prediction System'
        )
        sg_id = response['GroupId']
        
        # Add rules
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'SSH'}]
                },
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8501,
                    'ToPort': 8501,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0', 'Description': 'Streamlit'}]
                }
            ]
        )
        
        print(f"✅ Security group created: {sg_id}")
        return sg_id
    except Exception as e:
        print(f"❌ Error creating security group: {e}")
        return None

def launch_ec2_instance(sg_id):
    """Launch a new EC2 instance"""
    print("🚀 Launching EC2 instance...")
    try:
        # Get Ubuntu 22.04 AMI
        # Canonical's AWS account ID for Ubuntu AMIs
        # You can find the latest owner ID in AWS documentation
        amis = ec2.describe_images(
            Owners=['099720109477'],  # Canonical (public AMI owner)
            Filters=[
                {'Name': 'name', 'Values': ['ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*']},
                {'Name': 'state', 'Values': ['available']}
            ]
        )
        
        if not amis['Images']:
            print("❌ No Ubuntu AMI found")
            return None
        
        # Get latest AMI
        latest_ami = sorted(amis['Images'], key=lambda x: x['CreationDate'], reverse=True)[0]
        ami_id = latest_ami['ImageId']
        print(f"📦 Using AMI: {ami_id}")
        
        # Launch instance
        response = ec2.run_instances(
            ImageId=ami_id,
            MinCount=1,
            MaxCount=1,
            InstanceType='t2.micro',
            SecurityGroupIds=[sg_id],
            KeyName='retention-predictor-key',  # You'll need to create this
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': 'Customer-Retention-Predictor'},
                        {'Key': 'Project', 'Value': 'Retention-Prediction'}
                    ]
                }
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        print(f"✅ Instance launched: {instance_id}")
        return instance_id
    except Exception as e:
        print(f"❌ Error launching instance: {e}")
        return None

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 EC2 Deployment Script")
    print("=" * 50)
    
    # Check existing instances
    instances = check_existing_instances()
    if instances:
        print("\n📋 Existing instances found:")
        for inst in instances:
            print(f"  - {inst['id']} ({inst['state']}) - IP: {inst['ip']}")
        use_existing = input("\nUse existing instance? (y/n): ").lower()
        if use_existing == 'y' and instances:
            instance_id = instances[0]['id']
            print(f"✅ Using existing instance: {instance_id}")
        else:
            # Create security group and launch new instance
            sg_id = create_security_group()
            if sg_id:
                instance_id = launch_ec2_instance(sg_id)
            else:
                print("❌ Failed to create security group")
                exit(1)
    else:
        # Create security group and launch new instance
        sg_id = create_security_group()
        if sg_id:
            instance_id = launch_ec2_instance(sg_id)
        else:
            print("❌ Failed to create security group")
            exit(1)
    
    print("\n✅ Deployment script ready!")
    print("Note: You'll need to create a key pair and configure SSH access")

