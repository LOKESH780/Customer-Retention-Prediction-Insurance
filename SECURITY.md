# 🔒 Security Guidelines

## Important Security Notes

### ⚠️ Never Commit Credentials

**DO NOT** commit the following to the repository:

- AWS Access Keys
- AWS Secret Keys
- AWS Account IDs
- SSH Private Keys (.pem files)
- Instance IDs
- Public IPs
- Passwords
- API Keys
- Any sensitive configuration

### ✅ Safe Practices

1. **Use Environment Variables**
   ```bash
   export AWS_ACCESS_KEY_ID="your-key"
   export AWS_SECRET_ACCESS_KEY="your-secret"
   ```

2. **Use AWS Credentials File**
   ```bash
   aws configure
   # Credentials stored in ~/.aws/credentials
   ```

3. **Use .env Files** (and add to .gitignore)
   ```bash
   # .env file (not committed)
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   ```

4. **Use AWS IAM Roles** (for EC2 instances)
   - Attach IAM roles to EC2 instances
   - No need to store credentials

### 📝 Configuration Files

All sensitive values should use placeholders:
- `<YOUR_ACCESS_KEY>` instead of actual keys
- `<YOUR_INSTANCE_ID>` instead of actual IDs
- `<YOUR_PUBLIC_IP>` instead of actual IPs

### 🔐 Application Credentials

Change default credentials in `src/credentials.py` before deployment:
```python
CREDENTIALS = {
    "username": "your_secure_username",
    "password": "your_secure_password"
}
```

### 🛡️ Security Checklist

Before committing:
- [ ] No AWS credentials in code
- [ ] No AWS credentials in documentation
- [ ] No SSH keys in repository
- [ ] No instance IDs or IPs in docs
- [ ] Default passwords changed
- [ ] .env files in .gitignore
- [ ] All sensitive files excluded

---

**Remember**: If credentials are ever committed, rotate them immediately!

