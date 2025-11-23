# Deployment Verification Script for Windows
Write-Host "🔍 Verifying Deployment Package..." -ForegroundColor Cyan
Write-Host ""

$errors = @()
$warnings = @()

# Check required files
$requiredFiles = @("app.py", "login.py", "credentials.py", "requirements.txt", "Dockerfile", "docker-compose.yml", "rf_model.pkl", "scaler.pkl", "wallpaper.png")

Write-Host "📁 Checking required files..." -ForegroundColor Yellow
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "  ❌ $file - MISSING!" -ForegroundColor Red
        $errors += $file
    }
}

# Check model files
Write-Host ""
Write-Host "🤖 Checking model files..." -ForegroundColor Yellow
if (Test-Path "rf_model.pkl") {
    $size = (Get-Item "rf_model.pkl").Length / 1MB
    Write-Host "  ✅ rf_model.pkl ($([math]::Round($size, 2)) MB)" -ForegroundColor Green
} else {
    $errors += "rf_model.pkl"
}

# Check credentials
Write-Host ""
Write-Host "🔐 Checking security..." -ForegroundColor Yellow
if (Test-Path "credentials.py") {
    $content = Get-Content "credentials.py" -Raw
    if ($content -match "password123") {
        Write-Host "  ⚠️  Default password detected - Change before production!" -ForegroundColor Yellow
        $warnings += "Using default password"
    }
}

# Summary
Write-Host ""
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
if ($errors.Count -eq 0) {
    Write-Host "✅ All checks passed! Ready for deployment." -ForegroundColor Green
    if ($warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "⚠️  Warnings:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  - $warning" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "📋 Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Transfer files to EC2 instance" -ForegroundColor White
    Write-Host "  2. SSH into EC2 instance" -ForegroundColor White
    Write-Host "  3. Install Docker (see EC2_DEPLOYMENT.md)" -ForegroundColor White
    Write-Host "  4. Run: docker-compose up -d" -ForegroundColor White
} else {
    Write-Host "❌ Deployment check failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Missing files:" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  - $error" -ForegroundColor Red
    }
}
