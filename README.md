# 📈 Customer Retention Prediction System

A machine learning web application for predicting insurance customer retention ratios using Random Forest regression.

## 🎯 Overview

This application helps insurance companies predict customer retention ratios by analyzing key business metrics. It provides both single predictions and batch processing capabilities through an intuitive web interface.

## ✨ Features

- **Single Prediction**: Predict retention ratio for individual customers
- **Batch Processing**: Upload CSV files for bulk predictions
- **Interactive Dashboard**: View predictions with key metrics
- **Export Results**: Download predictions as CSV
- **Secure Access**: Login-protected application

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, AWS EC2

## 📁 Project Structure

```
Customer-Retention-Prediction-Insurance/
│
├── src/                    # Application source code
│   ├── app.py             # Main Streamlit application
│   ├── login.py           # Authentication module
│   └── credentials.py     # Login credentials
│
├── models/                 # Machine learning models
│   ├── rf_model.pkl       # Trained Random Forest model
│   └── scaler.pkl         # Feature scaler
│
├── assets/                 # Static assets
│   └── wallpaper.png     # Background image
│
├── scripts/                # Deployment scripts
│   ├── deploy_to_ec2.py
│   ├── deploy_to_existing_ec2.py
│   ├── deploy-ec2.sh
│   └── verify-deployment.ps1
│
├── docs/                   # Documentation
│   ├── EC2_DEPLOYMENT.md
│   └── DEPLOYMENT_CHECKLIST.md
│
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── README.md             # This file (Application Usage Guide)
└── DEPLOYMENT.md         # Deployment Guide
```

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Customer-Retention-Prediction-Insurance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - Login with default credentials:
     - **Username:** `admin`
     - **Password:** `password123`

⚠️ **Important**: Change these credentials in `src/credentials.py` before deploying to production!

## 📖 How to Use the Application

### 1. Login

1. Open the application in your browser
2. Enter your username and password
3. Click "Login"
4. You'll be redirected to the main application page

### 2. Single Prediction (Manual Input)

Use this mode to predict retention ratio for a single customer:

1. **Select "Manual Input"** from the input method options
2. **Fill in the customer data:**
   - **Agency Appointment Year**: Year when the agency was appointed (e.g., 2020)
   - **Written Premium Amount**: Total written premium amount in dollars (e.g., 150000)
   - **New Business Premium Amount**: New business written premium (e.g., 75000)
   - **Policies Inforce Quantity**: Current number of policies in force (e.g., 120)
   - **Previous Policies Inforce Quantity**: Previous period policies in force (e.g., 110)
   - **Loss Ratio**: Ratio of losses to premiums (e.g., 0.65)
   - **Growth Rate (3-Year)**: Three-year growth rate (e.g., 0.15)
   - **Active Producers**: Number of active insurance producers (e.g., 5)
3. **Click "Predict 🔮"**
4. **View the result**: The predicted retention ratio will be displayed

### 3. Batch Processing (CSV Upload)

Use this mode to predict retention ratios for multiple customers at once:

1. **Select "Upload CSV File"** from the input method options
2. **Prepare your CSV file** with the following columns:
   - `AGENCY_APPOINTMENT_YEAR`
   - `WRTN_PREM_AMT`
   - `NB_WRTN_PREM_AMT`
   - `POLY_INFORCE_QTY`
   - `PREV_POLY_INFORCE_QTY`
   - `LOSS_RATIO`
   - `GROWTH_RATE_3YR`
   - `ACTIVE_PRODUCERS`

3. **Upload the CSV file** using the file uploader
4. **View results**:
   - Summary statistics (Total Records, Average Retention, Min/Max)
   - Complete predictions table
5. **Download results**: Click "📥 Download Predictions as CSV" to save results

### 4. Sample CSV Format

Here's a sample CSV file you can use for testing:

```csv
AGENCY_APPOINTMENT_YEAR,WRTN_PREM_AMT,NB_WRTN_PREM_AMT,POLY_INFORCE_QTY,PREV_POLY_INFORCE_QTY,LOSS_RATIO,GROWTH_RATE_3YR,ACTIVE_PRODUCERS
2020,150000.00,75000.00,120.00,110.00,0.65,0.15,5.00
2018,200000.00,100000.00,150.00,140.00,0.58,0.20,8.00
2019,175000.00,85000.00,135.00,125.00,0.72,0.12,6.00
```

### 5. Logout

Click the "🚪 Logout" button in the top right corner to log out of the application.

## 🤖 Understanding the Model

### Model Information

The application uses a **Random Forest** regression model trained on historical insurance customer data. The model analyzes 8 key features to predict retention ratios.

### Features Explained

1. **AGENCY_APPOINTMENT_YEAR**: Year when the insurance agency was appointed
2. **WRTN_PREM_AMT**: Total written premium amount (revenue)
3. **NB_WRTN_PREM_AMT**: New business written premium amount
4. **POLY_INFORCE_QTY**: Current number of active policies
5. **PREV_POLY_INFORCE_QTY**: Previous period's active policies
6. **LOSS_RATIO**: Ratio of losses to premiums (lower is better)
7. **GROWTH_RATE_3YR**: Three-year compound growth rate
8. **ACTIVE_PRODUCERS**: Number of active insurance producers/agents

### Interpreting Results

- **Retention Ratio**: A value between 0 and 1 (or 0% to 100%)
  - Higher values indicate better retention probability
  - Values above 0.8 (80%) suggest excellent retention
  - Values below 0.4 (40%) indicate high churn risk

## 🔧 Configuration

### Changing Login Credentials

Edit `src/credentials.py`:

```python
CREDENTIALS = {
    "username": "your_username",
    "password": "your_secure_password"
}
```

### Running with Docker (Local)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🐛 Troubleshooting

### Application won't start

- **Check Python version**: Ensure Python 3.11+ is installed
- **Check dependencies**: Run `pip install -r requirements.txt`
- **Check model files**: Ensure `models/rf_model.pkl` and `models/scaler.pkl` exist

### Predictions not working

- **Verify model files**: Check that model files are in the `models/` directory
- **Check input format**: Ensure all required fields are filled
- **Check CSV format**: Verify CSV has all required columns

### Login issues

- **Verify credentials**: Check `src/credentials.py` for correct username/password
- **Clear browser cache**: Try clearing cookies and cache

## 📊 Example Use Cases

1. **Risk Assessment**: Identify customers at risk of churning
2. **Portfolio Analysis**: Analyze entire customer portfolios
3. **Strategic Planning**: Make data-driven retention decisions
4. **Performance Monitoring**: Track retention trends over time

## 🔒 Security Notes

- ⚠️ **Change default credentials** before production use
- 🔐 Use strong passwords
- 📝 Keep credentials secure and never commit them to version control
- 🌐 Consider using HTTPS in production

## 📝 Requirements

See `requirements.txt` for all Python dependencies:
- streamlit
- joblib
- pandas
- numpy
- scikit-learn
- requests

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

For detailed deployment guides, see the `docs/` directory.

---

**Happy Predicting! 🎯**
