# 📊 Excel Data Processor

A powerful web-based application that lets users upload Excel files, preprocess them (remove duplicates, handle missing data), generate insightful visualizations, and store the results in a MySQL database — all with a clean and simple Flask interface.

---

## 🚀 Features

- ✅ Upload `.xlsx` or `.xls` files directly via a web UI.
- 📉 Automatically cleans the data by:
  - Removing duplicate entries
  - Filling missing values (mean for numbers, "Unknown" for text)
- 📊 Generates intuitive, user-friendly visualizations:
  - **Data Overview**
  - **Missing Data Heatmap**
  - **Correlation Matrix**
  - **Distribution Charts**
- 🧠 Detects skewness, data quality issues, and relationships between numeric columns.
- 💾 Stores processed data in a **MySQL** database with full history logging.
- 🔍 Provides API endpoint to view history of recent uploads and analyses.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, Jinja2 templating (via Flask)
- **Backend**: Python, Flask
- **Database**: MySQL
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: Pandas, NumPy

---

## ⚙️ Setup Instructions

### 1. 🔧 Prerequisites

- Python 3.7+
- MySQL Server installed and running locally
- Git

### 2. 🐍 Create and Activate Virtual Environment

```bash
python -m venv excel_processor_env
.\excel_processor_env\Scripts\activate    # For Windows
**### 📦 Install Required Dependencies
pip install -r requirements.txt**
📁 Project Structure
excel_data_processor/
├── app.py                  # Main Flask application
├── uploads/                # Folder to store uploaded files
├── templates/
│   └── index.html          # Web UI for uploading and viewing
├── static/                 # Optional: add custom CSS/JS here
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

🔐 Security & Limitations
Max upload size: 16MB
Only .xlsx and .xls files are accepted
Assumes local MySQL server setup
Not meant for production without Docker, Auth, and HTTPS
