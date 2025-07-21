from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import os
from werkzeug.utils import secure_filename
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Change this to your MySQL username
    'password': '',  # Change this to your MySQL password
    'database': 'excel_data_db'
}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def get_db_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def create_database_and_table():
    """Create database and table if they don't exist"""
    try:
        # First connect without specifying database
        temp_config = DB_CONFIG.copy()
        temp_config.pop('database')
        connection = mysql.connector.connect(**temp_config)
        cursor = connection.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS excel_data_db")
        cursor.close()
        connection.close()
        
        # Now connect to the database
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Create table for storing processed data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                data_json LONGTEXT,
                upload_timestamp DATETIME,
                record_count INT,
                duplicate_count INT,
                missing_count INT
            )
        """)
        
        connection.commit()
        cursor.close()
        connection.close()
        print("Database and table created successfully")
        
    except Error as e:
        print(f"Error creating database/table: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls']

def preprocess_data(df):
    """Preprocess the dataframe - remove duplicates and handle missing values"""
    original_count = len(df)
    
    # Remove duplicates
    df_cleaned = df.drop_duplicates()
    duplicate_count = original_count - len(df_cleaned)
    
    # Count missing values
    missing_count = df_cleaned.isnull().sum().sum()
    
    # Handle missing values - you can customize this logic
    # For numeric columns: fill with mean
    # For text columns: fill with 'Unknown'
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    text_columns = df_cleaned.select_dtypes(include=['object']).columns
    
    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
    df_cleaned[text_columns] = df_cleaned[text_columns].fillna('Unknown')
    
    return df_cleaned, duplicate_count, missing_count

def generate_visualizations(df):
    """Generate various visualizations with clear explanations for non-technical users"""
    visualizations = {}
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Enhanced Data Overview with explanations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left side - Basic info
    ax1.text(0.1, 0.9, f'Your Data Summary', fontsize=18, weight='bold', color='darkblue')
    ax1.text(0.1, 0.8, f'📊 Total Records: {df.shape[0]:,}', fontsize=14)
    ax1.text(0.1, 0.75, f'📋 Total Columns: {df.shape[1]}', fontsize=14)
    
    # Data types explanation
    ax1.text(0.1, 0.65, f'Column Types:', fontsize=14, weight='bold')
    y_pos = 0.6
    for dtype, count in df.dtypes.value_counts().items():
        if 'int' in str(dtype) or 'float' in str(dtype):
            type_desc = f'🔢 Numbers: {count} columns'
        elif 'object' in str(dtype):
            type_desc = f'📝 Text: {count} columns'
        elif 'datetime' in str(dtype):
            type_desc = f'📅 Dates: {count} columns'
        else:
            type_desc = f'❓ Other: {count} columns'
        ax1.text(0.1, y_pos, type_desc, fontsize=12)
        y_pos -= 0.05
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Right side - Data quality info
    missing_total = df.isnull().sum().sum()
    complete_percentage = ((df.size - missing_total) / df.size) * 100
    
    ax2.text(0.1, 0.9, f'Data Quality Check', fontsize=18, weight='bold', color='darkgreen')
    ax2.text(0.1, 0.8, f'✅ Complete Data: {complete_percentage:.1f}%', fontsize=14, 
             color='green' if complete_percentage > 90 else 'orange' if complete_percentage > 70 else 'red')
    ax2.text(0.1, 0.75, f'❌ Missing Values: {missing_total:,}', fontsize=14,
             color='red' if missing_total > 0 else 'green')
    
    # Data completeness bar
    complete_ratio = complete_percentage / 100
    ax2.barh(0.6, complete_ratio, height=0.05, color='green', alpha=0.7)
    ax2.barh(0.6, 1-complete_ratio, left=complete_ratio, height=0.05, color='red', alpha=0.7)
    ax2.text(0.1, 0.55, 'Data Completeness:', fontsize=12, weight='bold')
    ax2.text(0.1, 0.5, f'Green = Complete, Red = Missing', fontsize=10, style='italic')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    img_buffer.seek(0)
    visualizations['overview'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # 2. Enhanced Missing values visualization
    if df.isnull().sum().sum() > 0:
        missing_by_column = df.isnull().sum()
        missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left - Bar chart of missing values by column
        bars = ax1.barh(range(len(missing_by_column)), missing_by_column.values, color='coral')
        ax1.set_yticks(range(len(missing_by_column)))
        ax1.set_yticklabels(missing_by_column.index, fontsize=10)
        ax1.set_xlabel('Number of Missing Values', fontsize=12)
        ax1.set_title('Missing Data by Column\n(Higher bars = more missing data)', fontsize=14, weight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=9)
        
        # Right - Explanation text
        ax2.text(0.1, 0.9, 'Understanding Missing Data:', fontsize=16, weight='bold', color='darkred')
        ax2.text(0.1, 0.8, '• Missing data appears as blank cells in your spreadsheet', fontsize=12, wrap=True)
        ax2.text(0.1, 0.75, '• This can happen when information was not collected', fontsize=12)
        ax2.text(0.1, 0.7, '• Or when data was lost during transfer/processing', fontsize=12)
        
        ax2.text(0.1, 0.6, 'What we did:', fontsize=14, weight='bold', color='darkblue')
        ax2.text(0.1, 0.5, '• For number columns: Filled with average values', fontsize=12)
        ax2.text(0.1, 0.45, '• For text columns: Marked as "Unknown"', fontsize=12)
        ax2.text(0.1, 0.4, '• This helps ensure all analyses can run properly', fontsize=12)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        img_buffer.seek(0)
        visualizations['missing_values'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    # 3. Enhanced Correlation Analysis with explanations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        ax_heatmap = fig.add_subplot(gs[0, 0])
        ax_explanation = fig.add_subplot(gs[0, 1])
        ax_insights = fig.add_subplot(gs[1, :])
        
        # Correlation heatmap
        correlation_matrix = df[numeric_cols].corr()
        
        # Create custom colormap explanation
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0, 
                   ax=ax_heatmap, fmt='.2f', square=True, cbar_kws={"shrink": .8})
        ax_heatmap.set_title('How Your Numbers Relate to Each Other', fontsize=14, weight='bold')
        
        # Explanation panel
        ax_explanation.text(0.1, 0.95, 'Reading the Chart:', fontsize=14, weight='bold', color='darkblue')
        ax_explanation.text(0.1, 0.85, '🔴 Red (close to 1.0):', fontsize=12, weight='bold', color='red')
        ax_explanation.text(0.1, 0.8, 'Strong positive relationship', fontsize=11)
        ax_explanation.text(0.1, 0.75, 'When one goes up, other goes up', fontsize=10, style='italic')
        
        ax_explanation.text(0.1, 0.65, '🔵 Blue (close to -1.0):', fontsize=12, weight='bold', color='blue')
        ax_explanation.text(0.1, 0.6, 'Strong negative relationship', fontsize=11)
        ax_explanation.text(0.1, 0.55, 'When one goes up, other goes down', fontsize=10, style='italic')
        
        ax_explanation.text(0.1, 0.45, '⚪ White (close to 0.0):', fontsize=12, weight='bold', color='gray')
        ax_explanation.text(0.1, 0.4, 'No clear relationship', fontsize=11)
        ax_explanation.text(0.1, 0.35, 'Changes are independent', fontsize=10, style='italic')
        
        ax_explanation.set_xlim(0, 1)
        ax_explanation.set_ylim(0, 1)
        ax_explanation.axis('off')
        
        # Generate insights
        strong_positive = []
        strong_negative = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                
                if corr_val > 0.7:
                    strong_positive.append(f"{col1} & {col2} ({corr_val:.2f})")
                elif corr_val < -0.7:
                    strong_negative.append(f"{col1} & {col2} ({corr_val:.2f})")
        
        # Insights panel
        ax_insights.text(0.02, 0.8, '🔍 Key Insights from Your Data:', fontsize=14, weight='bold', color='darkgreen')
        
        y_pos = 0.6
        if strong_positive:
            ax_insights.text(0.02, y_pos, f'📈 Strong Positive Relationships:', fontsize=12, weight='bold', color='red')
            y_pos -= 0.15
            for rel in strong_positive[:3]:  # Show top 3
                ax_insights.text(0.05, y_pos, f'• {rel}', fontsize=11)
                y_pos -= 0.1
        
        if strong_negative:
            ax_insights.text(0.52, 0.6, f'📉 Strong Negative Relationships:', fontsize=12, weight='bold', color='blue')
            y_pos_neg = 0.45
            for rel in strong_negative[:3]:  # Show top 3
                ax_insights.text(0.55, y_pos_neg, f'• {rel}', fontsize=11)
                y_pos_neg -= 0.1
        
        if not strong_positive and not strong_negative:
            ax_insights.text(0.02, y_pos, '• No strong relationships found - your data columns are mostly independent', fontsize=12)
        
        ax_insights.set_xlim(0, 1)
        ax_insights.set_ylim(0, 1)
        ax_insights.axis('off')
        
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        img_buffer.seek(0)
        visualizations['correlation'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    # 4. Enhanced Distribution Analysis
    if len(numeric_cols) > 0:
        n_cols_to_show = min(6, len(numeric_cols))  # Show max 6 columns
        cols_to_analyze = numeric_cols[:n_cols_to_show]
        
        fig = plt.figure(figsize=(18, 4 * ((n_cols_to_show + 2) // 3)))
        
        for i, col in enumerate(cols_to_analyze):
            # Create subplot
            ax = plt.subplot(((n_cols_to_show + 2) // 3), 3, i + 1)
            
            # Get statistics
            col_data = df[col].dropna()
            mean_val = col_data.mean()
            median_val = col_data.median()
            std_val = col_data.std()
            
            # Create histogram with better styling
            n, bins, patches = ax.hist(col_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add mean and median lines
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Average: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Middle: {median_val:.1f}')
            
            # Styling
            ax.set_title(f'Distribution of {col}\n(Shape of your data)', fontsize=12, weight='bold')
            ax.set_xlabel(f'{col} Values', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add interpretation text
            if abs(mean_val - median_val) / std_val < 0.1:
                interpretation = "📊 Normal distribution\n(Most values near center)"
            elif mean_val > median_val:
                interpretation = "📈 Right-skewed\n(More high values)"
            else:
                interpretation = "📉 Left-skewed\n(More low values)"
            
            ax.text(0.02, 0.98, interpretation, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add overall explanation
        if n_cols_to_show < len(numeric_cols):
            plt.figtext(0.5, 0.02, f'Showing {n_cols_to_show} of {len(numeric_cols)} number columns. Each chart shows how values are spread in that column.', 
                       ha='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        img_buffer.seek(0)
        visualizations['distributions'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
    
    return visualizations

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload Excel files only.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read Excel file
        df = pd.read_excel(filepath)
        
        # Preprocess data
        df_processed, duplicate_count, missing_count = preprocess_data(df)
        
        # Generate visualizations
        visualizations = generate_visualizations(df_processed)
        
        # Store in database
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            
            # Convert dataframe to JSON for storage - handle numpy types
            data_json = df_processed.to_json(orient='records', default_handler=str)
            
            # Convert numpy types to Python native types for database insertion
            record_count = int(len(df_processed))
            duplicate_count = int(duplicate_count)
            missing_count = int(missing_count)
            
            cursor.execute("""
                INSERT INTO processed_data 
                (filename, data_json, upload_timestamp, record_count, duplicate_count, missing_count)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (filename, data_json, datetime.now(), record_count, duplicate_count, missing_count))
            
            connection.commit()
            cursor.close()
            connection.close()
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Convert statistics to handle numpy types
        statistics = {}
        if len(df_processed.select_dtypes(include=[np.number]).columns) > 0:
            stats_df = df_processed.describe()
            statistics = convert_numpy_types(stats_df.to_dict())
        
        # Prepare response - convert numpy types
        response_data = {
            'success': True,
            'filename': filename,
            'original_rows': int(len(df)),
            'processed_rows': int(len(df_processed)),
            'columns': list(df_processed.columns),
            'duplicate_count': int(duplicate_count),
            'missing_count': int(missing_count),
            'data_preview': convert_numpy_types(df_processed.head(10).to_dict('records')),
            'statistics': statistics,
            'visualizations': visualizations
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/history')
def get_history():
    """Get processing history"""
    try:
        connection = get_db_connection()
        if not connection:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT id, filename, upload_timestamp, record_count, duplicate_count, missing_count
            FROM processed_data 
            ORDER BY upload_timestamp DESC 
            LIMIT 10
        """)
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'id': int(row[0]),  # Convert to Python int
                'filename': row[1],
                'upload_timestamp': row[2].strftime('%Y-%m-%d %H:%M:%S'),
                'record_count': int(row[3]),  # Convert to Python int
                'duplicate_count': int(row[4]),  # Convert to Python int
                'missing_count': int(row[5])  # Convert to Python int
            })
        
        cursor.close()
        connection.close()
        
        return jsonify({'history': history})
        
    except Exception as e:
        return jsonify({'error': f'Error fetching history: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize database
    create_database_and_table()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)