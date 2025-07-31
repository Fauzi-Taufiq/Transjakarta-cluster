from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Load model dan data
def load_models():
    try:
        # Load data clustering results first
        df = pd.read_csv('transjakarta_clustering_results.csv')
        print("✓ Data loaded successfully")
        
        # Try to load model clustering
        try:
            with open('transjakarta_density_model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Model file corrupted or incompatible. Using data-based clustering instead.")
            print(f"  Error details: {e}")
            model = None
        
        # Try to load scaler
        try:
            with open('scaler_transjakarta.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Scaler file corrupted or incompatible. Using data-based clustering instead.")
            print(f"  Error details: {e}")
            scaler = None
        
        if model is None and scaler is None:
            print("ℹ Info: Running in data-only mode. Predictions will use existing cluster data.")
        
        return model, scaler, df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None, None

# Load models saat startup
model, scaler, df = load_models()

# Get unique values untuk dropdown
def get_unique_values():
    if df is not None:
        koridor_asal = sorted(df['corridorName'].unique().tolist())
        waktu_kategori = sorted(df['timeCategory'].unique().tolist())
        return koridor_asal, waktu_kategori
    return [], []

@app.route('/')
def index():
    koridor_asal, waktu_kategori = get_unique_values()
    
    # Check application status
    app_status = {
        'data_loaded': df is not None,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'mode': 'full' if model is not None and scaler is not None else 'data_only'
    }
    
    return render_template('index.html', 
                         koridor_asal=koridor_asal,
                         waktu_kategori=waktu_kategori,
                         app_status=app_status)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        koridor_asal = data['koridor_asal']
        koridor_tujuan = data['koridor_tujuan']
        waktu = data['waktu']
        
        if df is None:
            return jsonify({'error': 'Data tidak tersedia'}), 500
        
        # Filter data berdasarkan input
        filtered_data = df[
            (df['corridorName'] == koridor_asal) &
            (df['corridorName'] == koridor_tujuan) &
            (df['timeCategory'] == waktu)
        ]
        
        if filtered_data.empty:
            # Jika tidak ada data yang cocok, gunakan data yang paling mirip
            filtered_data = df[
                (df['corridorName'] == koridor_asal) &
                (df['timeCategory'] == waktu)
            ].head(1)
        
        if filtered_data.empty:
            return jsonify({'error': 'Data tidak ditemukan untuk kombinasi input tersebut'}), 404
        
        # Jika model tersedia, gunakan untuk prediksi
        if model is not None and scaler is not None:
            try:
                # Ambil fitur untuk prediksi
                features = filtered_data[['transactionCount', 'totalAmount', 'avgAmount', 
                                        'avgDuration', 'medianDuration', 'uniqueUsers']].iloc[0]
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Predict cluster
                cluster = model.predict(features_scaled)[0]
            except Exception as e:
                print(f"Error in prediction: {e}")
                cluster = 0  # Default cluster
        else:
            # Jika model tidak tersedia, gunakan cluster dari data
            cluster = int(filtered_data['Cluster'].iloc[0]) if 'Cluster' in filtered_data.columns else 0
        
        # Analisis kepadatan berdasarkan cluster
        density_analysis = {
            0: "Sepi – Layanan sangat lancar, penumpang sangat sedikit.",
            1: "Padat – Penumpang ramai, kemungkinan terjadi antrian."
        }
        
        # Get density label yang konsisten
        density_label = density_analysis.get(cluster, f"Cluster_{cluster}")
        
        result = {
            'koridor_asal': koridor_asal,
            'waktu': waktu,
            'cluster': int(cluster),
            'density_label': density_label,
            'recommendation': get_recommendation(cluster, waktu)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

def get_recommendation(cluster, waktu):
    """Memberikan rekomendasi berdasarkan cluster dan waktu"""
    
    # Rekomendasi berdasarkan kepadatan dan waktu
    if cluster == 0:  # Kepadatan Rendah
        if waktu == "Pagi (05:00-09:00)":
            return "Rute ini relatif lancar meskipun pada waktu puncak pagi. Anda dapat menggunakan transportasi ini dengan nyaman."
        elif waktu == "Dini Hari (21:00-05:00)":
            return "Rute ini sangat lancar pada waktu dini hari. Perjalanan akan sangat nyaman dan cepat."
        elif waktu == "Malam (18:00-21:00)":
            return "Rute ini relatif lancar pada waktu malam. Anda dapat menggunakan transportasi ini dengan nyaman."
        elif waktu == "Sore (15:00-18:00)":
            return "Rute ini relatif lancar pada waktu sore. Anda dapat menggunakan transportasi ini tanpa khawatir kemacetan."
        else:
            return "Rute ini relatif lancar. Anda dapat menggunakan transportasi ini tanpa khawatir kemacetan."
    
    elif cluster == 1:  # Kepadatan Sedang
        if waktu == "Pagi (05:00-09:00)":
            return "Rute ini memiliki kepadatan sedang pada waktu puncak pagi. Pertimbangkan untuk berangkat sedikit lebih awal."
        elif waktu == "Dini Hari (21:00-05:00)":
            return "Rute ini memiliki kepadatan sedang meskipun pada waktu dini hari. Perjalanan tetap lancar."
        elif waktu == "Malam (18:00-21:00)":
            return "Rute ini memiliki kepadatan sedang pada waktu malam. Perjalanan akan tetap nyaman."
        elif waktu == "Sore (15:00-18:00)":
            return "Rute ini memiliki kepadatan sedang pada waktu sore. Perjalanan akan tetap nyaman."
        else:
            return "Rute ini memiliki kepadatan sedang. Perjalanan akan tetap nyaman."
    
    elif cluster == 2:  # Kepadatan Tinggi
        if waktu == "Pagi (05:00-09:00)":
            return "Rute ini cukup padat pada waktu puncak pagi. Disarankan untuk mencari alternatif rute atau waktu perjalanan."
        elif waktu == "Dini Hari (21:00-05:00)":
            return "Rute ini cukup padat meskipun pada waktu dini hari. Pertimbangkan untuk mencari alternatif transportasi."
        elif waktu == "Malam (18:00-21:00)":
            return "Rute ini cukup padat pada waktu malam. Disarankan untuk mencari alternatif rute atau waktu perjalanan."
        elif waktu == "Sore (15:00-18:00)":
            return "Rute ini cukup padat pada waktu sore. Disarankan untuk mencari alternatif rute atau waktu perjalanan."
        else:
            return "Rute ini cukup padat. Disarankan untuk mencari alternatif rute atau waktu perjalanan."
    
    elif cluster == 3:  # Kepadatan Sangat Tinggi
        if waktu == "Pagi (05:00-09:00)":
            return "Rute ini sangat padat pada waktu puncak pagi. Sangat disarankan untuk mencari alternatif transportasi atau menunda perjalanan."
        elif waktu == "Dini Hari (21:00-05:00)":
            return "Rute ini sangat padat meskipun pada waktu dini hari. Sangat disarankan untuk mencari alternatif transportasi."
        elif waktu == "Malam (18:00-21:00)":
            return "Rute ini sangat padat pada waktu malam. Sangat disarankan untuk mencari alternatif transportasi atau menunda perjalanan."
        elif waktu == "Sore (15:00-18:00)":
            return "Rute ini sangat padat pada waktu sore. Sangat disarankan untuk mencari alternatif transportasi atau menunda perjalanan."
        else:
            return "Rute ini sangat padat. Sangat disarankan untuk mencari alternatif transportasi atau menunda perjalanan."
    
    else:
        return "Tidak ada rekomendasi tersedia untuk kombinasi ini."

@app.route('/statistics')
def statistics():
    if df is None:
        return jsonify({'error': 'Data tidak tersedia'}), 500
    
    # Overall statistics
    total_transactions = df['transactionCount'].sum()
    total_amount = df['totalAmount'].sum()
    avg_transactions = df['transactionCount'].mean()
    
    # Statistics by cluster
    cluster_stats = df.groupby('Cluster').agg({
        'transactionCount': ['count', 'sum', 'mean'],
        'totalAmount': 'sum',
        'uniqueUsers': 'sum'
    }).round(2)
    
    # Statistics by time category
    time_stats = df.groupby('timeCategory').agg({
        'transactionCount': ['count', 'sum', 'mean'],
        'totalAmount': 'sum'
    }).round(2)
    
    return jsonify({
        'overall': {
            'total_transactions': int(total_transactions),
            'total_amount': float(total_amount),
            'avg_transactions': float(avg_transactions)
        },
        'by_cluster': cluster_stats.to_dict(),
        'by_time': time_stats.to_dict()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 