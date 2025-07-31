#!/usr/bin/env python3
"""
Script untuk memperbaiki model yang rusak dengan membuat model K-Means baru
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_new_model():
    """Membuat model K-Means baru dari data yang ada"""
    
    print("🔧 Memperbaiki model yang rusak...")
    
    try:
        # Load data
        print("📊 Loading data...")
        df = pd.read_csv('transjakarta_clustering_results.csv')
        print(f"✓ Data loaded: {len(df)} records")
        
        # Prepare features for clustering
        features = ['transactionCount', 'totalAmount', 'avgAmount', 
                   'avgDuration', 'medianDuration', 'uniqueUsers']
        
        # Check if features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"⚠ Warning: Missing features: {missing_features}")
            # Use available features
            features = [f for f in features if f in df.columns]
        
        if not features:
            print("✗ Error: No valid features found for clustering")
            return False
        
        print(f"✓ Using features: {features}")
        
        # Prepare data
        X = df[features].fillna(0)  # Fill missing values with 0
        
        # Create and fit scaler
        print("🔧 Creating scaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and fit K-Means model
        print("🔧 Creating K-Means model...")
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Save new model and scaler
        print("💾 Saving new model...")
        with open('transjakarta_density_model.pkl', 'wb') as f:
            pickle.dump(kmeans, f)
        
        with open('scaler_transjakarta.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("✓ Model and scaler saved successfully!")
        
        # Test the model
        print("🧪 Testing new model...")
        test_prediction = kmeans.predict(X_scaled[:1])
        print(f"✓ Test prediction: Cluster {test_prediction[0]}")
        
        # Show cluster distribution
        cluster_counts = np.bincount(kmeans.labels_)
        print("\n📈 Cluster distribution:")
        for i, count in enumerate(cluster_counts):
            print(f"  Cluster {i}: {count} records ({count/len(df)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def main():
    print("=" * 50)
    print("🔧 Transjakarta Model Repair Tool")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists('transjakarta_clustering_results.csv'):
        print("✗ Error: transjakarta_clustering_results.csv not found!")
        return
    
    # Backup existing files if they exist
    if os.path.exists('transjakarta_density_model.pkl'):
        os.rename('transjakarta_density_model.pkl', 'transjakarta_density_model.pkl.backup')
        print("📦 Backed up existing model file")
    
    if os.path.exists('scaler_transjakarta.pkl'):
        os.rename('scaler_transjakarta.pkl', 'scaler_transjakarta.pkl.backup')
        print("📦 Backed up existing scaler file")
    
    # Create new model
    success = create_new_model()
    
    if success:
        print("\n🎉 Model repair completed successfully!")
        print("🚀 You can now run 'python app.py' to start the application")
    else:
        print("\n❌ Model repair failed!")
        print("🔄 Restoring backup files...")
        
        # Restore backup files
        if os.path.exists('transjakarta_density_model.pkl.backup'):
            os.rename('transjakarta_density_model.pkl.backup', 'transjakarta_density_model.pkl')
        if os.path.exists('scaler_transjakarta.pkl.backup'):
            os.rename('scaler_transjakarta.pkl.backup', 'scaler_transjakarta.pkl')
        
        print("✓ Backup files restored")

if __name__ == "__main__":
    main() 