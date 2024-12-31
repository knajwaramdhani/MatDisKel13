from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Membaca dataset dan melatih model
file_path = 'C:/Users/HP/Matdis/Projek/parkinsons_disease_data.csv'
df = pd.read_csv(file_path)

# Fitur yang relevan untuk diagnosis Parkinson
selected_features = [
    'Age', 'UPDRS', 'MoCA', 'FunctionalAssessment', 'Tremor',
    'Rigidity', 'Bradykinesia', 'PosturalInstability', 
    'SpeechProblems', 'SleepDisorders', 'Constipation'
]

# Pisahkan fitur dan label
X = df[selected_features]
y = df['Diagnosis']

# Membagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan model ke file untuk digunakan kembali
with open('parkinsons_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load model dari file
with open('parkinsons_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Flask App
app = Flask(__name__)

# Rute untuk halaman utama (index.html)
@app.route('/')
@app.route('/home')  # Tambahan rute untuk home
def index():
    return render_template('index.html')

# Rute untuk halaman tentang (about.html)
@app.route('/about')
def about():
    return render_template('about.html')

# Rute untuk halaman input fitur (fitur.html)
@app.route('/fitur')
def fitur():
    return render_template('fitur.html')

# Rute untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari form
        data = request.get_json()

        # Ambil nilai dari form dan urutkan sesuai fitur
        input_data = {
            "Age": [float(data["Age"])],
            "UPDRS": [float(data["UPDRS"])],
            "MoCA": [float(data["MoCA"])],
            "FunctionalAssessment": [float(data["FunctionalAssessment"])],
            "Tremor": [float(data["Tremor"])],
            "Rigidity": [float(data["Rigidity"])],
            "Bradykinesia": [float(data["Bradykinesia"])],
            "PosturalInstability": [float(data["PosturalInstability"])],
            "SpeechProblems": [float(data["SpeechProblems"])],
            "SleepDisorders": [float(data["SleepDisorders"])],
            "Constipation": [float(data["Constipation"])]
        }

        # Konversi input menjadi DataFrame Pandas
        input_df = pd.DataFrame(input_data)

        # Prediksi menggunakan model
        prediction = loaded_model.predict(input_df)[0]
        diagnosis = "Terdiagnosis Parkinson" if prediction == 1 else "Tidak Terdiagnosis Parkinson"

        # Kirim hasil prediksi sebagai JSON
        return jsonify({"prediction": diagnosis})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)