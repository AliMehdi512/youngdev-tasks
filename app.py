from flask import Flask, request, render_template
import pickle
import os
import zipfile

app = Flask(__name__)

# Step 1: Extract model.pkl from model.zip
with zipfile.ZipFile("model.zip", "r") as zip_ref:
    zip_ref.extractall()  # Extracts all files, including model.pkl

# Step 2: Check the extracted file exists
if not os.path.exists("model.pkl"):
    raise FileNotFoundError("❌ model.pkl not found after extraction!")

# Step 3: Load the model
with open("model.pkl", "rb") as f:
    first = f.read(1)
    if first != b'\x80':
        raise ValueError("❌ Invalid pickle file (maybe HTML or corrupted)")
    f.seek(0)
    data = pickle.load(f)

model = data['model']
feature_names = data['features']


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form[f'feature{i}'])
                for i in range(1,
                               len(feature_names) + 1)
            ]
            prediction = model.predict([features])[0]
            prediction = round(prediction * 100000,
                               2)  # The dataset target is in 100k units approx
        except:
            prediction = "Invalid input"

    return render_template('index.html',
                           prediction=prediction,
                           feature_names=feature_names)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
