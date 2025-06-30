from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = tf.keras.models.load_model('multimodal_poultry_model.h5')
label_names = ['Salmonella', 'New castle diseases', 'Coccidiosis', 'Healthy poultry']
symptom_fields = ['symptom_lethargy', 'symptom_diarrhea', 'symptom_coughing', 'symptom_sneezing', 'symptom_loss_of_appetite']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files['image']
        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        symptoms_vector = [1 if request.form.get(field) == 'on' else 0 for field in symptom_fields]
        symptoms_vector = np.array([symptoms_vector])

        img_input = preprocess_image(filepath)
        # 4. Predict
        prediction_probs = model.predict([img_input, symptoms_vector])
        predicted_class = label_names[np.argmax(prediction_probs)]

        # 5. Suggestions dictionary (based on your predicted_class)
        suggestions = {
            "Salmonella": "🧼 Isolate infected birds and administer antibiotics under a vet's advice.",
            "New castle diseases": "💉 Vaccinate healthy birds and boost farm biosecurity.",
            "Coccidiosis": "💊 Use anticoccidial drugs and keep the environment dry and clean.",
            "Healthy poultry": "✅ Keep maintaining hygiene, proper feed, and vaccination schedules."
}

        recommendation = suggestions.get(predicted_class, "⚠️ Consult a vet for accurate treatment.")

        return render_template('result.html', prediction=predicted_class, image=filename, suggestion=recommendation)

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
