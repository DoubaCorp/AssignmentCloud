from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import onnxruntime as rt


app = Flask(__name__)


# Load the ONNX model
onnx_session = rt.InferenceSession("outputs/log_reg_model.onnx")

# Load preprocessing transformations from pickle files
with open('outputs/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)



@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        review = str(request.form['review'])

        # Vectorize the processed review
        review_vectorized = vectorizer.transform([review])
        review_dense = review_vectorized.toarray()

        # Reshape the input to match the expected shape
        review_dense = review_dense.reshape(1, -1)

        # Make predictions using the ONNX model
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        onnx_input = {input_name: review_dense.astype(np.float32)}
        prediction = onnx_session.run([output_name], onnx_input)[0]

        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)