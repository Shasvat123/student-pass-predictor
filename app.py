from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form['hours'])
        prediction = model.predict([[hours]])[0]
        prob = model.predict_proba([[hours]])[0][1]
        result = "Pass" if prediction == 1 else "Fail"

        return render_template('index.html', 
                               prediction=result, 
                               hours=hours, 
                               confidence=round(prob, 2))
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
