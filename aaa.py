from flask import Flask, render_template, request
import requests
import pickle
from xgboost import XGBClassifier
#Random Forest classifier model 
from sklearn.ensemble import GradientBoostingClassifier

#instantiate model
gr = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

app = Flask(__name__)

# def load_model():
#     # with open('model_gradientboosting.pkl', 'rb') as file:
#     #     model = pickle.load(file)
#     model = pickle.load(open("model_gradientboosting.pkl", 'rb'))
#     return model

# model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cekurl', methods=['POST'])
def scan():
    url = request.form['url']
    model = pickle.load(open('model_gradientboosting.pkl', 'rb'))
    try:
        response = requests.get(url)
        content = response.text
    
    except:
        return render_template('result.html', result='Invalid URL')
    
    prediction = model.predict([122.1])
    
    # if prediction == 'phishing':
    #     result = 'URL Berbahaya!'
    # else:
    #     result = 'URL Aman'
    print(prediction)
        
    # return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
