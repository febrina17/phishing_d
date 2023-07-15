from flask import Flask, request, render_template
import requests
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import warnings
import requests
warnings.filterwarnings('ignore')
from fitur import FeatureExtraction


file = open('model_gradientboosting.pkl','rb')
gr = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route('/cekurl', methods=["GET", "POST"])
def index():
    
    if request.method == "POST":
        
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.get_features_list()).reshape(1,-1)
        # print(feature)
        y_pred = gr.predict(x)[0]
        y_pro_phishing = gr.predict_proba(x)[0,0]
        y_pro_non_phishing = gr.predict_proba(x)[0,1]
        pred = 'it is {0:2f}% safe to go'.format(y_pro_phishing*100)
        payload_scoring = {"input_data":[{"field":[["UsingIP","LongURL","ShortURL","Symbol@,'Redirecting//","SubDomains ","HTTPS","DomainRegLen","Favicon","NonStdPort","HTTPSDomainURL","RequestURL","AnchorURL","LinksInScriptTags","ServerFormHandler","ServerFormHandler","AbnormalURL","WebsiteForwarding","StatusBarCust","DisableRightClick","UsingPopupWindow","IframeRedirection","IframeRedirection","DNSRecording","WebsiteTraffic","PageRank","GoogleIndex","LinksPointingToPage","StatsReport","class"
                                                    ]], "values" :[[1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,1,0,1,1,1,1,-1,-1,-1,-1,1,0,1]]}]}
        response_scoring = requests.post('', json=payload_scoring)
        print("Scoring Response")
        predictions=response_scoring.json()
        pred=print(predictions['predictions'][0]['values'][0][0])
        return render_template('indext.html' ,xx =round(y_pro_non_phishing, 2),url=url)
    return render_template("")

if __name__=="__main__":
    app.run(debug=True,port=2020)