from flask import Flask , render_template , request , jsonify
import pickle

application = Flask(__name__)
app = application

standard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'))
regressor = pickle.load(open('models/regressor.pkl' , 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        temp = int(request.form.get('Temperature'))
        rh = int(request.form.get('RH'))
        ws = int(request.form.get("Ws"))
        rain = float(request.form.get("Rain"))
        ffmc = float(request.form.get("FFMC"))
        dmc = float(request.form.get("DMC"))
        isi = float(request.form.get("ISI"))
        classes = float(request.form.get("Classes"))
        region = float(request.form.get("Region"))
        
        scaled_data = standard_scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        print(scaled_data.shape)
        pred = regressor.predict(scaled_data)
        
        
        print(pred[0])
        return render_template('home.html' , results = pred[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0' , debug = True)



