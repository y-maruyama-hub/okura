import os
import argparse
import cv2
import numpy as np
from flask import Flask,Response,request,jsonify
from keras.models import load_model
from dotenv import load_dotenv

#image_size = (40,30)
modelpath="model"
model_name="cat_model.h5"
weights_name="cat_weightsV2.h5"

##https://dev.to/yurfuwa/flask--tensorflow--keras--predict-bnk
##https://github.com/keras-team/keras/issues/2397#issuecomment-254919212

#gmodel=load_model(modelpath+"/"+model_name)
#gmodel.load_weights(modelpath+"/"+weights_name)



app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

@app.route("/predict",methods=["POST"])
def predict():

    _bytes = np.frombuffer(request.data, np.uint8)

    img = cv2.imdecode(_bytes, flags=cv2.IMREAD_COLOR)

    img = cv2.resize(img,(64,48))
    img = img.astype("float")/255

    r=gmodel.predict(np.array([img]),batch_size=32,verbose=0)

    res = r[0]

    res={"test":"aaa","argmax":res.argmax().item()}
    #res["test"]="aaa"
    #res["argmax"]=res.argmax().item()

    return jsonify(res)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-p","--port",type=int,default=5000)

    args = parser.parse_args()

    myport=int(args.port)

    load_dotenv()

    gmodel=load_model(os.getenv("MODEL_PATH"))
    gmodel.load_weights(os.getenv("WEIGHT_PATH"))


    app.run(host='0.0.0.0', debug=False,threaded=True,port=myport)
