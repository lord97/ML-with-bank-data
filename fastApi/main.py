import uvicorn
from fastapi import FastAPI 
from pydantic import BaseModel
import pickle
from model import Model
from model import __Version__ as version

#creation des objets 
app = FastAPI()
inf = pickle.load(open('model.sav', 'rb'))

class Input_value(BaseModel) :
    X : dict

class Output(BaseModel) :
    resultat: str
    probabilite : float
   

@app.get("/")
def home():
    return {"test": "OK", "model_version": version}


@app.post("/predict")
def predict(data ):
    classe,proba =  inf.predict(data)
    return {'resultat': classe, 'probabilite': proba}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
