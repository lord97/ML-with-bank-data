import pandas as pd
__Version__ = '0.1.0'
class Model :
    def __init__(self,encoders,model) :
        self.encoders = encoders
        self.model = model
        
    def __validate_data(self,X) :
        return True
    
    def __preprocessing(self,X) : 
        df = pd.DataFrame(data = [X])
        geo = self.encoders['geo'].transform([X['Geography']])
        gender = self.encoders['gender'].transform([X['Gender']])
        df = df.drop(['Geography','Gender'], axis = 1)
        column_geography = ['Geography_'+ val for val in self.encoders['geo'].classes_]
        df_geography = pd.DataFrame(geo,columns=column_geography, index=df.index)
        df_genre = pd.DataFrame(gender, columns=['Gender'], index=df.index)

        df = pd.concat([df,df_geography,df_genre], axis=1)
        return df.values
    
    def predict(self,X) :
        if not self.__validate_data(X):
            raise Exception('Données non valides')
        X_enc = self.__preprocessing(X)
        prob = self.model.predict_proba(X_enc)
        label_ind,proba = prob.argmax(), prob.max()
        classe = self.encoders['labels'].inverse_transform([label_ind])[-1] 
            
        return classe, proba
    