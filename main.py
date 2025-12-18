import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# создание объекта приложения
app = FastAPI(title='Предсказание сердечного приступа')

# кастомный класс, который используется в модели
# класс для конвертации типов
class TypeConverter(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None, dtype='Int64'):
        self.columns = columns
        self.dtype = dtype
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        if self.columns == None:
            return X_transformed.astype(self.dtype)
        else:
            for col in self.columns:
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].astype(self.dtype)
            return X_transformed

# загрузка модели
try:
    model = joblib.load('heart_model.pkl')
except Exception as e:
    print(f'Ошибка загрузки модели: {e}')

class Input(BaseModel):
    data: str

@app.post('/predict')
def predict(query: Input):
    try:
        # чтение файла по указанному пути
        data = pd.read_csv(query.data, index_col='id', usecols=lambda x: x not in ['Unnamed: 0'])
        
        # предсказание
        predictions = model.predict(data)
        
        # формирование результата
        result = {
            'prediction': predictions.tolist()
        }
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Файл не найден')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
