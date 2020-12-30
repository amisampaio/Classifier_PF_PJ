import joblib

def is_PF_PJ(texto):
    vectorizer = joblib.load('vectorizer.jbl')
    transformer = joblib.load('transformer.jbl')
    model = joblib.load('model.jbl')
    X = vectorizer.transform([texto])
    X = transformer.transform(X)
    predictions = model.predict(X)
    return predictions[0]
