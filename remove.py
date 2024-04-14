import joblib

with open('Wine_quality_model1.pkl','rb') as model_file:
    model=joblib.load(model_file)

if hasattr(model,'monotonic_cst'):
    delattr(model,'monotonic_cst')

with open ('Wine_quality_model2.pkl','wb') as model_file:
    joblib.dump(model,model_file)