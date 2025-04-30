import streamlit as st
import joblib
import pandas as pd

st.title("Titanic Survival Prediction")


# Buat Form Input
pclass = st.selectbox("pclass (1=First, 2=second, 3=Third)", [1,2,3])
age = st.slider("Age", 1, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("embarked", ["S", "C", "Q"])


# Load Model joblib
model = joblib.load("model_titanic_rf.joblib")

# Encoding helper
def preprocess_input(pclass, age, sibsp, parch, fare, embarked):
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0
    return pd.DataFrame([{
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_Q": embarked_q,
        "Embarked_S": embarked_s
    }])


if st.button("Predict"):
    input_data = preprocess_input(pclass, age, sibsp, parch, fare, embarked)
    prediction = model.predict(input_data)
    result = "🟢 Selamat" if prediction[0] == 1 else "🔴 Tidak Selamat"
    st.subheader("Hasil Prediksi:")
    st.success(result)
    
    
st.markdown("---")
st.markdown("Made with ❤️ by Amar")
st.markdown("Dataset: [Kaggle - Titanic](https://www.kaggle.com/c/titanic)")





