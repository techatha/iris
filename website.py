import pickle
import warnings
import streamlit as st
warnings.filterwarnings('ignore')
from PIL import Image


classifier = pickle.load(open('model_iris.pkl', 'rb'))


def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([sepal_length, sepal_width, petal_length, petal_width])
    print(prediction)
    return prediction


def Input_Output():
    st.title('Iris Variety Prediction')
    st.image('https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png', width=600)

    st.markdown('You are using Streamlit...', unsafe_allow_html=True)
    sepal_length = st.text_input('Enter Sepal Length', '.')
    sepal_width = st.text_input('Enter Sepal width', '.')
    petal_length = st.text_input('Enter petal Length', '.')
    petal_width = st.text_input('Enter petal width', '.')

    result = ''
    if st.button('Click here to predict'):
        result = predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width)
        st.balloons()
    st.success('The out put is {}'.format(result))


if __name__ == '__main__':
    Input_Output()

