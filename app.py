import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#-------TO IMPORT THE MODEL THAT IS TRAINED 
@st.cache(allow_output_mutation=True) #-----THIS WILL STORE THE MODEL IN CACHE MEMORY
def model():
    cnn=tf.keras.models.load_model(os.getcwd()+"/fire_and_smoke")
    return cnn

#------TO TEST THE IMAGE IF IT IS OF FIRE OR OF SMOKE
def image_results(test_image):
    testing=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    testing=tf.keras.preprocessing.image.img_to_array(testing)
    testing=testing/255
    testing=np.expand_dims(testing,axis=0)
    cnn=model()
    result=cnn.predict(testing)
    threshold = 0.5
    predicted_class = (result[0][0] > threshold).astype(int)
    Categories = ['Fire', 'Smoke']
    prediction = f"Prediction: {Categories[predicted_class]}"
    return prediction

#-------TO SHOW THE GRAPH OF ALL THE ACTIVATION FUNCTION
def graph():
    data=pd.read_csv(os.getcwd()+"/accuracy_value.csv",delimiter=',')
    epoch=data['epoch']
    relu_adam=100*data['relu (Adam)']
    leaky_relu_adam=100*data['leaky relu (Adam)']
    relu_RMSprop=100*data['relu (RMSprop)']
    leaky_relu_RMSprop=100*data['leaky relu (RMSprop)']
    fig=plt.figure()
    myaxis=fig.add_axes([0.1,0.1,1.6,1.6])
    myaxis.plot(epoch,relu_adam,'r',label='relu (Adam)')
    myaxis.plot(epoch,leaky_relu_adam,'b',label='leakyrelu (Adam)')
    myaxis.plot(epoch,relu_RMSprop,'m',label='relu (RMSprop)')
    myaxis.plot(epoch,leaky_relu_RMSprop,'y',label='leakyrelu (RMSprop)')
    myaxis.legend()
    return fig

st.set_page_config(page_title="forest_fire_detection", layout="wide")

#----header section

with st.container():
    
    st.markdown(
        """
        <div style="text-align: center; margin-top: -20px;">
            <h1>FOREST FIRE DETECTOR</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Team")
    st.write("""
        1. [Shivam Kumar](https://www.linkedin.com/in/shivamkumar14/)
        2. [Prince Banger](https://www.linkedin.com/in/prince-banger-44aa20224/)
        3. [Mohd Aman]()
        4. [Devansh Prashar]()
    """)

    st.subheader("Our project")
    st.write(" Wild fire are now more common so we have build an AI powered forest fire detetion sytem which will help us minimize the fire damage to some extent below is more information about our project")
    


#-----about our AI model
with st.container():
    st.write("---")
    left_column,right_column=st.columns(2)
    with left_column:
        st.header("What we did")
        st.write("##")
        st.write("""

        -We have created a CNN model using different activation function like Relu and Leaky Relu.

        -On right we have provided the graph comparision of both the activation function with different optimizer.
        
        -A specific kind of such a deep neural network is the convolutional network, which is commonly referred to as CNN or ConvNet. It's a deep, feed-forward artificial neural network. Remember that feed-forward neural networks are also called multi-layer perceptrons(MLPs), which are the quintessential deep learning models.
        
        -The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.
        
        """)
    
    with right_column:
        st.pyplot(graph())
    
    st.write("""
        -Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function based on a ReLU, but it has a small slope for negative values instead of a flat slope. The slope coefficient is determined before training, i.e. it is not learnt during training. This type of activation function is popular in tasks where we we may suffer from sparse gradients

        -Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent. The method is really efficient when working with large problem involving a lot of data or parameters. It requires less memory and is efficient. Intuitively, it is a combination of the gradient descent with momentum algorithm and the RMSP algorithm.

        -The RMSprop optimizer is similar to the gradient descent algorithm with momentum. The RMSprop optimizer restricts the oscillations in the vertical direction. Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster.

 
    """)

#------Testing of image
with st.container():
    st.write("---")
    text_column,image_column=st.columns((1,2))
    with text_column:
        file=st.file_uploader("Upload image of fire or smoke to test out our model",type=["jpg","png"])

    with image_column:
        if file is not None:
            prediction=image_results(file)
            st.image(file,use_column_width=True)
            st.success(prediction)

