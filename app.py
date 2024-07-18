import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Define your model architecture here
class KAN(nn.Module):
    def __init__(self, width, grid, k, seed):
        super(KAN, self).__init__()
        torch.manual_seed(seed)
        layers = []
        for i in range(len(width) - 1):
            layers.append(nn.Linear(width[i], width[i + 1]))
            if i < len(width) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Recreate the model architecture
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)

# Load the state dictionary
# model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
model.eval()  # Set the model to evaluation mode

# Function to make predictions
def make_prediction(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).numpy()
    return prediction

# Streamlit interface
st.title("KAN Model Deployment with Streamlit")

# Upload file or enter data manually
uploaded_file = st.file_uploader("Upload an input file (CSV)", type=["csv"])
manual_input = st.text_area("Or enter input data manually (comma separated, e.g., '1, 2')")

if uploaded_file is not None:
    input_data = np.loadtxt(uploaded_file, delimiter=',')
    st.write("Input Data:")
    st.write(input_data)
    
    prediction = make_prediction(input_data)
    st.write("Prediction:")
    st.write(prediction)

elif manual_input:
    try:
        input_data = np.array([float(x) for x in manual_input.split(',')]).reshape(1, -1)
        st.write("Input Data:")
        st.write(input_data)
        
        prediction = make_prediction(input_data)
        st.write("Prediction:")
        st.write(prediction)
    except ValueError:
        st.write("Invalid input format. Please enter numbers separated by commas.")
