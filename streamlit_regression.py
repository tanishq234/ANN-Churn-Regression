import streamlit as st
import numpy as np
import tensorflow as tf
import h5py
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load preprocessing artifacts first (we may need scaler to infer input size if the HDF5 contains weights-only)
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
# Helper: robust model loader that handles full saved models and weights-only HDF5
def load_model_robust(path, scaler=None, fallback_feature_order=None):
    """Try to load a full Keras model from `path`. If that fails and the HDF5
    contains only weights, reconstruct a small Sequential model (matching the
    training architecture) and load_weights.

    Arguments:
        path: str, path to .h5 file
        scaler: scaler object (optional). Used to infer input_dim from
                scaler.feature_names_in_.
        fallback_feature_order: list of feature names fallback if scaler missing.
    Returns:
        tf.keras.Model instance
    """
    # Try the easy path first
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception:
        # fall through to inspect HDF5
        pass

    # Inspect HDF5 to see if it contains model config or only weights
    try:
        with h5py.File(path, 'r') as f:
            keys = list(f.keys())
    except Exception as e:
        raise RuntimeError(f"Cannot open HDF5 file '{path}': {e}")

    # Common indicator for weights-only Keras HDF5: contains 'model_weights' but
    # lacks full model serialization keys like 'model_config' or 'model_topology'.
    if 'model_weights' in keys and not any(k in keys for k in ('model_config', 'model_topology', 'model_config')):
        # Need to reconstruct architecture and load weights
        # Infer input_dim
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            input_dim = len(list(scaler.feature_names_in_))
        elif fallback_feature_order is not None:
            input_dim = len(fallback_feature_order)
        else:
            raise RuntimeError("Cannot infer model input dimension: provide a scaler with feature_names_in_ or feature_order.pkl")

        # Reconstruct the same architecture used during training: Dense(64)->Dense(32)->Dense(1)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Load weights from the HDF5 file
        try:
            model.load_weights(path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load weights into reconstructed model: {e}")

    # If we get here, the HDF5 didn't look like a weights-only file and load_model failed.
    raise RuntimeError(f"Unable to load model from '{path}' (not a full model and not a recognized weights-only file)")
    
#streamlit app

st.title('Estimated Salary prediction')


geography=st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('credit Score')
exited=st.selectbox('Exited',[0,1])
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('is Active Member',[0,1])


#Prepare input data

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'Exited' : [exited]
})

#One-hot encoded 'Geography'

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# --- Align feature names/order with the scaler used at training time ---
if hasattr(scaler, 'feature_names_in_'):
    feature_order = list(scaler.feature_names_in_)
else:
    # fallback: try loading a saved feature order
    try:
        feature_order = pickle.load(open('feature_order.pkl','rb'))
    except Exception:
        feature_order = None

if feature_order is not None:
    # Add any missing features with default 0
    for col in feature_order:
        if col not in input_data.columns:
            input_data[col] = 0
    # Drop unexpected columns (for example 'Exited' which is not a training feature)
    extras = [c for c in input_data.columns if c not in feature_order]
    if extras:
        input_data = input_data.drop(columns=extras)
    # Reorder columns to match scaler
    input_data = input_data[feature_order]
else:
    # If we can't determine feature order, just drop obvious target-like columns and proceed
    if 'Exited' in input_data.columns:
        input_data = input_data.drop(columns=['Exited'])

# Load model (prefer a full saved model if available, otherwise try the weights file)
model_file_preferred = 'regression_model_full.h5' if os.path.exists('regression_model_full.h5') else 'regression_model.h5'
try:
    model = load_model_robust(model_file_preferred, scaler=scaler, fallback_feature_order=feature_order)
except Exception as e:
    st.error('Failed to load model for prediction: ' + str(e))
    st.stop()

# Now scale
input_data_scaled = scaler.transform(input_data)


#Predict Salary

prediction=model.predict(input_data_scaled)
predicted_salary=prediction[0][0]

st.write(f'Predicted Estimated Salary: ${predicted_salary:.2f}')

 