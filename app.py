import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load the trained model

# Load the trained model (assuming the ensemble model is the best and saved as 'ensemble_model.pkl')
# You would need to save your meta_model after training it.
# Example: joblib.dump(meta_model, 'ensemble_model.pkl')
try:
    meta_model = joblib.load('ensemble_model.pkl')
except FileNotFoundError:
    st.error("Error: Model file 'ensemble_model.pkl' not found. Please train and save the ensemble model first.")
    st.stop()


# Load the original data or a processed version to get unique values for dropdowns
# Assuming 'df' is available from your Colab session or loaded from a file
# It's better to load a cleaned version of the data used for training the models
try:
    # Assuming you saved the cleaned dataframe somewhere
    # df = pd.read_csv('cleaned_car_data.csv')
    # For now, let's try to reconstruct necessary info from the Colab environment if possible
    # This is not ideal for a standalone app, better to save and load.
    # Accessing variables directly from Colab environment is not possible in a standalone script.
    # Let's assume we have a way to get the unique values and mapping used during training
    # In a real scenario, you'd save these mappings (like LabelEncoder classes)
    # For demonstration, let's create some dummy data based on the notebook state

    # Recreate LabelEncoders and get unique values - this requires having the original df or mappings saved
    # Since we don't have the saved state or mappings in a standalone script,
    # I'll use some placeholder data and mention that in a real app, you'd load these.

    # Placeholder data - replace with loading your actual processed data and encoders
    unique_brands = ['Maruti', 'Tata', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Mahindra', 'Renault', 'Volkswagen', 'Skoda', 'Nissan', 'Datsun', 'BMW', 'Mercedes-Benz', 'Audi', 'Jeep', 'Kia', 'MG', 'Volvo', 'Jaguar', 'Porsche', 'Land Rover', 'Rolls-Royce', 'Ferrari', 'Lamborghini', 'Bentley', 'Maserati', 'Aston Martin', 'Bugatti'] # Example brands
    unique_owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'] # Example owner types
    unique_fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'] # Example fuel types
    unique_transmission_types = ['Manual', 'Automatic'] # Example transmission types
    unique_body_types = ['Hatchback', 'MUV', 'Sedan', 'SUV', 'Minivans', 'Coupe', 'Pickup', 'Cars', 'Wagon', 'Convertibles'] # Example body types
    # You would need similar lists for car_name based on brand, insurance, and seats if they were categorical
    # For numerical columns, you'd need the scaler used for transformation

    # In a real app, load your fitted LabelEncoders and StandardScaler
    # For this example, we'll simulate the encoding and scaling based on the notebook's steps
    # This is a simplification and might not perfectly match the training data

    # Load the original dataframe used for training to create mappings
    # Assuming the cleaned dataframe is saved as 'cleaned_df.csv'
    try:
        original_df_cleaned = pd.read_csv('cleaned_df.csv') # You need to save this in Colab
    except FileNotFoundError:
        st.warning("Warning: cleaned_df.csv not found. Using placeholder values. Dynamic car name filtering will not work correctly.")
        original_df_cleaned = pd.DataFrame({
            'brand': unique_brands * 5,
            'car_name': [f'Car_{i}' for i in range(len(unique_brands) * 5)],
            'insurance': unique_owner_types * 3,
            'transmission_type': unique_transmission_types * 4,
            'owner_type': unique_owner_types * 2,
            'fuel_type': unique_fuel_types * 6,
            'body_type': unique_body_types * 7,
            'model_year': np.random.randint(2000, 2023, len(unique_brands) * 5),
            'registered_year': np.random.randint(2000, 2023, len(unique_brands) * 5),
            'engine_capacity': np.random.randint(800, 3000, len(unique_brands) * 5),
            'kms_driven': np.random.randint(1000, 100000, len(unique_brands) * 5),
            'max_power': np.random.uniform(50, 200, len(unique_brands) * 5),
            'seats': np.random.randint(4, 8, len(unique_brands) * 5),
            'mileage': np.random.uniform(10, 30, len(unique_brands) * 5),
            'resale_price': np.random.randint(100000, 2000000, len(unique_brands) * 5)
        })


    # Create mappings for categorical features based on the cleaned data
    brand_car_mapping = original_df_cleaned.groupby('brand')['car_name'].unique().apply(list).to_dict()
    # Add a default or empty list for brands not in the mapping if needed

    # Recreate LabelEncoders for categorical features
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le_dict = {}
    for col in ['brand', 'car_name', 'insurance', 'transmission_type', 'owner_type', 'fuel_type', 'body_type']:
         if col in original_df_cleaned.columns:
            le = LabelEncoder()
            # Handle potential unseen labels in the future by fitting on all unique values
            le.fit(original_df_cleaned[col].astype(str).unique())
            le_dict[col] = le
         else:
             st.warning(f"Column '{col}' not found in cleaned_df.csv. Label encoding for this column will not be applied.")


    # Recreate StandardScaler for numerical features
    # Assuming the numerical columns are the same as in your training data
    num_cols = ['model_year', 'registered_year', 'engine_capacity', 'kms_driven', 'max_power', 'seats', 'mileage']
    # Filter original_df_cleaned to only include num_cols and drop NaNs before fitting scaler
    df_for_scaler = original_df_cleaned[num_cols].dropna()
    scaler = StandardScaler()
    scaler.fit(df_for_scaler)


except FileNotFoundError:
    st.error("Error: cleaned_df.csv not found. Cannot load data for dropdowns and scaling.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data and creating mappings: {e}")
    st.stop()


st.title("Car Resale Price Prediction")

# Create input fields with dropdown menus
brand = st.selectbox("Select Brand", sorted(brand_car_mapping.keys()))

# Dynamically update car names based on selected brand
car_names_for_brand = brand_car_mapping.get(brand, [])
car_name = st.selectbox("Select Car Name", sorted(car_names_for_brand))


model_year = st.number_input("Model Year", min_value=1900, max_value=2025, value=2020)
registered_year = st.number_input("Registered Year", min_value=1900, max_value=2025, value=2020)
engine_capacity = st.number_input("Engine Capacity (cc)", min_value=50, max_value=6000, value=1200)
insurance = st.selectbox("Insurance Type", sorted(le_dict.get('insurance', LabelEncoder()).classes_.tolist() if 'insurance' in le_dict else ['Comprehensive', 'Third Party', 'No Insurance']))
transmission_type = st.selectbox("Transmission Type", sorted(le_dict.get('transmission_type', LabelEncoder()).classes_.tolist() if 'transmission_type' in le_dict else ['Manual', 'Automatic']))
kms_driven = st.number_input("Kms Driven", min_value=0, max_value=1000000, value=50000)
owner_type = st.selectbox("Owner Type", sorted(le_dict.get('owner_type', LabelEncoder()).classes_.tolist() if 'owner_type' in le_dict else ['First Owner', 'Second Owner', 'Third Owner']))
fuel_type = st.selectbox("Fuel Type", sorted(le_dict.get('fuel_type', LabelEncoder()).classes_.tolist() if 'fuel_type' in le_dict else ['Petrol', 'Diesel', 'CNG']))
max_power = st.number_input("Max Power (bhp)", min_value=10.0, max_value=600.0, value=100.0)
seats = st.number_input("Number of Seats", min_value=2, max_value=15, value=5)
mileage = st.number_input("Mileage (kmpl)", min_value=1.0, max_value=150.0, value=20.0)
body_type = st.selectbox("Body Type", sorted(le_dict.get('body_type', LabelEncoder()).classes_.tolist() if 'body_type' in le_dict else ['Sedan', 'Hatchback', 'SUV']))


# Create a button to predict
if st.button("Predict Resale Price"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'model_year': [model_year],
        'brand': [brand],
        'car_name': [car_name],
        'registered_year': [registered_year],
        'engine_capacity': [engine_capacity],
        'insurance': [insurance],
        'transmission_type': [transmission_type],
        'kms_driven': [kms_driven],
        'owner_type': [owner_type],
        'fuel_type': [fuel_type],
        'max_power': [max_power],
        'seats': [seats],
        'mileage': [mileage],
        'body_type': [body_type]
    })

    # Apply the same preprocessing steps as used during training

    # Label Encoding for categorical features
    categorical_cols = ['brand', 'car_name', 'insurance', 'transmission_type', 'owner_type', 'fuel_type', 'body_type']
    for col in categorical_cols:
        if col in le_dict and col in input_data.columns:
            # Handle potential unseen labels by transforming known classes and marking others
            # A more robust approach would be to use a pipeline that handles unseen labels
            input_data[col] = input_data[col].apply(lambda x: le_dict[col].transform([x])[0] if x in le_dict[col].classes_ else -1) # Use -1 for unseen

    # Scaling for numerical features
    numerical_cols = ['model_year', 'registered_year', 'engine_capacity', 'kms_driven', 'max_power', 'seats', 'mileage']
    # Ensure columns exist before scaling
    cols_to_scale = [col for col in numerical_cols if col in input_data.columns]
    if cols_to_scale:
        input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])
    else:
        st.warning("No numerical columns found to scale.")


    # Ensure the order of columns matches the training data used for the meta-model
    # This requires knowing the exact columns and their order in the meta-model's training data (x_meta_train)
    # Based on the notebook, x_meta_train columns were: 'lr_pred', 'ridge_pred', 'lasso_pred', 'elasticnet_pred', 'nn_pred'
    # This means the meta-model expects predictions from the base models as input.
    # So, the Streamlit app needs to:
    # 1. Load the individual base models (LR, Ridge, Lasso, ElasticNet, NN)
    # 2. Make predictions using each base model on the preprocessed input_data
    # 3. Create the meta-model input DataFrame using these predictions
    # 4. Use the meta_model to predict on the meta-model input

    st.warning("Note: The Streamlit app currently loads only the meta-model. To make predictions with the stacking ensemble, you need to load all the base models (Linear Regression, Ridge, Lasso, ElasticNet, and the improved Neural Network) and use their predictions as input to the meta-model.")
    st.warning("Please ensure all base models are saved (e.g., using joblib) and loaded here to make accurate ensemble predictions.")

    # Placeholder for making predictions with base models and then the meta-model
    # In a real implementation, load base models and predict:
    # lr_model = joblib.load('lr_model.pkl')
    # ridge_model = joblib.load('ridge_model.pkl')
    # ...
    # nn_model = tf.keras.models.load_model('nn_model.h5') # Example for Keras model

    # Example of creating meta-model input (requires base models loaded and predicting)
    # base_preds = pd.DataFrame({
    #     'lr_pred': lr_model.predict(input_data_processed),
    #     'ridge_pred': ridge_model.predict(input_data_processed),
    #     'lasso_pred': lasso_model.predict(input_data_processed),
    #     'elasticnet_pred': en_model.predict(input_data_processed),
    #     'nn_pred': nn_model.predict(input_data_processed).flatten()
    # })


    # Assuming, for now, you have a way to get the final prediction (replace with actual ensemble prediction)
    # This part needs to be corrected to use the base models to predict and then feed to meta_model
    try:
        # This is a placeholder and will likely give an error or incorrect result
        # because the input_data is not in the expected format for the meta_model
        # which expects base model predictions.
        # You need to load and use base models here.
        # For demonstration, let's assume a direct prediction for now (INCORRECT for ensemble)
        # Replace this with the actual ensemble prediction process:
        # prediction = meta_model.predict(base_preds)
        prediction = meta_model.predict(input_data) # This line is incorrect for ensemble prediction


        st.success(f"Predicted Resale Price: ₹ {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
