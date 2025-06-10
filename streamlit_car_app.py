'''
Project - UAS
'''
import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Car Evaluation App",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Load model components
@st.cache_resource
def load_model():
    """Load the trained model components"""
    try:
        components = joblib.load('class_prediction_components.joblib')
        return components
    except FileNotFoundError:
        st.error("Model file 'class_prediction_components.joblib' not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_income(data, model_components):
    """Make car evaluation using the trained model"""
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    # Get components
    model = model_components['model']
    encoding_maps = model_components['encoding_maps']
    feature_names = model_components['feature_names']
    
    # Apply encodings to categorical columns
    for column in df.columns:
        if column in encoding_maps and column != 'class':
            df[column] = df[column].map(encoding_maps[column])
    
    # Ensure we only use features that the model was trained on
    df_for_pred = df[feature_names].copy()
    
    # Make prediction
    prediction = model.predict(df_for_pred)[0]
    probabilities = model.predict_proba(df_for_pred)[0]
    class_index = list(model.classes_).index(prediction)
    
    # Get income label
    income_map_inverse = {v: k for k, v in encoding_maps['class'].items()}
    prediction_label = income_map_inverse[prediction]

    return {
        'prediction': prediction,
        'prediction_label': prediction_label,
        'probability': probabilities[class_index],
        'probabilities': probabilities.tolist()
    }

def validate_inputs(data):
    """Validate input data"""
    errors = []
    
    # Age validation
    if data['buying'] == 'nol':
        errors.append("Age should be between 17 and 90")
    
#     # Education number validation
#     if data['education_num'] < 1 or data['education_num'] > 16:
#         errors.append("Education number should be between 1 and 16")
    
#     # Hours per week validation
#     if data['hours_per_week'] < 1 or data['hours_per_week'] > 99:
#         errors.append("Hours per week should be between 1 and 99")
    
#     # Capital gain/loss validation
#     if data['capital_gain'] < 0 or data['capital_gain'] > 99999:
#         errors.append("Capital gain should be between 0 and 99999")
    
#     if data['capital_loss'] < 0 or data['capital_loss'] > 4356:
#         errors.append("Capital loss should be between 0 and 4356")
    
#     # Final weight validation
#     if data['fnlwgt'] < 12285 or data['fnlwgt'] > 1484705:
#         errors.append("Final weight should be between 12285 and 1484705")
    
    return errors

def export_prediction(data, result):
    """Export prediction result to JSON"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'input_data': data,
        'prediction': {
            'class': result['prediction_label'],
            'confidence': result['probability'],
            'raw_prediction': result['prediction']
        }
    }
    return json.dumps(export_data, indent=2)

def reset_session_state():
    """Reset all input values to default"""
    keys_to_reset = [
        'buying', 'maint', 'doors', 'persons', 'lug_boot',
        'safety'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# Load model
model_components = load_model()

# Define mappings (from the original notebook)
buying_options = ['vhigh','high','med','low']

maint_options = ['vhigh','high','med','low']

doors_options = ['2','3','4','5more']

persons_options = ['2','4','more']

lug_boot_options = ['small','med','big']

safety_options = ['low','med','high']

# Main app
st.title("Car Evaluation App - UAS")
st.markdown("Predict Car Evaluation based on price, tech and comfort")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Features")
    
    # Create form for inputs
    with st.form("prediction_form"):
        # Demographic Information
        st.markdown("**Demographic Information**")
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            buying = st.selectbox("Buying", buying_options, key="buying")
            maint = st.selectbox("Maint", maint_options, key="maint")
            doors = st.selectbox("Doors", doors_options, key="doors")
        
        with col_demo2:
            persons = st.selectbox("Persons", persons_options, key="persons")
            lug_boot = st.selectbox("Luggage", lug_boot_options, key="lug_boot")
            safety = st.selectbox("Safety", safety_options, key="safety")
        
        st.divider()
        
        # # Work Information
        # st.markdown("**Work Information**")
        # col_work1, col_work2 = st.columns(2)
        
        # with col_work1:
        #     workclass = st.selectbox("Work Class", workclass_options, key="workclass")
        #     occupation = st.selectbox("Occupation", occupation_options, key="occupation")
        #     hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, key="hours_per_week")
        
        # with col_work2:
        #     education_num = st.number_input("Education Level (Years)", min_value=1, max_value=16, value=10, key="education_num")
        #     fnlwgt = st.number_input("Final Weight", min_value=12285, max_value=1484705, value=77516, key="fnlwgt")
        
        # st.divider()
        
        # # Financial Information
        # st.markdown("**Financial Information**")
        # col_fin1, col_fin2 = st.columns(2)
        
        # with col_fin1:
        #     capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0, key="capital_gain")
        
        # with col_fin2:
        #     capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4356, value=0, key="capital_loss")
        
        # Buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_button = st.form_submit_button("🔮 Predict", type="primary")
        with col_btn2:
            reset_button = st.form_submit_button("🔄 Reset")
        with col_btn3:
            export_button = st.form_submit_button("📤 Export Last Result")

# Handle reset button
if reset_button:
    reset_session_state()
    st.rerun()

# Handle prediction
if predict_button:
    # Collect input data
    input_data = {
        'buying': buying,
        'maint': maint,
        'doors': doors,
        'persons': persons,
        'lug_boot': lug_boot,
        'safety': safety,
    }
    
    # Validate inputs
    validation_errors = validate_inputs(input_data)
    
    if validation_errors:
        with col2:
            st.error("❌ Validation Errors:")
            for error in validation_errors:
                st.error(f"• {error}")
    else:
        # Make prediction
        try:
            result = predict_income(input_data, model_components)
            
            # Store result in session state for export
            st.session_state['last_prediction'] = {
                'input_data': input_data,
                'result': result
            }
            
            with col2:
                st.subheader("🎯 Prediction Results")
                # Display prediction
                prediction_color = "green" if result['prediction_label'] == 'vgood' else "yellow" if result['prediction_label'] == 'good' else "red" if result['prediction_label'] == 'unacc' else "orange"

                st.markdown(f"**Predicted Car Evaluation:** :{prediction_color}[{result['prediction_label']}]")
                
                # Confidence level with gauge
                confidence = result['probability'] * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': prediction_color},
                        'steps': [
                            {'range': [0, 25], 'color': "red"},
                            {'range': [25, 50], 'color': "orange"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # # Probability breakdown
                class_map_inverse = {v: k for k, v in model_components['encoding_maps']['class'].items()}
                prediction_label = [class_map_inverse[c] for c in model_components['model'].classes_]

                prob_df = pd.DataFrame({
                    'Class': prediction_label,
                    'Probability': result['probabilities']
                })

                
                fig_bar = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    title='Probability Distribution',
                    color='Probability',
                    color_continuous_scale=['orange', 'green']
                )
                fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            with col2:
                st.error(f"❌ Prediction Error: {str(e)}")

# Feature Importance section
st.subheader("📊 Feature Importance")

if 'model' in model_components:
    try:
        feature_names = model_components['feature_names']
        feature_importance = model_components['model'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Feature Importance in Decision Tree Model',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_importance, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")

# Handle export
if export_button:
    if 'last_prediction' in st.session_state:
        export_data = export_prediction(
            st.session_state['last_prediction']['input_data'],
            st.session_state['last_prediction']['result']
        )
        
        st.download_button(
            label="📥 Download Prediction Results",
            data=export_data,
            file_name=f"income_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ No prediction results to export. Please make a prediction first.")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit • RPL ...*")