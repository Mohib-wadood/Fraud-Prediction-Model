import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection EDA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 24px; color: #1f77b4; font-weight: bold;}
    .section-header {font-size: 20px; color: #2ca02c; border-bottom: 2px solid #2ca02c; padding-bottom: 5px;}
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
    .fraud-alert {background-color: #ffcccc; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;}
    .normal-alert {background-color: #ccffcc; padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🔍 Fraud Detection Explorer")
st.markdown("""
This interactive application allows you to explore and analyze transaction data for fraud detection.
Upload your dataset or use the sample data to get started.
""")

# Sidebar navigation and file uploader
with st.sidebar:
    st.header("Data Input")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    # Navigation
    st.header("Navigation")
    page = st.radio(
        "Select Analysis Section",
        ["Data Preview", "Feature Exploration", "Class Distribution", "Correlation Heatmap", "Model Prediction"]
    )
    
    # Theme toggle
    st.header("Preferences")
    theme = st.radio("Theme", ["Light", "Dark"])
    
    # Sample data option
    if uploaded_file is None:
        use_sample_data = st.checkbox("Use sample data for demonstration")
    else:
        use_sample_data = False

# Load data function with caching
@st.cache_data
def load_data(uploaded_file, use_sample_data):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    elif use_sample_data:
        # Generate sample data similar to the credit card fraud dataset
        np.random.seed(42)
        n_samples = 10000
        n_frauds = int(0.01 * n_samples)  # 1% fraud cases
        
        # Create sample data with V1-V28 features
        data = np.random.randn(n_samples, 30)
        df = pd.DataFrame(data, columns=[f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'])
        
        # Create target variable
        fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
        df['Class'] = 0
        df.loc[fraud_indices, 'Class'] = 1
        
        # Make some features more meaningful for fraud
        df.loc[fraud_indices, 'V1'] += 2
        df.loc[fraud_indices, 'V2'] -= 1.5
        df.loc[fraud_indices, 'V3'] += 1.2
        df.loc[fraud_indices, 'V4'] -= 2.2
        df.loc[fraud_indices, 'Amount'] *= 3
        
        return df
    else:
        return None

# Load the data
df = load_data(uploaded_file, use_sample_data)

# Show appropriate message if no data is available
if df is None:
    st.info("Please upload a CSV file or enable the sample data option to begin analysis.")
    st.stop()

# Main content area
if page == "Data Preview":
    st.header("Data Preview")
    
    # Display dataset metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Rows", df.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Columns", df.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Missing Values", df.isnull().sum().sum())
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        fraud_count = df['Class'].sum() if 'Class' in df.columns else 0
        st.metric("Fraud Cases", fraud_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Data Summary"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(df.tail(10), use_container_width=True)
    
    with tab3:
        # Create a summary dataframe
        summary_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(summary_df, use_container_width=True)
        
        # Show detailed statistics in an expander
        with st.expander("View Detailed Statistics"):
            st.dataframe(df.describe(), use_container_width=True)

elif page == "Feature Exploration":
    st.header("Feature Exploration")
    
    # Check if Class column exists
    if 'Class' not in df.columns:
        st.error("The dataset does not contain a 'Class' column. Please upload a valid fraud detection dataset.")
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Feature selector
        feature_options = [col for col in df.columns if col != 'Class' and df[col].dtype in ['float64', 'int64']]
        selected_feature = st.selectbox("Select a feature to explore:", feature_options)
        
        # Add some options for the user
        show_by_class = st.checkbox("Show distribution by class", value=True)
        bin_size = st.slider("Bin size for histogram", min_value=5, max_value=100, value=30)
    
    with col2:
        # Create subplots
        if show_by_class:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'Distribution of {selected_feature}', f'Box Plot of {selected_feature} by Class')
            )
            
            # Add histogram
            for cls, color in zip([0, 1], ['green', 'red']):
                cls_data = df[df['Class'] == cls][selected_feature]
                fig.add_trace(
                    go.Histogram(
                        x=cls_data, 
                        name=f'Class {cls}', 
                        marker_color=color, 
                        opacity=0.7,
                        nbinsx=bin_size
                    ),
                    row=1, col=1
                )
            
            # Add box plot
            fig.add_trace(
                go.Box(
                    x=df['Class'], 
                    y=df[selected_feature], 
                    boxpoints='outliers',
                    marker_color='lightblue',
                    line_color='darkblue'
                ),
                row=1, col=2
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'Distribution of {selected_feature}', f'Box Plot of {selected_feature}')
            )
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[selected_feature], 
                    marker_color='blue', 
                    opacity=0.7,
                    nbinsx=bin_size
                ),
                row=1, col=1
            )
            
            # Add box plot
            fig.add_trace(
                go.Box(
                    y=df[selected_feature], 
                    boxpoints='outliers',
                    marker_color='lightblue',
                    line_color='darkblue',
                    name=selected_feature
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(height=500, showlegend=True if show_by_class else False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.markdown(f"**Statistics for {selected_feature}:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{df[selected_feature].mean():.4f}")
        col2.metric("Std Dev", f"{df[selected_feature].std():.4f}")
        col3.metric("Min", f"{df[selected_feature].min():.4f}")
        col4.metric("Max", f"{df[selected_feature].max():.4f}")

elif page == "Class Distribution":
    st.header("Class Distribution Analysis")
    
    # Check if Class column exists
    if 'Class' not in df.columns:
        st.error("The dataset does not contain a 'Class' column. Please upload a valid fraud detection dataset.")
        st.stop()
    
    # Calculate class distribution
    class_counts = df['Class'].value_counts()
    class_percentages = df['Class'].value_counts(normalize=True) * 100
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        fig_bar = px.bar(
            x=class_counts.index.astype(str), 
            y=class_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title='Count of Fraud vs Normal Transactions',
            color=class_counts.index.astype(str),
            color_discrete_map={'0': 'green', '1': 'red'}
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = px.pie(
            values=class_counts.values,
            names=['Normal (0)', 'Fraud (1)'],
            title='Percentage of Fraud vs Normal Transactions',
            color=['Normal (0)', 'Fraud (1)'],
            color_discrete_map={'Normal (0)': 'green', 'Fraud (1)': 'red'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Display metrics
    st.markdown("### Class Distribution Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", df.shape[0])
    col2.metric("Normal Transactions (0)", f"{class_counts[0]} ({class_percentages[0]:.2f}%)")
    col3.metric("Fraud Transactions (1)", f"{class_counts[1]} ({class_percentages[1]:.2f}%)")
    
    # Imbalance ratio
    imbalance_ratio = class_counts[0] / class_counts[1]
    st.metric("Imbalance Ratio (Normal:Fraud)", f"{imbalance_ratio:.2f}:1")

elif page == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    
    # Show loading spinner for computationally intensive operation
    with st.spinner("Generating correlation heatmap..."):
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title='Correlation Heatmap of Numerical Features',
            zmin=-1,
            zmax=1
        )
        
        # Update layout
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display highest correlations with Class if available
    if 'Class' in df.columns:
        st.subheader("Top Features Correlated with Class")
        class_correlations = corr_matrix['Class'].drop('Class').sort_values(key=abs, ascending=False)
        top_correlations = class_correlations.head(10)
        
        fig_bar = px.bar(
            x=top_correlations.values,
            y=top_correlations.index,
            orientation='h',
            title='Top 10 Features Correlated with Class',
            labels={'x': 'Correlation Coefficient', 'y': 'Feature'},
            color=top_correlations.values,
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "Model Prediction":
    st.header("Fraud Prediction Model")
    
    # Check if we have the required features
    required_features = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        st.error(f"The dataset is missing the following required features: {', '.join(missing_features)}")
        st.info("Please upload a dataset with features V1-V28, Time, and Amount for model prediction.")
        st.stop()
    
    # Load or train a model (in a real app, you would load a pre-trained model)
    @st.cache_resource
    def load_model():
        # This is a placeholder - in a real application, you would load a pre-trained model
        # For demonstration, we'll create a simple model
        from sklearn.ensemble import IsolationForest
        
        # Train a simple model on the available data
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(df[required_features])
        return model
    
    model = load_model()
    
    st.info("""
    This section demonstrates how a trained machine learning model could be used to predict fraudulent transactions.
    Adjust the feature values using the sliders below and see the prediction result.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Input Feature Values")
        
        # Create two columns for the sliders
        col1, col2 = st.columns(2)
        
        # Generate sliders for each feature
        input_values = {}
        
        with col1:
            for i in range(1, 15):
                feature = f'V{i}'
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].mean())
                input_values[feature] = st.slider(
                    feature, min_val, max_val, default_val,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
        
        with col2:
            for i in range(15, 29):
                feature = f'V{i}'
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].mean())
                input_values[feature] = st.slider(
                    feature, min_val, max_val, default_val,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
            
            # Time and Amount sliders
            time_min = float(df['Time'].min())
            time_max = float(df['Time'].max())
            time_default = float(df['Time'].mean())
            input_values['Time'] = st.slider(
                'Time', time_min, time_max, time_default,
                help=f"Range: {time_min:.2f} to {time_max:.2f}"
            )
            
            amount_min = float(df['Amount'].min())
            amount_max = float(df['Amount'].max())
            amount_default = float(df['Amount'].mean())
            input_values['Amount'] = st.slider(
                'Amount', amount_min, amount_max, amount_default,
                help=f"Range: {amount_min:.2f} to {amount_max:.2f}"
            )
        
        # Submit button
        submitted = st.form_submit_button("Predict")
    
    # Make prediction when form is submitted
    if submitted:
        # Create input array
        input_array = np.array([input_values[feature] for feature in required_features]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Display result
        st.subheader("Prediction Result")
        
        # For IsolationForest, -1 indicates anomaly (fraud), 1 indicates normal
        if prediction[0] == -1:
            st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
            st.error("🚨 Fraudulent Transaction Detected!")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="normal-alert">', unsafe_allow_html=True)
            st.success("✅ Normal Transaction")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show prediction confidence (for demonstration purposes)
        decision_score = model.decision_function(input_array)
        st.metric("Anomaly Score", f"{decision_score[0]:.4f}")
        
        # Interpretation
        st.info("""
        **Interpretation:** 
        - The model uses an Isolation Forest algorithm to detect anomalies.
        - A negative anomaly score indicates a higher likelihood of fraud.
        - Values closer to 0 represent more normal transactions.
        """)

# Add footer
st.markdown("---")
st.markdown("**Fraud Detection Explorer** · Built with Streamlit")