import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='SmartML Studio', page_icon='🧠')
st.markdown("""
<h1 style='
color:#22D3EE;
font-size:42px;
font-weight:800;
margin-bottom:0;
'>
🧠 SmartML Studio
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p style='
color:#94A3B8;
font-size:15px;
margin-top:-10px;
margin-bottom:25px;
'>
Interactive AutoML dashboard for preprocessing, analytics, model training, and evaluation.
</p>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* Main App Background */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22 ;
    border-right: 1px solid #30363D;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #F0F6FC !important;
}

            
/* Custom Headers */
h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #22D3EE !important;
    margin-top: 25px !important;
}

[data-testid="metric-container"] {
    background: linear-gradient(
        145deg,
        #161B22,
        #111827
    );

    border: 1px solid #30363D;

    padding: 18px;

    border-radius: 16px;

    box-shadow:
        0 4px 20px rgba(0,0,0,0.25);

    transition: 0.3s ease;
}

/* Tables */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

.stButton>button,
.stDownloadButton>button {

    background: linear-gradient(
        135deg,
        #06B6D4,
        #2563EB
    );

    color: white;

    border: none;

    border-radius: 12px;

    padding: 10px 18px;

    font-weight: 700;

    transition: 0.3s ease;
}

.stButton>button:hover,
.stDownloadButton>button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 4px 15px rgba(34,211,238,0.3);
}

.stButton>button:hover {
    background-color: #2EA043;
    color: white;
}

/* Download Buttons */
.stDownloadButton>button {
    background-color: #1F6FEB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    font-weight: 600;
}


/* Expander */
.streamlit-expanderHeader {
    background-color: #161B22;
    border-radius: 8px;
    color: white !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #161B22;
    border-radius: 10px;
}

/* Active selected range only */
.stSlider [data-baseweb="slider"] > div > div > div > div {
    background-color: #22D3EE !important;
}

/* Slider handle */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #22D3EE !important;
    border-color: #22D3EE !important;
}
            
/* Status Box */
[data-testid="stStatusWidget"] {
    background-color: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 10px;
}

/* Chart Containers */
.element-container {
    border-radius: 12px;
}

/* Reduce top padding */
.block-container {
    padding-top: 2rem;
}

/* Horizontal line */
hr {
    border-color: #30363D;
}

    .stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid #30363D;
}
            
.streamlit-expanderHeader {
    background-color: #161B22;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #30363D;
}
            
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

with st.expander('About SmartML Studio'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload, data pre-processing, ML model building and post-model analysis.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Drug solubility data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.sidebar.markdown("""
    <h3 style='
    color:#22D3EE;
    font-size:26px;
    font-weight:800;
    margin-bottom:20px;
    '>
    ⚡ SmartML Controls
    </h3>
    """, unsafe_allow_html=True)
   

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)
   
  

    # Download example data
    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')
    example_csv = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')
    csv = convert_df(example_csv)
    st.download_button(
        label="Download sample CSV",
        data=csv,
        file_name='delaney_solubility_with_descriptors.csv',
        mime='text/csv',
    )

    # Select example data
    st.markdown('**1.2. Use sample data**')
    example_data = st.toggle('Load sample data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

    
    st.markdown("""
    <style>
    .model-label {
        font-size: 28px;
        font-weight: bold;
        color: #00E5FF;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    model_option = st.selectbox(
    "Select ML Model",
    ["Random Forest", "Decision Tree", "Linear Regression", "SVM"]
  )

    st.markdown('<div class="model-label">🚀 Select ML Model</div>', unsafe_allow_html=True)

    model = st.selectbox(
        "",
        ["Random Forest", "Linear Regression", "SVM", "Decision Tree"]
    )
    
    st.header('⚙️ Model Configuration')
    

    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)


    st.subheader('🤖 Learning Parameters')
    with st.expander('See parameters'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)


    st.subheader('⚡ General Parameters')
    with st.expander('See parameters', expanded=False):
        parameter_random_state = st.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.select_slider('Performance measure (criterion)', options=['squared_error', 'absolute_error', 'friedman_mse'])
        parameter_bootstrap = st.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])

    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Initiate the model building process
if uploaded_file or example_data: 
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)

        # Track preprocessing information
        initial_rows = df.shape[0]
        initial_cols = df.shape[1]

        missing_values_before = df.isnull().sum().sum()
        duplicate_rows_before = df.duplicated().sum()

        # Remove duplicates
        df = df.drop_duplicates()

        # Remove fully empty columns
        df = df.dropna(axis=1, how='all')

        # Remove rows with missing values
        df = df.dropna()

        missing_values_after = df.isnull().sum().sum()
        duplicate_rows_after = df.duplicated().sum()

        # Features and target
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].copy()

        # Encode categorical features
        X = pd.get_dummies(X)

        # Detect classification automatically
        is_classification = False

        if y.dtype == 'object' or y.nunique() <= 10:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)))
            is_classification = True

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X = pd.DataFrame(X_scaled, columns=X.columns)

        # Make sure column names are strings
        X.columns = X.columns.astype(str)
            
        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)
    
        st.write("Model training ...")
        time.sleep(sleep_time)

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]
            
        if is_classification:

            if model_option == 'Random Forest':
                model = RandomForestClassifier(
                    n_estimators=parameter_n_estimators,
                    random_state=parameter_random_state
                )

            elif model_option == 'Decision Tree':
                model = DecisionTreeClassifier(
                    random_state=parameter_random_state
                )

            elif model_option == 'SVM':
                model = SVC()

            else:
                model = LogisticRegression()

        else:

            if model_option == 'Random Forest':
                model = RandomForestRegressor(
                    n_estimators=parameter_n_estimators,
                    max_features=parameter_max_features,
                    min_samples_split=parameter_min_samples_split,
                    min_samples_leaf=parameter_min_samples_leaf,
                    random_state=parameter_random_state,
                    criterion=parameter_criterion,
                    bootstrap=parameter_bootstrap,
                    oob_score=parameter_oob_score
                )

            elif model_option == 'Linear Regression':
                model = LinearRegression()

            elif model_option == 'Decision Tree':
                model = DecisionTreeRegressor(
                    random_state=parameter_random_state
                )

            else:
                model = SVR()

        model.fit(X_train, y_train)

        if is_classification:
                st.success("✅ Classification workflow detected")
        else:
                st.info("📈 Regression workflow detected")            
        
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Convert predictions safely
        y_train_pred = np.array(y_train_pred)
        y_test_pred = np.array(y_test_pred)
            
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        
        if is_classification:

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            rf_results = pd.DataFrame([
                model_option,
                train_acc,
                test_acc
            ]).transpose()

            rf_results.columns = [
                'Method',
                'Training Accuracy',
                'Test Accuracy'
            ]

        else:

            train_mse = mean_squared_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)

            test_mse = mean_squared_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            rf_results = pd.DataFrame([
                model_option,
                train_mse,
                train_r2,
                test_mse,
                test_r2
            ]).transpose()

            rf_results.columns = [
                'Method',
                'Training MSE',
                'Training R2',
                'Test MSE',
                'Test R2'
            ]

        rf_results = rf_results.round(3)
        
        status.update(label="Status", state="complete", expanded=False)  
        
        # -----------------------------------
    # DATA QUALITY INSIGHTS
    # -----------------------------------

    st.header('🧹 Data Quality Insights', divider='rainbow')

    quality_col = st.columns(4)

    quality_col[0].metric(
        "Missing Values",
        int(missing_values_before)
    )

    quality_col[1].metric(
        "Duplicates Removed",
        int(duplicate_rows_before - duplicate_rows_after)
    )

    quality_col[2].metric(
        "Numerical Columns",
        len(df.select_dtypes(include=np.number).columns)
    )

    quality_col[3].metric(
        "Categorical Columns",
        len(df.select_dtypes(exclude=np.number).columns)
    )

            
        # -----------------------------------
    # DATASET PROFILE
    # -----------------------------------

    st.subheader('📋 Dataset Profile', divider='rainbow')

    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        summary_df,
        use_container_width=True
    )
    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)



    # Zip dataset files
    df.to_csv('dataset.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
    
    list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
                label='Download ZIP',
                data=datazip,
                file_name="dataset.zip",
                mime="application/octet-stream"
                )
    
    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
    
    # Display feature importance plot
    
    if hasattr(model, 'feature_importances_'):
     importances = model.feature_importances_
    else:
     importances = np.zeros(X.shape[1])

    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})
    
    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
             x='value:Q',
             y=alt.Y('feature:N', sort='-x')
           ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    if is_classification:

        st.header('Classification Metrics', divider='rainbow')

        cm = confusion_matrix(y_test, y_test_pred)

        st.write('Confusion Matrix')
        st.dataframe(pd.DataFrame(cm))

        st.write('Classification Report')
        report = classification_report(
            y_test,
            y_test_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

    st.header('Prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'
        
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'
    
    df_prediction = pd.concat([df_train, df_test], axis=0)
    
    prediction_col = st.columns((2, 0.2, 3))
    
    # Display dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                        x='actual',
                        y='predicted',
                        color='class'
                  )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    
# Ask for CSV upload if none is detected
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')

st.markdown("""
<hr>

<p style='
text-align:center;
color:#64748B;
font-size:14px;
'>
Built with Streamlit • Scikit-learn • Altair • SmartML Studio
</p>
""", unsafe_allow_html=True)