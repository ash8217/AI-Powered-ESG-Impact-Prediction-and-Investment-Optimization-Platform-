import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, fbeta_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from pulp import LpProblem, LpVariable, LpMaximize, lpSum
from imblearn.over_sampling import SMOTE
import streamlit as st
import plotly.express as px


# Load the CSV files
country_data = pd.read_csv('ESGCountry.csv')
series_data = pd.read_csv('ESGSeries.csv')
series_time_data = pd.read_csv('ESGSeries-Time.csv')
data = pd.read_csv('ESGData.csv')
footnote = pd.read_csv('ESGFootNote.csv')
country_series = pd.read_csv('ESGCountry-Series.csv')


# Clean column names by stripping any extra spaces
country_data.columns = country_data.columns.str.strip()
series_data.columns = series_data.columns.str.strip()
series_time_data.columns = series_time_data.columns.str.strip()
data.columns = data.columns.str.strip()
footnote.columns = footnote.columns.str.strip()
country_series.columns = country_series.columns.str.strip()


# Merge the data with other datasets
merged_data = pd.merge(data, series_time_data, left_on='Indicator Code', right_on='SeriesCode')
merged_data = pd.merge(merged_data, country_data, on='Country Code', how='left')


# Check and add 'Sector' if it's missing in merged_data
if 'Sector' not in merged_data.columns:
    if 'Sector' in country_data.columns:
        merged_data = pd.merge(merged_data, country_data[['Country Code', 'Sector']], on='Country Code', how='left')
    else:
        merged_data['Sector'] = 'Unknown'

# Reshape data for time series analysis
reshaped_data = merged_data.melt(
    id_vars=['Country Name', 'Country Code', 'Sector', 'Indicator Name', 'Indicator Code', 'DESCRIPTION'],
    var_name='Year',
    value_name='Value'
)

# Convert 'Year' to numeric and drop rows with missing values
reshaped_data['Year'] = pd.to_numeric(reshaped_data['Year'], errors='coerce')
reshaped_data = reshaped_data.dropna(subset=['Year', 'Value'])

# Normalize the ESG values using MinMaxScaler
scaler = MinMaxScaler()
reshaped_data['Normalized Value'] = scaler.fit_transform(reshaped_data[['Value']])


country_year_data = reshaped_data.groupby(['Country Name', 'Year']).agg({'Normalized Value': 'mean'}).reset_index()


reshaped_data = reshaped_data.dropna(subset=['Sector', 'Year'])
reshaped_data = pd.get_dummies(reshaped_data, columns=['Sector'], drop_first=True)

# Define features and target variable
X = reshaped_data[['Year'] + [col for col in reshaped_data.columns if 'Sector' in col]]
y = reshaped_data['Normalized Value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Rank the countries based on 'Normalized Value'
data_sorted = reshaped_data.sort_values(by='Normalized Value', ascending=False)
data_sorted['Rank'] = range(1, len(data_sorted) + 1)

# Save the ranked projects to a new CSV file
data_sorted.to_csv('ranked_esg_projects.csv', index=False)
print(data_sorted.head())

# Streamlit Dashboard
st.title('Green Finance Optimization Platform')

# Allow the user to define the total budget
total_budget = st.number_input("Enter the total budget:", min_value=100000, max_value=10000000, step=50000, value=1000000)

# File upload section for the user input CSV file
uploaded_file = st.file_uploader("Upload ESG Data (CSV)", type='csv')
if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", uploaded_data)

    # Merge the uploaded project data with the existing ESG data for scoring
    merged_projects = pd.merge(uploaded_data, country_data, on='Country Code', how='left')

    # Normalize project data (e.g., using investment amount or impact)
    project_scaler = MinMaxScaler()
    merged_projects['Normalized Impact'] = project_scaler.fit_transform(merged_projects[['Projected Impact']])

    # Prepare the input features for the prediction model
    project_features = merged_projects[['Year']]  # Using 'Year' for simplicity
    merged_projects['Predicted ESG Score'] = model.predict(project_features)

    # Rank the projects based on the predicted ESG scores
    merged_projects['Rank'] = merged_projects['Predicted ESG Score'].rank(ascending=False, method='min')

    # Display the ranked projects
    st.subheader("Ranked Projects Based on Predicted ESG Score")
    st.dataframe(merged_projects[['Project ID', 'Project Name', 'Country Code', 'Projected Impact', 'Predicted ESG Score', 'Rank']])

    # Visualizations
    st.subheader("Visualizations")
    scatter_fig = px.scatter(
        merged_projects,
        x='Projected Impact',
        y='Predicted ESG Score',
        color='Rank',
        title="Projected Impact vs Predicted ESG Score",
        labels={'Projected Impact': 'Projected Impact', 'Predicted ESG Score': 'Predicted ESG Score'},
    )
    st.plotly_chart(scatter_fig)

    top_projects = merged_projects.nlargest(10, 'Predicted ESG Score')
    bar_fig = px.bar(
        top_projects,
        x='Project Name',
        y='Predicted ESG Score',
        color='Rank',
        title="Top 10 Projects by Predicted ESG Score",
        labels={'Predicted ESG Score': 'Predicted ESG Score'},
    )
    st.plotly_chart(bar_fig)

    # Optimization with Risk and Diversification
    st.subheader("Optimization with Risk and Diversification")
    lp_problem = LpProblem("Maximize_ESG_Score", LpMaximize)

    merged_projects['Risk'] = np.random.uniform(0.1, 1.0, size=len(merged_projects))

    project_variables = {row['Project ID']: LpVariable(row['Project ID'], lowBound=0, cat='Continuous') for index, row in merged_projects.iterrows()}

    lp_problem += lpSum([
        project_variables[row['Project ID']] * (row['Predicted ESG Score'] / row['Risk'])
        for index, row in merged_projects.iterrows()
    ])

    lp_problem += lpSum([project_variables[row['Project ID']] for index, row in merged_projects.iterrows()]) <= total_budget

    lp_problem.solve()

    optimized_results = {row['Project ID']: project_variables[row['Project ID']].varValue for index, row in merged_projects.iterrows()}
    merged_projects['Optimized Allocation'] = merged_projects['Project ID'].map(optimized_results)

    st.dataframe(merged_projects[['Project ID', 'Project Name', 'Predicted ESG Score', 'Risk', 'Optimized Allocation']])

    # Improved Binary Classification
    threshold = reshaped_data['Normalized Value'].median()
    reshaped_data['Target'] = (reshaped_data['Normalized Value'] > threshold).astype(int)

    X = reshaped_data[['Year'] + [col for col in reshaped_data.columns if 'Sector' in col]]
    y = reshaped_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_dist = {
        'n_estimators': [10, 20, 30],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                       n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)
    random_search.fit(X_resampled, y_resampled)

    best_rf_model = random_search.best_estimator_

    y_pred = best_rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f2_score = fbeta_score(y_test, y_pred, beta=2)

    print("Accuracy:", accuracy)
    print("F2 Score:", f2_score)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))