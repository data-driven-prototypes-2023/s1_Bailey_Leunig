import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_extras.buy_me_a_coffee import button
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.bottom_container import bottom


# Custom CSS for styling
css = """
<style>
/* Make the header larger and bold */
h1 {
    font-size: 32px !important;
    font-weight: bold !important;
    color: #2E4053 !important;
}

/* Style the main content area */
.css-1d391kg {
    background-color: #F4F6F6 !important;
    padding: 20px !important;
    border-radius: 10px !important;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1) !important;
}

/* Style the sidebar */
.css-1lcbmhc {
    background-color: #2E4053 !important;
    color: #FFFFFF !important;
    padding: 20px !important;
    border-radius: 10px !important;
}

/* Style the buttons */
.stButton button {
    background-color: #5DADE2 !important;
    color: #FFFFFF !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    font-size: 16px !important;
}

.stButton button:hover {
    background-color: #3498DB !important;
}

/* Style the text inputs */
.css-1cpxqw2 {
    border: 1px solid #D5DBDB !important;
    padding: 10px !important;
    border-radius: 5px !important;
}

/* Style the tables */
.dataframe {
    border-collapse: collapse !important;
    width: 100% !important;
    margin: 20px 0 !important;
    font-size: 18px !important;
    text-align: left !important;
}

.dataframe th, .dataframe td {
    padding: 12px 15px !important;
    border: 1px solid #D5DBDB !important;
}

.dataframe th {
    background-color: #2E4053 !important;
    color: #FFFFFF !important;
}

.dataframe tr:nth-child(even) {
    background-color: #F2F3F4 !important;
}

.dataframe tr:hover {
    background-color: #D5DBDB !important;
}
</style>
"""

# Inject custom CSS
st.markdown(css, unsafe_allow_html=True)

# Load the data
file = '/Users/baileyleunig/Downloads/Barcelona_Airbnb_clean.csv'
df = pd.read_csv(file)

# Convert the rest of the columns to categorical
for col in df.columns:
    if col != 'price_per_night':
        df[col] = df[col].astype('category')

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview and Exploratory Data Analysis", "Data Visualization", "Price Estimator"])

# Overview Section
if section == "Overview and Exploratory Data Analysis":
    st.title("Overview and Exploratory Data Analysis for AirBnB listings in Barcelona")
    st.write("This Streamlit app can be used to display AirBnB listings in **Barcelona** to understand price trends, common apartment features, and to predict the price per night given certain aparment features.")

    # Cofeee Button
    button(username="fake-username", floating=False, width=220, bg_color='#ADD8E6')

    tab1, tab2, tab3 = st.tabs(["Basic Exploration", "Advanced Filtering", "Statistics"])
    with tab1:
        st.header("Basic Exploration of Raw data")

        neighborhood_select = st.selectbox("Neigborhood", ['All'] + df['neighbourhood'].unique().tolist())
        listing_select = st.radio("Listing Type", ['All'] + df['Listing_type'].unique().tolist())

        if neighborhood_select != 'All':
            df = df[df['neighbourhood'] == neighborhood_select]

        if listing_select != 'All':
            df = df[df['Listing_type'] == listing_select]


        st.dataframe(df)
        st.write(f'Number of rows: {df.shape[0]}')

    with tab2:

        st.header("Advanced Filtering and Exploration")
        filtered_df = dataframe_explorer(df)
        st.dataframe(filtered_df, use_container_width=True)
        st.write(f'Number of rows: {filtered_df.shape[0]}')

    with tab3:
        numeric_cols = ['price_per_night', 'number_of_bedrooms', 'number_of_bathrooms', 'number_of_beds', 'minimum_nights']
        # Convert the columns to numeric for summary statistics
        for column in numeric_cols:
            df[column] = df[column].astype('float')
            
        summary_df = df[numeric_cols].describe().transpose().drop(columns=["25%", "50%", "75%"]) 
        st.write(round(summary_df, 2)) 


# Data Visualization Section
if section == "Data Visualization":
    st.header('Data Visualization')
    tab1, tab2 = st.tabs(["Relationship Visualizer", "Feature Inspector"])
    with tab1:
        st.header("Relationship Visualizer")

    # Select x and y axis values
        x_axis = st.selectbox('Select X axis', df.columns)
        y_axis = st.selectbox('Select Y axis', df.columns)

    # Plotting the relationship
        fig, ax = plt.subplots()
        if 'price_per_night' in [x_axis, y_axis]:
            if pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis]):
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
                ax.set_title(f'Scatterplot of {x_axis} vs {y_axis}')
            elif pd.api.types.is_numeric_dtype(df[x_axis]):
                sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
                ax.set_title(f'Boxplot of {x_axis} vs {y_axis}')
            elif pd.api.types.is_numeric_dtype(df[y_axis]):
                sns.boxplot(x=y_axis, y=x_axis, data=df, ax=ax)
                ax.set_title(f'Boxplot of {y_axis} vs {x_axis}')
        else:
            sns.countplot(data=df, x=x_axis, hue=y_axis, ax=ax)
            ax.set_title(f'Countplot of {x_axis} by {y_axis}')

        # Display the plot in Streamlit
        st.pyplot(fig)

    with tab2:
        # Feauture Inspector
        st.header("Feature Inspector")

        # Select columns to plot
        columns = df.columns.tolist()
        selected_column = st.selectbox('Select a feature to plot', columns)

        # Plotting
        fig, ax = plt.subplots()

        if pd.api.types.is_numeric_dtype(df[selected_column]):
            sns.histplot(df[selected_column], ax=ax)
            ax.set_title(f'Histogram of {selected_column}')
        else:
            sns.countplot(x=df[selected_column], ax=ax)
            ax.set_title(f'Count Plot of {selected_column}')

    # Display the plot in Streamlit
        st.pyplot(fig)


# Price Estimator Section
if section == "Price Estimator":

    # Load the trained model
    model = joblib.load('/Users/baileyleunig/Desktop/ESADE/Data Prototyping/pipeline_lr.joblib')

    st.title('AirBnB Price Estimator')
    st.write('This section can be used to estimate the price per night for an AirBnB listing in Barcelona. Use it to make sure you are being charged a fair price!')

    # User input for apartment features
    bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value=4, value=1)
    bathrooms = st.slider('Number of Bathrooms', min_value=1, max_value=8, value=1)
    number_of_beds = st.slider('Number of Beds', min_value=1, max_value=16, value=1)
    minimum_nights = st.slider('Minimum Nights', min_value=1, max_value=32, value=1)
    location = st.selectbox('Location', df['neighbourhood'].unique())
    listing_type = st.selectbox('Listing Type', df['Listing_type'].unique())
    ada_compliant = st.selectbox('ADA Compliant', [True, False])
    pet_friendly = st.selectbox('Pet Friendly', [True, False])
    washing_machine = st.selectbox('Washing Machine', [True, False])
    welcome_gift = st.selectbox('Welcome Gift', [True, False])
    smoking = st.selectbox('Smoking Allowed', [True, False])
    smoke_detector = st.selectbox('Smoke Detector', [True, False])
    balcony = st.selectbox('Balcony', [True, False])
    ac = st.selectbox('AC', [True, False])
    oven = st.selectbox('Oven', [True, False])
    jacuzzi = st.selectbox('Jacuzzi', [True, False])
    wifi = st.selectbox('WI-FI', [True, False])
    smoking = st.selectbox('Smoking ', [True, False])

    # Create a DataFrame from user input
    input_data = pd.DataFrame([{
        'number_of_beds': number_of_beds,
        'ADA_Compliant': ada_compliant,
        'Pet_Friendly': pet_friendly,
        'Washing_Machine': washing_machine,
        'Welcome_Gift': welcome_gift,
        'number_of_bedrooms': bedrooms,
        'Smoking': smoking,
        'minimum_nights': minimum_nights,
        'Smoke_detector': smoke_detector,
        'neighbourhood': location,
        'number_of_bathrooms': bathrooms,
        'Balcony': balcony,
        'AC': ac,
        'Listing_type': listing_type,
        'Oven': oven,
        'Jacuzzi': jacuzzi,
        'WI-FI': wifi,
        'Smoking ': smoking
    }])

    # Make prediction
    with bottom():
        if st.button('Estimate Price'):
            prediction = model.predict(input_data)[0]
            st.write(f'Estimated Price per Night: ${prediction:.2f}')








