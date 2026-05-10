import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Page Title
st.title("AI Powered Analytics Dashboard")

st.markdown(
    "Analyze datasets, generate visualizations, and get AI-powered insights instantly."
)

# File Upload
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Sidebar Filters
   
    st.sidebar.header("Filter Data")

    filter_column = st.sidebar.selectbox(
        "Select Column to Filter",
        df.columns
    )

    unique_values = df[filter_column].unique()

    selected_value = st.sidebar.selectbox(
        "Select Value",
        ["All"] + list(unique_values)
    )

    # Apply Filter

    if selected_value == "All":
        filtered_df = df

    else:
        filtered_df = df[
            df[filter_column] == selected_value
        ]


    # Data Preview

    st.subheader("Data Preview")

    st.dataframe(filtered_df)

    # Dataset Information

    st.subheader("Dataset Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric(
            "📄 Rows",
            filtered_df.shape[0]
        )

    with info_col2:
        st.metric(
            "📊 Columns",
            filtered_df.shape[1]
        )

    with info_col3:
        st.metric(
            "⚠ Missing Values",
            filtered_df.isnull().sum().sum()
        )

    st.write("### Column Names")

    st.table(
        pd.DataFrame(
            filtered_df.columns,
            columns=["Column Names"]
        )
    )

    st.divider()

    # KPI Metrics

    st.subheader("Key Metrics")

    num_cols = filtered_df.select_dtypes(
        include='number'
    ).columns

    if len(num_cols) > 0:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "📄 Total Rows",
                filtered_df.shape[0]
            )

        with col2:
            st.metric(
                "📊 Total Columns",
                filtered_df.shape[1]
            )

        with col3:
            st.metric(
                "🔢 Numeric Columns",
                len(num_cols)
            )

    st.divider()

    # Data Visualization

    st.subheader("Data Visualization")

    cat_cols = filtered_df.select_dtypes(
        include='object'
    ).columns

    num_cols = filtered_df.select_dtypes(
        include='number'
    ).columns

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Pie Chart",
            "Histogram",
            "Count Plot"
        ]
    )

    if chart_type == "Bar Chart":

        x_axis = st.selectbox(
            "Select Categorical Column",
            cat_cols,
            key="bar_x"
        )

        y_axis = st.selectbox(
            "Select Numeric Column",
            num_cols,
            key="bar_y"
        )

        fig = px.bar(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=x_axis
        )

    elif chart_type == "Line Chart":

        x_axis = st.selectbox(
            "Select X-axis",
            cat_cols,
            key="line_x"
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            num_cols,
            key="line_y"
        )

        fig = px.line(
            filtered_df,
            x=x_axis,
            y=y_axis,
            markers=True
        )

    elif chart_type == "Pie Chart":

        names_col = st.selectbox(
            "Select Category",
            cat_cols,
            key="pie_names"
        )

        values_col = st.selectbox(
            "Select Numeric Column",
            num_cols,
            key="pie_values"
        )

        fig = px.pie(
            filtered_df,
            names=names_col,
            values=values_col
        )

    elif chart_type == "Histogram":

        histogram_col = st.selectbox(
            "Select Numeric Column",
            num_cols,
            key="hist_col"
        )

        fig = px.histogram(
            filtered_df,
            x=histogram_col
        )

    elif chart_type == "Count Plot":

        x_axis = st.selectbox(
            "Select First Category",
            cat_cols,
            key="count_x"
        )

        color_axis = st.selectbox(
            "Select Second Category",
            cat_cols,
            key="count_color"
        )

        fig = px.histogram(
            filtered_df,
            x=x_axis,
            color=color_axis,
            barmode="group",
        )

    fig.update_layout(
        title=chart_type,
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Correlation Heatmap

    st.subheader("Correlation Heatmap")

    correlation = filtered_df.corr(numeric_only=True)

    heatmap = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    # Download Filtered Data

    st.subheader("Download Filtered Data")

    csv = filtered_df.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(
        label="Download Filtered CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    st.divider()

    # AI Generated Insights

    st.subheader("AI Generated Insights")

    if st.button("Generate Insights"):

        try:

            with st.spinner(
                "Generating insights..."
            ):

                model = genai.GenerativeModel(
                    "models/gemini-2.5-flash"
                )

                prompt = f"""
                Analyze this dataset and provide useful business insights.

                Dataset Columns:
                {filtered_df.columns.tolist()}

                Sample Data:
                {filtered_df.head().to_string()}
                """

                response = model.generate_content(
                    prompt
                )

                st.write(response.text)

        except Exception as e:

            st.error(f"Error: {e}")

    st.divider()

    # Chat With Your Data

    st.subheader("Chat With Your Data")

    user_question = st.text_input(
        "Ask a question about your dataset"
    )

    if st.button("Get Answer"):

        try:

            with st.spinner(
                "Analyzing your question..."
            ):

                model = genai.GenerativeModel(
                    "models/gemini-2.5-flash"
                )

                chat_prompt = f"""
                You are a data analyst.

                Answer the user's question based on this dataset.

                Dataset Columns:
                {filtered_df.columns.tolist()}

                Sample Data:
                {filtered_df.head(20).to_string()}

                User Question:
                {user_question}
                """

                response = model.generate_content(
                    chat_prompt
                )

                st.write(response.text)

        except Exception as e:

            st.error(f"Error: {e}")