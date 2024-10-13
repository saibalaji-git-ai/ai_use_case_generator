import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from market_research_agent import run_market_research_crew, compare_reports  
import json
import sys
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

st.write("Current Python version:", sys.version)

# Set up the Streamlit page configuration
st.set_page_config(page_title="Advanced Market Research & Use Case Generation Agent", layout="wide")

# Title for the application
st.title("Advanced Market Research & Use Case Generation Agent")

# Sidebar for mode selection
mode = st.sidebar.radio("Select Mode", ["Single Analysis", "Comparison"])

if mode == "Single Analysis":
    # Input field for single company or industry analysis
    company_or_industry = st.text_input("Enter a company or industry to research:")

    if st.button("Generate Report"):
        if company_or_industry:
            with st.spinner("Generating comprehensive report... This may take a few minutes."):
                # Call the function to perform market research
                result = run_market_research_crew(company_or_industry)

                st.subheader(f"Comprehensive Report for: {company_or_industry}")

                # Display the full report
                st.markdown(result)

                # Parse the result to extract structured data
                use_cases = []
                datasets = []
                for line in result.split('\n'):
                    if line.startswith("Use Case:"):
                        use_cases.append(line.split(":")[1].strip())
                    elif line.startswith("Dataset:"):
                        datasets.append(line.split(":")[1].strip())

                # Create a bar chart of use cases
                if use_cases:
                    use_case_df = pd.DataFrame({'Use Case': use_cases, 'Count': [1] * len(use_cases)})
                    fig = px.bar(use_case_df, x='Use Case', y='Count', title='Proposed AI/GenAI Use Cases')
                    st.plotly_chart(fig)

                # Create a pie chart of dataset sources
                if datasets:
                    dataset_sources = [d.split()[0] for d in datasets]  # Assuming the first word is the source (e.g., Kaggle, GitHub)
                    source_counts = pd.Series(dataset_sources).value_counts()
                    fig = px.pie(values=source_counts.values, names=source_counts.index, title='Dataset Sources')
                    st.plotly_chart(fig)

                # Generate and display a word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(result)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

        else:
            st.warning("Please enter a company or industry name.")

elif mode == "Comparison":
    st.subheader("Compare Multiple Companies or Industries")

    # Input for multiple companies or industries
    companies_or_industries = st.text_area("Enter companies or industries to compare (one per line):")

    if st.button("Compare"):
        if companies_or_industries:
            items = [item.strip() for item in companies_or_industries.split('\n') if item.strip()]
            if len(items) > 1:
                with st.spinner("Generating comparison... This may take several minutes."):
                    comparison_result = compare_reports(items)

                    # Display similarity heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=comparison_result['similarities'],
                        x=items,
                        y=items,
                        colorscale='Viridis'))
                    fig.update_layout(title='Similarity Between Reports')
                    st.plotly_chart(fig)

                    # Display individual reports
                    for report in comparison_result['reports']:
                        st.subheader(f"Report for {report['name']}")
                        st.markdown(report['report'])
            else:
                st.warning("Please enter at least two companies or industries for comparison.")
        else:
            st.warning("Please enter companies or industries to compare.")

# Add a section for additional resources
st.sidebar.header("Additional Resources")
st.sidebar.markdown("""
- [McKinsey & Company - AI insights](https://www.mckinsey.com/featured-insights/artificial-intelligence)
- [Deloitte - AI services](https://www2.deloitte.com/us/en/pages/deloitte-analytics/solutions/artificial-intelligence-services.html)
- [Nexocode - AI solutions](https://nexocode.com/blog/posts/artificial-intelligence-in-business/)
""")

# Add a feedback section with sentiment analysis
st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Please provide any feedback or suggestions for improvement:")
if st.sidebar.button("Submit Feedback"):
    # Perform sentiment analysis on the feedback
    sentiment = TextBlob(feedback).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

    # Here you would typically send this feedback and sentiment to a database
    st.sidebar.success(f"Thank you for your feedback! Sentiment: {sentiment_label}")

    # Display feedback statistics (this is a placeholder - in a real application, you'd aggregate this data)
    st.sidebar.subheader("Feedback Statistics")
    feedback_data = {"Positive": 65, "Neutral": 20, "Negative": 15}  # Placeholder data
    fig = px.pie(values=list(feedback_data.values()), names=list(feedback_data.keys()), title='Overall Feedback Sentiment')
    st.sidebar.plotly_chart(fig)
