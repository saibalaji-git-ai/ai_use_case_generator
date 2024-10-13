import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
from tavily import TavilyClient
from langchain.llms import OpenAI
import logging
import json
from functools import lru_cache
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Get API key from environment or directly specify it
openai_api_key = os.getenv("OPENAI_API_KEY", "sk-JFIPceRIpa2uuI4uFxGVGM3iSewhC8kuQplm_c1LW2T3BlbkFJ3eruqsrIsS8CPlHpiKZaS8pNr_jfxjzG3j8tde2BUA")

# Load the API key from environment variables or specify it directly
serpapi_api_key = os.getenv("SERPAPI_API_KEY", "d92d9ac959f6753f2d987e807cccbebea5663ec748adc9c1d6879fd46eb5ae79")

# Initialize language model and search tools
llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
tavily_client = TavilyClient(api_key = os.getenv("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")  # Make sure this is set in your environment



# Define tools
tavily_search = Tool(
    name="Tavily Search",
    func=lambda q: tavily_client.search(query=q, search_depth="advanced").get("results", []),
    description="Useful for searching the web for up-to-date information on companies, industries, and market trends."
)

serpapi_search = Tool(
    name="SerpAPI Search",
    func=serpapi.run,
    description="Useful for general web searches and finding specific information about companies and industries."
)

# Define agents (same as before)
industry_researcher = Agent(
    role='Industry Researcher',
    goal='Research and analyze the given company or industry thoroughly',
    backstory="""You are an expert in market research and industry analysis. Your task is to gather 
    comprehensive information about the given company or industry, including key players, market trends, 
    and strategic focus areas.""",
    verbose=True,
    allow_delegation=False,
    tools=[tavily_search, serpapi_search],
    llm=llm
)

use_case_generator = Agent(
    role='Use Case Generator',
    goal='Generate relevant AI and GenAI use cases for the company or industry',
    backstory="""You are an AI solutions architect with extensive knowledge of how AI, ML, and GenAI 
    can be applied in various industries. Your task is to propose innovative and practical use cases 
    that can improve processes, enhance customer satisfaction, and boost operational efficiency.""",
    verbose=True,
    allow_delegation=False,
    tools=[tavily_search],
    llm=llm
)

resource_collector = Agent(
    role='Resource Collector',
    goal='Find and collect relevant datasets and resources for the proposed use cases',
    backstory="""You are a data scientist specializing in finding and evaluating datasets and resources 
    for AI/ML projects. Your task is to identify relevant datasets from platforms like Kaggle, HuggingFace, 
    and GitHub that can support the implementation of the proposed use cases.""",
    verbose=True,
    allow_delegation=False,
    tools=[serpapi_search],
    llm=llm
)

proposal_writer = Agent(
    role='Proposal Writer',
    goal='Compile a comprehensive final proposal with all findings and recommendations',
    backstory="""You are a seasoned consultant with a knack for creating compelling and actionable proposals. 
    Your task is to synthesize all the information gathered by the other agents and create a clear, 
    well-structured final proposal that highlights the most promising AI/GenAI use cases and their potential impact.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define tasks (same as before)
def create_tasks(company_or_industry):
    return [
        Task(
            description=f"""Research {company_or_industry} thoroughly. Identify key players, market trends, 
            and strategic focus areas. Pay special attention to any existing AI/ML initiatives or opportunities 
            for AI adoption. Provide a comprehensive report of your findings.""",
            agent=industry_researcher
        ),
        Task(
            description=f"""Based on the industry research, generate at least 5 innovative and relevant AI/GenAI 
            use cases for {company_or_industry}. Focus on solutions that can significantly improve processes, 
            enhance customer satisfaction, or boost operational efficiency. For each use case, provide a brief 
            description and potential impact.""",
            agent=use_case_generator
        ),
        Task(
            description="""For each proposed use case, find relevant datasets or resources from Kaggle, HuggingFace, 
            or GitHub that could be used to develop the solution. Provide links and brief descriptions of how each 
            dataset or resource could be utilized.""",
            agent=resource_collector
        ),
        Task(
            description="""Compile all the gathered information into a comprehensive final proposal. The proposal 
            should include:
            1. An executive summary of the industry analysis
            2. A list of the top 5 AI/GenAI use cases, with detailed descriptions and potential impact
            3. For each use case, include relevant datasets or resources with links
            4. A section on implementation feasibility and next steps
            5. References for all sources used in the research
            Ensure the proposal is well-structured, clear, and provides actionable insights.""",
            agent=proposal_writer
        )
    ]

# Create the crew
crew = Crew(
    agents=[industry_researcher, use_case_generator, resource_collector, proposal_writer],
    tasks=[],
    verbose=True,  # Change this to True or False
    process=Process.sequential
)

# Implement caching
@lru_cache(maxsize=100)
def cached_market_research(company_or_industry: str):
    crew.tasks = create_tasks(company_or_industry)
    result = crew.kickoff()
    return result

# Main function to run the multi-agent system
def run_market_research_crew(company_or_industry: str):
    try:
        result = cached_market_research(company_or_industry)
        return result
    except Exception as e:
        logging.error(f"Error in run_market_research_crew: {str(e)}")
        return f"An error occurred: {str(e)}"

# Function to compare multiple companies or industries
def compare_reports(companies_or_industries):
    reports = []
    for item in companies_or_industries:
        report = run_market_research_crew(item)
        reports.append({"name": item, "report": report})
    
    # Use TF-IDF and cosine similarity to find similarities between reports
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([r['report'] for r in reports])
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    comparison_result = {
        "reports": reports,
        "similarities": cosine_similarities.tolist()
    }
    
    return comparison_result

if __name__ == "__main__":
    company_or_industry = input("Enter a company or industry to research: ")
    result = run_market_research_crew(company_or_industry)
    print(result)
