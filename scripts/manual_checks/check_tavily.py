import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
results = client.search("remote AI internships 2025", max_results=3)

for r in results["results"]:
    print(r["title"], "→", r["url"])