"""Web search and scraping tools for stock research"""
from crewai.tools import BaseTool
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo (no API key required)"""

    name: str = "web_search"
    description: str = (
        "Search for information about stocks, companies, and financial news. "
        "Input: search query string."
    )

    def _run(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No search results found."
            output = []
            for r in results:
                output.append(f"**{r.get('title', 'No title')}**")
                output.append(f"URL: {r.get('href', 'N/A')}")
                output.append(f"{r.get('body', 'No description')}\n")
            return "\n".join(output)
        except Exception as e:
            return f"Search error: {str(e)}"


class WebScrapeTool(BaseTool):
    """Scrape content from a web page"""

    name: str = "web_scrape"
    description: str = (
        "Fetch and extract text content from a URL. "
        "Input: full URL to scrape."
    )

    def _run(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; StockAnalysisBot/1.0)"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()

            # Get text content
            text = soup.get_text(separator="\n", strip=True)

            # Limit to first 4000 chars to avoid token limits
            if len(text) > 4000:
                text = text[:4000] + "\n...[truncated]"

            return text
        except Exception as e:
            return f"Scrape error: {str(e)}"
