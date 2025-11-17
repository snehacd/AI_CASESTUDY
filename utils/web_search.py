"""
Web Search Integration
Uses SerpAPI for real-time web searches
"""
import requests
from config.config import SERPAPI_KEY


class WebSearch:
    """Handles web search functionality"""

    def __init__(self, api_key=None):
        """
        Initialize web search

        Args:
            api_key: SerpAPI key
        """
        self.api_key = api_key or SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"

    def search(self, query, num_results=5):
        """
        Perform web search

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results
        """
        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results,
                "engine": "google"
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract relevant results
            results = []
            if "organic_results" in data:
                for result in data["organic_results"][:num_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "")
                    })

            return results
        except requests.exceptions.Timeout:
            return [{"error": "Search request timed out"}]
        except requests.exceptions.RequestException as e:
            return [{"error": f"Search error: {str(e)}"}]
        except Exception as e:
            return [{"error": f"Unexpected error: {str(e)}"}]

    def format_results(self, results):
        """
        Format search results as context string

        Args:
            results: List of search results

        Returns:
            Formatted string
        """
        if not results:
            return "No search results found."

        if "error" in results[0]:
            return results[0]["error"]

        context = "Web Search Results:\n\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['title']}\n"
            context += f"   {result['snippet']}\n"
            context += f"   Source: {result['link']}\n\n"

        return context
