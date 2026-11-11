"""Tool schemas (for the LLM) and their Python implementations."""
import html as html_lib
import io
import math
import re
import sys

import requests

TOOLS = [
    {
        "name": "search_documents",
        "description": "Semantic search over the user's uploaded documents. Use for any "
        "question about the content of their files or notes.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "e.g. 'Bengaluru'"}},
            "required": ["city"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for current events or facts that are NOT in the "
        "user's documents.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": "Fetch and read the full text content from a web URL or article link.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The HTTP or HTTPS URL to read"}},
            "required": ["url"],
        },
    },
    {
        "name": "execute_python",
        "description": "Execute a Python code snippet to perform exact calculations, math, formulas, data aggregation, or transformations.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code or expression to evaluate"}},
            "required": ["code"],
        },
    },
    {
        "name": "summarize_document",
        "description": "Retrieve the full content of an uploaded document by its exact filename to generate an executive summary or overview.",
        "input_schema": {
            "type": "object",
            "properties": {"source": {"type": "string", "description": "The exact filename of the document"}},
            "required": ["source"],
        },
    },
]


def get_weather(city: str) -> str:
    """Open-Meteo: free, no API key."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find a place called '{city}'."
        loc = geo["results"][0]
        w_res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            },
            timeout=10,
        ).json()
        w = w_res.get("current")
        if not w:
            return f"Weather data currently unavailable for '{city}'."
        return (
            f"Weather in {loc['name']}, {loc.get('country', '')}: "
            f"{w['temperature_2m']}°C, humidity {w['relative_humidity_2m']}%, "
            f"wind {w['wind_speed_10m']} km/h."
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


def web_search(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No web results found."
        return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)
    except Exception as e:
        return f"Web search failed: {e}"


def read_url(url: str, max_chars: int = 4000) -> str:
    """Fetch and extract readable text content from any webpage URL."""
    try:
        url = url.strip().strip(".,;:()[]{}<>\"'")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()

        html = resp.text
        cleaned = re.sub(
            r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>",
            " ",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", " ", cleaned)
        text = html_lib.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

        if not text:
            return f"No readable text could be extracted from {url}."

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [Content truncated at {max_chars} characters]"

        return f"=== Content from {url} ===\n\n{text}"
    except Exception as e:
        return f"Failed to fetch content from {url}: {e}"


def execute_python(code: str) -> str:
    """Execute Python code for math, calculations, data transformations, or logic."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    code = code.removesuffix("```")
    code = code.strip()

    old_stdout = sys.stdout
    redirected = io.StringIO()
    sys.stdout = redirected

    safe_globals = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
            "float": float, "format": format, "hex": hex, "int": int, "isinstance": isinstance,
            "len": len, "list": list, "map": map, "max": max, "min": min, "oct": oct,
            "ord": ord, "pow": pow, "print": print, "range": range, "reversed": reversed,
            "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "zip": zip,
        },
        "math": math,
    }

    try:
        try:
            expr_val = eval(code, safe_globals)
            out = redirected.getvalue().strip()
            if out:
                return f"{out}\nResult: {expr_val}"
            return str(expr_val)
        except SyntaxError:
            exec(code, safe_globals)
            out = redirected.getvalue().strip()
            return out if out else "Execution finished with no output."
    except Exception as e:
        return f"Python execution error: {e}"
    finally:
        sys.stdout = old_stdout
