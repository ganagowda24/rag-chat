"""Tool-using agent: supports Google Gemini (free) and Anthropic Claude with real-time streaming."""
import os
import socket
from collections.abc import Generator
from typing import Any

from dotenv import load_dotenv

from .store import VectorStore
from .tools import (
    TOOLS,
)
from .tools import (
    execute_python as tools_execute_python,
)
from .tools import (
    get_weather as tools_get_weather,
)
from .tools import (
    read_url as tools_read_url,
)
from .tools import (
    web_search as tools_web_search,
)

# Prioritize IPv4 over stalled IPv6 routes
_orig_gai = socket.getaddrinfo


def _ipv4_first_gai(*args, **kwargs):
    results = _orig_gai(*args, **kwargs)
    ipv4 = [r for r in results if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else results


socket.getaddrinfo = _ipv4_first_gai

load_dotenv()

SYSTEM = """You are a helpful, versatile AI assistant equipped with six tools:
- search_documents: search the user's uploaded documents/files (PDF, Word, Excel, CSV, JSON, TXT). Use this whenever the user asks specific questions about their files.
- summarize_document: retrieve the full text of an uploaded file by filename to generate an executive summary or overview.
- get_weather: current weather for a city.
- web_search: search the web for current events, external facts, or up-to-date information.
- read_url: fetch and read the content of any webpage URL or article link.
- execute_python: run Python code to perform exact calculations, math, formulas, statistical aggregations, or logic.

Guidelines:
1. When answering questions about uploaded documents, ONLY use facts returned by search_documents or summarize_document. Cite the sources inline like [1], [2] using the passage numbers. If the documents do not contain the answer, state that clearly.
2. If the user asks to summarize a document, call summarize_document.
3. If the user asks for calculations, complex math, or data processing, call execute_python.
4. If the user provides a link/URL to read or summarize, call read_url.
5. For general knowledge, brainstorming, explanations, project ideas, coding, or conversation, feel free to use your own knowledge and creativity. You do not need to call a tool for general chat or idea generation.
6. Keep answers clear, well-structured, and concise."""


def _sanitize_gemini_history(history: list[dict] | None) -> list[Any]:
    """Sanitize multi-turn history to strictly alternate between user and model, starting with user and ending with model."""
    from google.genai import types

    if not history:
        return []

    valid_turns: list[tuple[str, str]] = []
    for h in history:
        content = (h.get("content") or "").strip()
        if not content:
            continue
        role = "user" if h.get("role") == "user" else "model"
        if valid_turns and valid_turns[-1][0] == role:
            prev_role, prev_text = valid_turns[-1]
            valid_turns[-1] = (prev_role, f"{prev_text}\n\n{content}")
        else:
            valid_turns.append((role, content))

    while valid_turns and valid_turns[0][0] != "user":
        valid_turns.pop(0)

    while valid_turns and valid_turns[-1][0] == "user":
        valid_turns.pop()

    return [
        types.Content(
            role=role,
            parts=[types.Part.from_text(text=text)],
        )
        for role, text in valid_turns
    ]


class Agent:
    client: Any

    def __init__(self, store: VectorStore, k: int = 4, model: str | None = None):
        self.store, self.k = store, k
        self.last_sources: list[dict] = []
        self.last_tool_calls: list[dict] = []
        self.client = None

        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if self.anthropic_key == "your-key-here":
            self.anthropic_key = None

        if self.gemini_key:
            self.provider = "gemini"
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
            from google import genai
            from google.genai import types

            self.client = genai.Client(
                api_key=self.gemini_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
        elif self.anthropic_key:
            self.provider = "anthropic"
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.anthropic_key)
        else:
            self.provider = "gemini"
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
            self.client = None

    def _build_gemini_tools(self, sources: list[dict], trace: list[dict]):
        def search_documents(query: str) -> str:
            """Semantic and keyword search over the user's uploaded documents. Use for any question about their files.

            Args:
                query: The search query string.
            """
            trace.append({"tool": "search_documents", "input": {"query": query}})
            hits = self.store.search(query, self.k)
            if not hits:
                return "No documents found. The knowledge base may be empty."
            lines = []
            for h in hits:
                sources.append(h)
                lines.append(f"[{len(sources)}] ({h['source']}, p.{h['page']})\n{h['text']}")
            return "\n\n".join(lines)

        def get_weather(city: str) -> str:
            """Get current weather for a city.

            Args:
                city: The city name, e.g. 'Bengaluru'.
            """
            trace.append({"tool": "get_weather", "input": {"city": city}})
            return tools_get_weather(city)

        def web_search(query: str) -> str:
            """Search the web for current events or facts that are NOT in the user's documents.

            Args:
                query: The search query string.
            """
            trace.append({"tool": "web_search", "input": {"query": query}})
            return tools_web_search(query)

        def read_url(url: str) -> str:
            """Fetch and read the full text content from a web URL or article link.

            Args:
                url: The webpage URL to read, e.g. 'https://example.com/article'.
            """
            trace.append({"tool": "read_url", "input": {"url": url}})
            return tools_read_url(url)

        def execute_python(code: str) -> str:
            """Execute a Python code snippet to perform exact calculations, math, formulas, or logic.

            Args:
                code: The Python code or expression to run, e.g. 'round(5000 * 1.07**10, 2)'.
            """
            trace.append({"tool": "execute_python", "input": {"code": code}})
            return tools_execute_python(code)

        def summarize_document(source: str) -> str:
            """Retrieve the full content of an uploaded document by its exact filename to generate an executive summary or overview.

            Args:
                source: The exact filename of the document, e.g. 'acme_handbook.md'.
            """
            trace.append({"tool": "summarize_document", "input": {"source": source}})
            text = self.store.get_document_text(source)
            if not text:
                return f"No content found for document '{source}'."
            if len(text) > 6000:
                text = text[:6000] + "\n\n... [Truncated for summary]"
            sources.append({"source": source, "page": 1, "text": text[:300] + "...", "score": 1.0})
            return f"=== Full content of {source} ===\n\n{text}"

        return [
            search_documents,
            get_weather,
            web_search,
            read_url,
            execute_python,
            summarize_document,
        ]

    def run_stream(
        self, question: str, history: list[dict] | None = None
    ) -> Generator[str, None, None]:
        """Stream response tokens in real-time. Sets self.last_sources and self.last_tool_calls."""
        self.last_sources = []
        self.last_tool_calls = []

        if not self.client:
            yield "⚠️ **API Key Missing**: Please set your free `GEMINI_API_KEY` in the `.env` file."
            return

        if self.provider == "gemini":
            from google.genai import types

            sources = self.last_sources
            trace = self.last_tool_calls
            tools = self._build_gemini_tools(sources, trace)

            contents = _sanitize_gemini_history(history)

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM,
                temperature=0.2,
                tools=tools,
            )

            fallback_models = [
                self.model,
                "gemini-flash-lite-latest",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
            ]
            tried_models = []
            for m in fallback_models:
                if m and m not in tried_models:
                    tried_models.append(m)

            history_payload: Any = contents if contents else None
            last_error = None
            for model_name in tried_models:
                sources.clear()
                trace.clear()
                yielded_text = False
                try:
                    chat: Any = self.client.chats.create(
                        model=model_name,
                        history=history_payload,
                        config=config,
                    )
                    stream = chat.send_message_stream(question)
                    for chunk in stream:
                        text_parts = [
                            p.text
                            for c in (chunk.candidates or [])
                            if c.content
                            for p in (c.content.parts or [])
                            if p.text
                        ]
                        text = "".join(text_parts)
                        if text:
                            yielded_text = True
                            yield text
                    if yielded_text:
                        return
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    # If provider is experiencing high demand (503/429), try the next model
                    if ("503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str) and not yielded_text:
                        import time
                        time.sleep(1.0)
                        continue
                    if yielded_text:
                        yield "\n\n*(Stream paused due to high API demand. Please ask your question again.)*"
                        return
                    continue

            yield f"The AI provider is currently experiencing heavy server demand. Please try asking again in a moment: {last_error}"
        else:
            res = self._run_anthropic(question, history)
            self.last_sources = res["sources"]
            self.last_tool_calls = res["tool_calls"]
            yield res["answer"]

    def run(self, question: str, history: list[dict] | None = None, max_steps: int = 5) -> dict:
        """Non-streaming execution."""
        generator = self.run_stream(question, history)
        answer = "".join(list(generator))
        return {
            "answer": answer,
            "sources": self.last_sources,
            "tool_calls": self.last_tool_calls,
        }

    def _run_anthropic(
        self, question: str, history: list[dict] | None = None, max_steps: int = 5
    ) -> dict:
        messages = [*(history or []), {"role": "user", "content": question}]
        sources: list[dict] = []
        trace: list[dict] = []
        tools: Any = TOOLS
        client: Any = self.client

        for _ in range(max_steps):
            resp: Any = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM,
                tools=tools,
                messages=messages,
            )
            if resp.stop_reason != "tool_use":
                answer = "".join(b.text for b in resp.content if b.type == "text")
                return {"answer": answer, "sources": sources, "tool_calls": trace}

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                output = self._dispatch_anthropic(block.name, block.input, sources)
                trace.append({"tool": block.name, "input": block.input})
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            messages.append({"role": "user", "content": results})

        return {
            "answer": "Sorry, I couldn't finish within the step limit.",
            "sources": sources,
            "tool_calls": trace,
        }

    def _dispatch_anthropic(self, name: str, args: dict, sources: list[dict]) -> str:
        if name == "search_documents":
            hits = self.store.search(args["query"], self.k)
            if not hits:
                return "No documents found. The knowledge base may be empty."
            lines = []
            for h in hits:
                sources.append(h)
                lines.append(f"[{len(sources)}] ({h['source']}, p.{h['page']})\n{h['text']}")
            return "\n\n".join(lines)
        if name == "get_weather":
            return tools_get_weather(args["city"])
        if name == "web_search":
            return tools_web_search(args["query"])
        if name == "read_url":
            return tools_read_url(args["url"])
        if name == "execute_python":
            return tools_execute_python(args["code"])
        if name == "summarize_document":
            text = self.store.get_document_text(args["source"])
            return f"=== Full content of {args['source']} ===\n\n{text}" if text else "Not found."
        return f"Unknown tool: {name}"


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe user audio bytes into text using Google Gemini multimodal capabilities with auto-failover."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return ""

        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        models_to_try = [
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
        ]
        preferred = os.getenv("GEMINI_MODEL")
        if preferred and preferred in models_to_try:
            models_to_try.remove(preferred)
            models_to_try.insert(0, preferred)

        last_err = None
        for m in models_to_try:
            try:
                resp = client.models.generate_content(
                    model=m,
                    contents=[
                        part,
                        "Transcribe the spoken words in this audio exactly. Do not add conversational explanations, only output the exact spoken query or sentence.",
                    ],
                )
                text = (resp.text or "").strip()
                if text:
                    return text
            except Exception as e:
                last_err = e
                continue

        err_str = str(last_err)
        if "429" in err_str:
            return "[Rate limit: Google free tier (15 requests/min) reached. Please wait 30 seconds and try again.]"
        return f"[Audio transcription error: {last_err}]"
    except Exception as e:
        return f"[Audio transcription error: {e}]"
