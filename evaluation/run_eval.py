"""Evaluate the agent on a list of test questions.

Metrics per question:
  tool_ok     - did the agent pick the expected tool?
  retrieval   - did retrieved chunks include the expected source file?
  keywords    - fraction of expected keywords present in the answer
  faithful    - LLM-judge score 1-5: is the answer supported by the retrieved
                passages (or an honest "I don't know")? Penalises hallucination.

Usage:  python -m evaluation.run_eval [--no-judge] [--docs data/sample_docs]
"""
import argparse
import json
import os
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv

from rag.agent import Agent
from rag.ingest import load_path
from rag.store import VectorStore

load_dotenv()

JUDGE_PROMPT = """You are grading a RAG assistant's answer.

Question: {q}

Retrieved passages:
{ctx}

Tools used: {tools}

Answer: {a}

Score FAITHFULNESS from 1 to 5:
5 = every claim is supported by the passages (or the tool output), or the answer honestly says the information isn't available
3 = partly supported, some unsupported detail
1 = mostly invented / contradicts the passages
If a non-document tool (weather, web) was used, judge whether the answer is plausible and not self-contradictory.
Reply with ONLY JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def judge(judge_client, provider: str, q, out) -> dict:
    ctx = "\n---\n".join(s["text"] for s in out["sources"]) or "(none)"
    tools = [t["tool"] for t in out["tool_calls"]] or "none"
    prompt = JUDGE_PROMPT.format(q=q, ctx=ctx, tools=tools, a=out["answer"])

    if provider == "gemini":
        r = judge_client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            contents=prompt,
        )
        text = r.text or ""
    else:
        r = judge_client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"score": None, "reason": "judge parse error"}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"score": None, "reason": "judge parse error"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="evaluation/test_questions.json")
    ap.add_argument("--docs", default="data/sample_docs")
    ap.add_argument("--db", default="chroma_eval_db")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    store = VectorStore(path=args.db, collection="eval")
    for p in Path(args.docs).glob("*"):
        if p.suffix.lower() in {".pdf", ".txt", ".md"}:
            store.add_document(p.name, load_path(p))

    agent = Agent(store)
    provider = agent.provider
    judge_client = agent.client

    rows = []
    for case in json.loads(Path(args.questions).read_text()):
        out = agent.run(case["question"])
        used = [t["tool"] for t in out["tool_calls"]]
        row = {
            "question": case["question"],
            "answer": out["answer"],
            "tools": used,
            "tool_ok": case["expected_tool"] in used,
        }
        if case.get("expected_source"):
            row["retrieval"] = any(s["source"] == case["expected_source"] for s in out["sources"])
        kws = case.get("expected_keywords", [])
        if kws:
            row["keywords"] = sum(k.lower() in out["answer"].lower() for k in kws) / len(kws)
        if not args.no_judge and judge_client:
            j = judge(judge_client, provider, case["question"], out)
            row["faithful"], row["judge_reason"] = j["score"], j["reason"]
        rows.append(row)
        flag = "✅" if row["tool_ok"] else "❌"
        print(f"{flag} {case['question'][:60]:60} tools={used} "
              f"kw={row.get('keywords', '-')} faith={row.get('faithful', '-')}")

    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(map(float, vals)), 3) if vals else None

    summary = {
        "n": len(rows),
        "tool_accuracy": avg("tool_ok"),
        "retrieval_hit_rate": avg("retrieval"),
        "keyword_score": avg("keywords"),
        "faithfulness_avg_of_5": avg("faithful"),
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:24} {v}")
    Path("evaluation/results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print("\nSaved evaluation/results.json")


if __name__ == "__main__":
    main()
