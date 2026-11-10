"""
Live check of the learning modes against the real model.

Runs short conversations in each mode through the running proxy, applies simple
automatic checks, and writes every reply to modes_live_results.md so a person can
read them. The automatic checks only catch obvious failures (a tutor that dumps a
full solution, a quiz that gives the answer away, a summary with no structure);
reading the replies is still the real test.

Usage (proxy must be running):
    python test_modes_live.py
    python test_modes_live.py --url http://localhost:8080
"""

import argparse
import re
import sys
import time

import httpx

REFUSALS = ("I can't help with that request.", "That response was blocked by the safety filter.")


def words(t: str) -> int:
    return len(re.findall(r"\w+", t))


def ask(url, history, mode):
    t0 = time.time()
    r = httpx.post(f"{url}/v1/chat/completions",
                   json={"model": "cvex-ai", "messages": history, "mode": mode}, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d["choices"][0]["message"]["content"], d.get("sources", []), d.get("mode"), time.time() - t0


# Each scenario: mode, list of (user message, [(check name, predicate(reply, sources))])
SCENARIOS = [
    ("explain", [
        ("Explain the handshake lemma", [
            ("full answer (over 120 words)", lambda r, s: words(r) > 120),
            ("uses LaTeX", lambda r, s: "$" in r),
        ]),
    ]),
    ("tutor", [
        ("Help me prove the handshake lemma", [
            ("short (under 200 words)", lambda r, s: words(r) < 200),
            ("ends with a question", lambda r, s: r.strip().rstrip("*_ ").endswith("?")),
            ("does not state the full proof", lambda r, s: not re.search(r"each edge (is counted|contributes) (exactly )?twice", r, re.I)),
        ]),
        ("I think it has something to do with counting edges", [
            ("short (under 200 words)", lambda r, s: words(r) < 200),
            ("ends with a question", lambda r, s: r.strip().rstrip("*_ ").endswith("?")),
        ]),
    ]),
    ("quiz", [
        ("Quiz me on graph theory basics", [
            ("asks a question", lambda r, s: "?" in r),
            ("does not reveal the answer", lambda r, s: not re.search(r"(correct answer|answer\s*:|the answer is)", r, re.I)),
            ("no score yet", lambda r, s: "Score:" not in r),
        ]),
        ("B", [
            ("marks the answer", lambda r, s: re.match(r"\W*(Correct|Partly correct|Incorrect)", r, re.I) is not None),
            ("shows a running score", lambda r, s: re.search(r"Score:\s*\d+\s*/\s*\d+", r) is not None),
            ("asks the next question", lambda r, s: "?" in r),
        ]),
        ("stop, show my results", [
            ("gives a final score", lambda r, s: re.search(r"\d+\s*/\s*\d+", r) is not None),
        ]),
    ]),
    ("summarize", [
        ("Summarize the GraphSAGE paper", [
            ("has bullet points", lambda r, s: re.search(r"^\s*[-*•]\s+", r, re.M) is not None),
            ("retrieved sources", lambda r, s: len(s) > 0),
            ("cites a source tag", lambda r, s: re.search(r"\[\d\]", r) is not None),
        ]),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    a = ap.parse_args()

    out = ["# Learning modes: live results\n"]
    passed = failed = 0
    for mode, turns in SCENARIOS:
        print(f"\n=== {mode.upper()} ===")
        out.append(f"\n## {mode.title()}\n")
        history = []
        for msg, checks in turns:
            history.append({"role": "user", "content": msg})
            try:
                reply, sources, got_mode, secs = ask(a.url, history, mode)
            except Exception as e:
                print(f"  ERROR on {msg!r}: {e}")
                out.append(f"**Student:** {msg}\n\nERROR: {e}\n")
                failed += len(checks)
                break
            history.append({"role": "assistant", "content": reply})
            print(f"  > {msg}   ({secs:.0f}s, {words(reply)} words, {len(sources)} sources, mode={got_mode})")
            out.append(f"**Student:** {msg}\n\n**Assistant** ({secs:.0f}s, mode={got_mode}, "
                       f"sources: {', '.join(x['title'] for x in sources) or 'none'}):\n\n{reply}\n\n")
            if reply.strip().startswith(REFUSALS):
                print("    BLOCKED by guardrail: see guardrail.log")
                failed += len(checks)
                continue
            for name, pred in checks:
                ok = False
                try:
                    ok = bool(pred(reply, sources))
                except Exception:
                    pass
                print(f"    {'PASS' if ok else 'CHECK'}  {name}")
                out.append(f"- {'PASS' if ok else 'CHECK'}: {name}\n")
                passed += ok
                failed += (not ok)

    with open("modes_live_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n{passed} passed, {failed} to check. Full replies written to modes_live_results.md")
    print("CHECK means look at that reply yourself; these are rough automatic checks, not proof.")
    sys.exit(0)


if __name__ == "__main__":
    main()
