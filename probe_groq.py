from dotenv import load_dotenv
import os
load_dotenv()
from groq import Groq
api_key = os.getenv("GROQ_API_KEY")
print("GROQ_KEY present:", bool(api_key))
if not api_key:
    raise SystemExit("GROQ_API_KEY missing in .env")

g = Groq(api_key=api_key)

candidates = [
    "llama3-8b-8192",
    "llama3-8b-4096",
    "llama3-8b-2048",
    "llama3-13b-8192",
    "llama3-70b-8192"
]

for m in candidates:
    try:
        print("Trying model:", m)
        r = g.chat.completions.create(model=m, messages=[{"role":"user","content":"hello"}], max_tokens=8)
        content = getattr(getattr(r.choices[0], 'message', None), 'content', None) or str(r.choices[0])
        print("OK model:", m, "->", content[:200])
        break
    except Exception as e:
        print("Model", m, "failed:", str(e))
