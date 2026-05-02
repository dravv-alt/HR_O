import os
from dotenv import load_dotenv
load_dotenv()

def test_anthropic():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key or not key.startswith("sk-ant"):
        return False, "ANTHROPIC_API_KEY missing or malformed"
    try:
        import anthropic
        c = anthropic.Anthropic(api_key=key)
        r = c.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply: OK"}]
        )
        return True, r.content[0].text.strip()
    except Exception as e:
        return False, str(e)

def test_groq():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return False, "GROQ_API_KEY missing"
    try:
        from openai import OpenAI
        c = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        r = c.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=10,
            messages=[{"role": "user", "content": "Reply: OK"}]
        )
        return True, r.choices[0].message.content.strip()
    except Exception as e:
        return False, str(e)

print("=== API KEY VALIDATION ===")
ok_a, msg_a = test_anthropic()
ok_g, msg_g = test_groq()

print(f"Anthropic: {'[WORKING]' if ok_a else '[BROKEN]'} - {msg_a}")
print(f"Groq:      {'[WORKING]' if ok_g else '[BROKEN]'} - {msg_g}")

if not ok_a and not ok_g:
    print()
    print("[STOP] EXECUTION HALTED")
    print("No working LLM provider found. Please update .env with valid keys:")
    print("  ANTHROPIC_API_KEY=your_anthropic_key_here")
    print("  GROQ_API_KEY=your_groq_key_here")
    print()
    print("After updating keys, re-run: python code/validate_keys.py")
    print("Then restart the finisher prompt execution.")
    exit(1)

active = "anthropic" if ok_a else "groq"
print(f"\\n[OK] Active provider: {active}")
print("Proceeding with execution.")
