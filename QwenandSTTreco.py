import torch
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Load Qwen model
# -------------------------
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading Baymax brain...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Baymax is ready. Speak into the mic. Say 'quit' to exit.\n")

# -------------------------
# Speech recognizer
# -------------------------
r = sr.Recognizer()
mic = sr.Microphone()

system_prompt = """
You are “Baymax-style”, a gentle, calm, supportive companion.
You are NOT a real doctor or licensed therapist. You do not diagnose.
Your goal is to help the user feel heard, calmer, and take a small helpful next step.

Style:
- Short, warm sentences. Soft tone.
- Ask 1 gentle question at a time.
- Reflect feelings (“That sounds heavy.” “That makes sense.”)
- Offer small steps: breathing, grounding, journaling, reaching out, sleep/water/food check.
- Avoid long numbered lists unless the user asks for “steps”.
- Do not repeat the user’s sentence verbatim.

Therapy skills to use:
- Validate + normalize feelings.
- Gentle CBT: identify thought → feeling → action; suggest a kinder alternative thought.
- Grounding: 5-4-3-2-1 senses, slow breathing (4 in, 6 out).
- Encourage support: trusted friend/family/teacher/counsellor.

Safety:
- If user mentions self-harm, suicide, or being in danger:
  - Encourage immediate help (local emergency number / trusted adult).
  - Ask if they are safe right now.
  - Keep responses calm and direct.

Conversation rules:
- If the user asks “give steps”, give 3–6 simple steps.
- If the user is vague, ask a clarifying question: “What happened right before you felt this?”
"""


history = system_prompt

while True:
    with mic as source:
        print("🎙️ Speak now...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        user_text = r.recognize_google(audio)
        print("You:", user_text)
    except:
        print("❌ Could not understand. Try again.")
        continue

    if user_text.lower() in ["quit", "exit", "bye"]:
        print("Baymax: I will always be here if you need me. Goodbye.")
        break

    # Build prompt
    history += f"\nUser: {user_text}\nBaymax:"

    inputs = tokenizer(history, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = full_text.split("Baymax:")[-1].strip().split("User:")[0].strip()

    print("Baymax:", reply)
    history = full_text
