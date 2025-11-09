from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
client = OpenAI()

app = FastAPI()

SUMMARY_FILE = "conversation_summary.json"

# ---------- Helper: load and save summary ----------
def load_summary():
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {"summary": ""}
    return data

def save_summary(summary_text: str):
    with open(SUMMARY_FILE, "w") as f:
        json.dump({"summary": summary_text}, f)

# ---------- Chatbot core logic ----------
def chatbot(dialogue: str, character_description: str, scenario: str):
    data = load_summary()
    summary = data.get("summary", "")

    summary_context = f"\nConversation summary so far: {summary}" if summary else ""

    prompt = f"""
        You are an NPC in a fantasy RPG.
        Character: {character_description}
        Scenario: {scenario}.
        {summary_context}

        The player says: "{dialogue}"

        Reply naturally in character with one or two sentences.
        After replying, return only JSON in this format (and nothing else):
        {{"reply": "<npc reply>", "summary": "<updated summary of the conversation so far>"}}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.8,
    )

    content = response.choices[0].message.content.strip()

    # Attempt to extract valid JSON from model output
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    json_str = content[json_start:json_end]

    try:
        parsed = json.loads(json_str)
        reply = parsed.get("reply", "")
        summary_new = parsed.get("summary", summary)
        save_summary(summary_new)
    except Exception:
        # fallback in case model output isn't clean JSON
        reply = content
        summary_new = summary
        save_summary(summary_new)

    return reply


# ---------- FastAPI Endpoint ----------
class DialogueInput(BaseModel):
    dialogue: str
    character_description: str
    scenario: str

@app.post("/generate_reply")
async def generate_reply(input_data: DialogueInput):
    print("Received dialogue:", input_data.dialogue)
    reply = chatbot(
        dialogue=input_data.dialogue,
        character_description=input_data.character_description,
        scenario=input_data.scenario
    )
    return {"npc_reply": reply}
