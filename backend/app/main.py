import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from parser import decode_gmail_raw_body

app = FastAPI(title="AI Shield Gateway Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_SERVICE_URL = "http://127.0.0.1:8000"

class RawGmailMessage(BaseModel):
    id: str
    subject: str = ""
    sender: str = ""
    rawPayload: dict

class InboxSyncRequest(BaseModel):
    messages: List[RawGmailMessage]

@app.post("/api/inbox/scan")
async def scan_inbox(req: InboxSyncRequest):
    """Mobile app raw Gmail payload bhejegi, ye extract karke ML ko bhejega"""
    sanitized_emails = []
    for msg in req.messages:
        plain_body = decode_gmail_raw_body(msg.rawPayload)
        sanitized_emails.append({
            "id": msg.id,
            "subject": msg.subject,
            "sender": msg.sender,
            "body": plain_body,
            "urls": []
        })

    # Call ML Service in Batch
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(f"{ML_SERVICE_URL}/predict/email/batch", json={"emails": sanitized_emails})
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="ML Engine Error")
            return resp.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="ML Service Unreachable")