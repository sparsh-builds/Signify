import base64
from bs4 import BeautifulSoup

def extract_plain_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    # Script aur style tags hatao
    for s in soup(["script", "style"]):
        s.extract()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())

def decode_gmail_raw_body(payload: dict) -> str:
    """Gmail API ke nested parts aur base64 ko plain text me badalta hai"""
    body_data = ""
    if "parts" in payload:
        for part in payload["parts"]:
            mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data", "")
            if data and mime in ["text/plain", "text/html"]:
                decoded = base64.urlsafe_b64decode(data.encode("ASCII")).decode("utf-8", errors="ignore")
                if mime == "text/html":
                    decoded = extract_plain_text_from_html(decoded)
                body_data += " " + decoded
    elif "body" in payload and "data" in payload["body"]:
        data = payload["body"]["data"]
        decoded = base64.urlsafe_b64decode(data.encode("ASCII")).decode("utf-8", errors="ignore")
        body_data = extract_plain_text_from_html(decoded)
    return body_data.strip() 