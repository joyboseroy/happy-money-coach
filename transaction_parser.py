"""
transaction_parser.py
Parses bank/credit card statements from CSV or PDF.
Returns a clean list of Transaction dicts.
"""

import re
import io
import json
import requests
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Transaction:
    date: str
    description: str
    amount: float
    tx_type: str          # "Debit" or "Credit"
    category: str = ""
    merchant_clean: str = ""
    honda_tag: str = ""   # Ken Honda emotional tag
    arigato_score: int = 0
    happy_money: bool = False
    honda_reason: str = ""
    feedback: Optional[str] = None  # "up" / "down" from user


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
    "%d %b %Y", "%d-%b-%Y", "%d %B %Y",
]


def _parse_date(raw: str) -> str:
    raw = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw  # return as-is if unrecognised


def _infer_columns(df: pd.DataFrame) -> dict:
    """Heuristically map dataframe columns to date / description / amount / type."""
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Date
    for candidate in ["date", "transaction date", "txn date", "value date", "posted date"]:
        if candidate in cols_lower:
            col_map["date"] = cols_lower[candidate]
            break

    # Description
    for candidate in ["description", "narration", "particulars", "details",
                      "transaction details", "merchant", "payee"]:
        if candidate in cols_lower:
            col_map["description"] = cols_lower[candidate]
            break

    # Amount — look for debit/credit split or single amount column
    for candidate in ["amount", "transaction amount", "debit amount", "debit"]:
        if candidate in cols_lower:
            col_map["amount"] = cols_lower[candidate]
            break

    # Credit column (used when debit/credit are separate)
    for candidate in ["credit", "credit amount"]:
        if candidate in cols_lower:
            col_map["credit"] = cols_lower[candidate]
            break

    # Type
    for candidate in ["type", "transaction type", "dr/cr", "dr / cr", "txn type"]:
        if candidate in cols_lower:
            col_map["type"] = cols_lower[candidate]
            break

    return col_map


def parse_csv(file_obj) -> list[Transaction]:
    """
    Accept a file-like object or path. Returns list[Transaction].
    Handles most Indian bank CSV export formats.
    """
    # Try reading with different encodings
    try:
        df = pd.read_csv(file_obj, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        df = pd.read_csv(file_obj, encoding="latin-1")

    # Drop fully empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    col_map = _infer_columns(df)

    if "date" not in col_map or "description" not in col_map:
        raise ValueError(
            "Could not find Date and Description columns. "
            "Please ensure your CSV has columns named Date and Description (or similar)."
        )

    transactions = []

    for _, row in df.iterrows():
        raw_date = str(row.get(col_map["date"], "")).strip()
        if not raw_date or raw_date.lower() in ("nan", "date"):
            continue

        desc = str(row.get(col_map["description"], "")).strip()
        if not desc or desc.lower() in ("nan", ""):
            continue

        # Resolve amount and type
        amount = 0.0
        tx_type = "Debit"

        if "amount" in col_map:
            raw_amt = str(row[col_map["amount"]]).replace(",", "").strip()
            try:
                amount = float(raw_amt)
            except ValueError:
                amount = 0.0
            # Negative = debit in most formats
            if amount < 0:
                tx_type = "Debit"
                amount = abs(amount)
            else:
                tx_type = "Credit"

        # Override with explicit credit column when debit/credit are split
        if "credit" in col_map:
            raw_credit = str(row[col_map["credit"]]).replace(",", "").strip()
            try:
                credit_amt = float(raw_credit)
                if credit_amt > 0:
                    amount = credit_amt
                    tx_type = "Credit"
            except ValueError:
                pass

        # Override with explicit type column
        if "type" in col_map:
            raw_type = str(row[col_map["type"]]).strip().upper()
            if any(k in raw_type for k in ["CR", "CREDIT"]):
                tx_type = "Credit"
            elif any(k in raw_type for k in ["DR", "DEBIT"]):
                tx_type = "Debit"

        transactions.append(Transaction(
            date=_parse_date(raw_date),
            description=desc,
            amount=round(amount, 2),
            tx_type=tx_type,
        ))

    return transactions


# ---------------------------------------------------------------------------
# PDF parsing — LLM-based, works across all Indian bank formats
# ---------------------------------------------------------------------------
#
# Design principle: don't parse the PDF structure at all.
# Extract raw text page by page, then ask Qwen3 to read it as a human would.
# This handles ICICI, HDFC, SBI, Axis, Kotak, Yes Bank, Canara etc.
# without any bank-specific code.
#
# Requires Ollama running with qwen2.5:7b.
# Falls back to a best-effort regex pass if Ollama is unavailable.
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def _parse_indian_amount(s: str) -> float:
    """Parse Indian number format: 2,00,335.86 -> 200335.86"""
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _extract_pages_from_pdf(file_obj) -> list[str]:
    """
    Extract and reassemble ICICI-format multi-line transactions.

    ICICI format (confirmed from actual statement):
      Line N-2: Merchant display name     (e.g. "Joy Bose")
      Line N-1: UPI narration             (e.g. "UPI/Joy Bose/8151832200@yes/...")
      Line N:   DD-MM-YYYY  amount  balance

    Key challenge: ICICI uses DEPOSITS | WITHDRAWALS columns with no label
    on the amount line. We resolve debit/credit by tracking balance direction:
      balance went UP  → Credit (deposit)
      balance went DOWN → Debit (withdrawal)
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber:  pip install pdfplumber")

    date_pat = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
    amount_pat = re.compile(r"([\d,]+\.\d{2})")

    # Lines to skip unconditionally
    skip_pat = re.compile(
        r"^(DATE\s|MODE\s|PARTICULARS|Statement of|ACCOUNT|TOTAL\s|Page\s*\d|Sincerely|Team ICICI|"
        r"This is a system|iMobile|Personal Banking|Internet Banking|Card block|"
        r"CropBox|Dial your|Never share|Visit www|Please call|Available on|Scan to|"
        r"Offers are|SGST\d|CGST\d)",
        re.IGNORECASE
    )
    # Hash/reference lines — long alphanumeric tokens with no spaces
    hash_pat = re.compile(r"^[A-Z0-9/]{25,}$")

    pages = []
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            if not raw.strip():
                continue

            raw_lines = [l.strip() for l in raw.splitlines() if l.strip()]

            # --- Pass 1: group description lines with their date+amount line ---
            groups = []  # each group: {"desc": [...], "date_line": str}
            desc_buffer = []

            for line in raw_lines:
                if skip_pat.match(line) or hash_pat.match(line):
                    continue

                has_date = bool(date_pat.search(line))
                amounts = amount_pat.findall(line)

                if has_date and len(amounts) >= 1:
                    # This is the anchor line for a transaction
                    groups.append({
                        "desc": list(desc_buffer),
                        "date_line": line
                    })
                    desc_buffer = []
                else:
                    # Description / narration line — skip hash/reference strings
                    if len(line) < 100 and line:
                        # Skip lines that are pure reference hashes (letters+digits+slash, no spaces)
                        if not re.match(r'^[A-Za-z0-9/]{20,}$', line):
                            desc_buffer.append(line)

            # --- Pass 2: resolve debit/credit by balance direction ---
            result_lines = []
            prev_balance = None

            for g in groups:
                date_line = g["date_line"]
                desc_parts = g["desc"]

                date_m = date_pat.search(date_line)
                if not date_m:
                    continue
                date_str = date_m.group(1)

                # Convert DD-MM-YYYY to YYYY-MM-DD
                parts = date_str.split("-")
                if len(parts) == 3:
                    iso_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                else:
                    iso_date = date_str

                amounts = [_parse_indian_amount(a) for a in amount_pat.findall(date_line)]

                # Skip B/F (opening balance) line
                desc_joined = " ".join(desc_parts).lower()
                if "b/f" in date_line.lower() or (not desc_parts and len(amounts) == 1):
                    prev_balance = amounts[-1] if amounts else prev_balance
                    continue

                if not amounts:
                    continue

                # Last amount is always the running balance
                balance = amounts[-1]

                if len(amounts) == 1:
                    # Only balance on line — amount is embedded in description
                    # Try to extract from description
                    desc_amounts = []
                    for dp in desc_parts:
                        desc_amounts += [_parse_indian_amount(a) for a in amount_pat.findall(dp)]
                    if not desc_amounts:
                        continue
                    tx_amount = desc_amounts[-1]
                    # Remove the amount from description
                    desc_parts = [re.sub(r"[\d,]+\.\d{2}", "", dp).strip() for dp in desc_parts]
                elif len(amounts) == 2:
                    # One transaction amount + balance
                    tx_amount = amounts[0]
                else:
                    # Two transaction amounts (deposits + withdrawals columns both present)
                    # Use non-zero one, or balance direction to disambiguate
                    if amounts[0] > 0 and amounts[1] == balance:
                        tx_amount = amounts[0]
                    elif amounts[1] > 0 and amounts[2] == balance if len(amounts) > 2 else True:
                        tx_amount = amounts[0]
                    else:
                        tx_amount = amounts[0]

                # Determine debit/credit from balance direction
                if prev_balance is not None:
                    tx_type = "Credit" if balance > prev_balance else "Debit"
                else:
                    # Fallback: look for credit keywords in description
                    full_desc = " ".join(desc_parts).lower() + " " + date_line.lower()
                    credit_keywords = ["salary", "neft cr", "imps cr", "dividend", "interest",
                                      "refund", "cashback", "deposit", "ach cr", "cms",
                                      "reversal", "paid via", "received"]
                    tx_type = "Credit" if any(k in full_desc for k in credit_keywords) else "Debit"

                prev_balance = balance

                # Build clean description
                # Filter out hash-like fragments from desc_parts
                clean_descs = [dp for dp in desc_parts
                               if dp and not re.match(r'^[A-Za-z0-9/]{20,}$', dp) and len(dp) > 2]

                if clean_descs:
                    description = " | ".join(clean_descs)
                else:
                    # Inline transaction: extract narration from date_line itself
                    # Strip the date, amounts — what's left is the description
                    inline = date_line
                    inline = date_pat.sub("", inline).strip()
                    inline = amount_pat.sub("", inline).strip(" |,")
                    description = inline if inline else "Unknown"

                # Format for LLM: one clean line per transaction
                result_lines.append(
                    f"{iso_date} | {tx_type} | ₹{tx_amount:,.2f} | {description}"
                )

            if result_lines:
                pages.append("\n".join(result_lines))

    return pages


def _parse_preprocessed_lines(page_text: str) -> list[dict]:
    """
    Parse the preprocessed lines directly into transaction dicts.
    No LLM needed. Each line is already:
      YYYY-MM-DD | Credit/Debit | ₹amount | description

    This is the default path. Fast, reliable, works offline.
    The LLM is only called on top of this for description prettifying.
    """
    results = []
    for line in page_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        try:
            date = parts[0].strip()
            tx_type = parts[1].strip()
            # Amount: strip ₹ and commas
            amount_str = parts[2].replace("₹", "").replace(",", "").strip()
            amount = float(amount_str)
            description_raw = " | ".join(parts[3:]).strip()
            description = _clean_description(description_raw)

            if tx_type not in ("Credit", "Debit"):
                continue
            if amount <= 0:
                continue

            results.append({
                "date": date,
                "description": _clean_description(description),
                "amount": amount,
                "type": tx_type,
            })
        except (ValueError, IndexError):
            continue
    return results


def _clean_description(raw: str) -> str:
    """
    Rule-based description cleaner. Handles common Indian bank narration formats.
    No LLM required.
    """
    desc = raw.strip()

    # UPI: extract merchant name from UPI/MerchantName/vpa@bank/...
    upi_m = re.match(r"UPI[/\s]+([^/]+)", desc, re.IGNORECASE)
    if upi_m:
        merchant = upi_m.group(1).strip()
        # Map known UPI merchants
        merchant_lower = merchant.lower()
        if any(k in merchant_lower for k in ["zerodha", "iccl zerod", "iccl"]):
            return "Zerodha - Shares"
        if any(k in merchant_lower for k in ["swiggy"]):
            return "Swiggy"
        if any(k in merchant_lower for k in ["zomato"]):
            return "Zomato"
        if any(k in merchant_lower for k in ["amazon"]):
            return "Amazon"
        if any(k in merchant_lower for k in ["flipkart"]):
            return "Flipkart"
        if any(k in merchant_lower for k in ["irctc"]):
            return "IRCTC"
        if any(k in merchant_lower for k in ["cred"]):
            return "CRED"
        if any(k in merchant_lower for k in ["nmdc", "relay", "indigo", "air india"]):
            return merchant[:40].title()
        return f"UPI - {merchant[:35].title()}"

    # NEFT/IMPS/RTGS transfers
    neft_m = re.match(r"NEFT[-/\s]+[A-Z0-9]+[-/\s]+(.+)", desc, re.IGNORECASE)
    if neft_m:
        return neft_m.group(1).strip()[:45].title()

    # ACH/ECS (dividends, SIPs, EMIs)
    ach_m = re.match(r"ACH[/\s]+([^/]+)", desc, re.IGNORECASE)
    if ach_m:
        return ach_m.group(1).strip()[:45].title()

    # CMS transactions (dividends)
    cms_m = re.match(r"CMS[/\s]+\w+[/\s]+(.+?)(?:\s*-\s*Dividend)?$", desc, re.IGNORECASE)
    if cms_m:
        return f"{cms_m.group(1).strip()[:35].title()} (Dividend)"

    # NET BANKING
    if re.match(r"NET BANKING", desc, re.IGNORECASE):
        return "Net Banking Transfer"

    # ICICI cash deposit
    if re.match(r"ICICI CRM", desc, re.IGNORECASE):
        return "Cash Deposit"

    # Foreign currency
    if re.match(r"NRS/", desc, re.IGNORECASE):
        return "Foreign Currency Purchase"

    # BIL/NEFT self transfers
    if re.match(r"BIL/NEFT", desc, re.IGNORECASE):
        return "Self Transfer"

    # Fallback: title-case and trim
    cleaned = re.sub(r"[/|]{2,}", " ", desc)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:50].title()


def _llm_extract_transactions(page_text: str, page_num: int = 0) -> list[dict]:
    """
    LLM-based description cleanup. Called only when use_llm=True AND
    the preprocessed parse succeeded. Improves description quality but
    is not required for correct extraction.
    """
    prompt = f"""You are parsing pre-processed Indian bank statement transactions.

Each line is in format: DATE | TYPE | AMOUNT | DESCRIPTION

Your only job: clean up the DESCRIPTION field.
- "UPI/ICCL ZEROD/zerodha.iccl6..." → "Zerodha - Shares"
- "UPI/Joy Bose/8151832200@yes..." → "UPI Transfer - Joy Bose"
- "NEFT-AXISP00796105962-EMBASSY OFFICE PARKS" → "Embassy Office Parks"
- "ACH/MOLD TEK PACKAGING" → "Mold-Tek Packaging (Dividend)"
- "NET BANKING VIN/RAZ MMTCPAM/..." → "Net Banking Transfer"
- Keep description under 45 characters
- date, amount, type: copy exactly as-is

Return ONLY a JSON array. No explanation, no markdown. If empty input, return [].

Transactions:
{page_text}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 2000,
                    "num_ctx": 4096,
                }
            },
            timeout=300   # 5 minutes — generous for slow CPU machines
        )
        raw = response.json().get("response", "").strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[pdf_parser] LLM timeout/error on page {page_num + 1}: {e}")
        print(f"[pdf_parser] Falling back to rule-based description parsing")
    return []


def _regex_fallback(lines: list[str]) -> list[Transaction]:
    """
    Best-effort regex extraction when Ollama is unavailable.
    Handles the most common Indian bank date/amount patterns.
    """
    transactions = []
    date_pat = re.compile(
        r"(\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}"
        r"|\d{2}\s+\w{3}\s+\d{4}|\d{2}-\w{3}-\d{4})"
    )
    amount_pat = re.compile(r"[\d,]+\.\d{2}")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        date_m = date_pat.search(line)
        amounts = amount_pat.findall(line)
        if not date_m or not amounts:
            continue
        desc_start = date_m.end()
        desc_end = line.find(amounts[0])
        desc = line[desc_start:desc_end].strip() if desc_end > desc_start else line[desc_start:].strip()
        desc = desc.strip("| \t")
        if not desc:
            continue
        amount_val = float(amounts[0].replace(",", ""))
        # Heuristic: last amount on line is usually the running balance; use second-to-last if present
        if len(amounts) >= 2:
            amount_val = float(amounts[-2].replace(",", ""))
        lower = line.lower()
        tx_type = "Credit" if any(k in lower for k in ["cr", "credit", "deposit", "salary"]) else "Debit"
        transactions.append(Transaction(
            date=_parse_date(date_m.group()),
            description=desc,
            amount=amount_val,
            tx_type=tx_type,
        ))
    return transactions


def parse_pdf(file_obj, use_llm: bool = True) -> list[Transaction]:
    """
    Extract transactions from any Indian bank statement PDF.

    Architecture:
      1. pdfplumber extracts raw text
      2. Preprocessor reassembles multi-line transactions and resolves
         debit/credit by tracking the running balance — no LLM needed
      3. _parse_preprocessed_lines() converts directly to Transaction objects
      4. If use_llm=True and Ollama is fast enough, LLM cleans up descriptions
         (this is optional — extraction is complete without it)
    """
    pages = _extract_pages_from_pdf(file_obj)

    if not pages:
        print("[pdf_parser] pdfplumber extracted no text — PDF may be image-based")
        print("[pdf_parser] Try downloading as Excel/CSV from your bank's website instead")
        return []

    # Always parse directly first — this is reliable and instant
    all_transactions = []
    for page_num, page_text in enumerate(pages):
        if not page_text.strip():
            continue

        raw_txs = _parse_preprocessed_lines(page_text)

        # Optionally enhance descriptions with LLM
        if use_llm and raw_txs:
            ollama_available = False
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=2)
                ollama_available = r.status_code == 200
            except Exception:
                pass

            if ollama_available:
                llm_txs = _llm_extract_transactions(page_text, page_num)
                if llm_txs and len(llm_txs) >= len(raw_txs) * 0.8:
                    # LLM gave reasonable output — use its cleaned descriptions
                    # but keep the rule-based amounts/types as ground truth
                    for i, tx_dict in enumerate(raw_txs):
                        if i < len(llm_txs):
                            llm_desc = llm_txs[i].get("description", "")
                            if llm_desc and len(llm_desc) < 60:
                                tx_dict["description"] = llm_desc
                # If LLM timed out or gave fewer results, raw_txs already has good data

        for item in raw_txs:
            try:
                tx = Transaction(
                    date=item["date"],
                    description=item["description"],
                    amount=abs(float(item["amount"])),
                    tx_type=item["type"],
                )
                if tx.description and tx.amount > 0:
                    all_transactions.append(tx)
            except (ValueError, TypeError, KeyError):
                continue

    # Deduplicate
    seen = set()
    unique = []
    for tx in all_transactions:
        key = (tx.date, tx.description[:40], tx.amount)
        if key not in seen:
            seen.add(key)
            unique.append(tx)

    print(f"[pdf_parser] Extracted {len(unique)} unique transactions from {len(pages)} pages")
    return unique


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_statement(file_obj, filename: str, use_llm: bool = True) -> list[Transaction]:
    """
    Auto-detect format from filename extension and parse.
    use_llm=True enables Qwen3-based PDF extraction (recommended).
    use_llm=False falls back to regex (no Ollama required).
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return parse_pdf(file_obj, use_llm=use_llm)
    elif ext in ("csv", "txt"):
        return parse_csv(file_obj)
    elif ext in ("xls", "xlsx"):
        df = pd.read_excel(file_obj)
        return parse_csv(io.StringIO(df.to_csv(index=False)))
    else:
        return parse_csv(file_obj)
