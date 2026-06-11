"""
debug_pdf.py
Run this on your bank statement PDF to diagnose extraction issues.

Usage:
    python debug_pdf.py your_statement.pdf

It will tell you exactly what pdfplumber sees, what the LLM extracts,
and where the problem is.
"""

import sys
import json
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_pdf.py your_statement.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print("=" * 60)
    print("STEP 1: What does pdfplumber see?")
    print("=" * 60)

    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
        sys.exit(1)

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages in PDF: {len(pdf.pages)}\n")

        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_text.append(text)

            print(f"--- PAGE {i+1} ---")
            print(f"Characters extracted: {len(text)}")
            print(f"Lines extracted: {len(text.splitlines())}")

            if not text.strip():
                print("*** WARNING: pdfplumber got NO TEXT from this page ***")
                print("This usually means the PDF is image-based (scanned).")
                print("You need OCR. See STEP 4 below.")
            else:
                # Show first 500 chars
                print("First 500 characters:")
                print(repr(text[:500]))
            print()

    # Check if all pages are empty
    total_text = "".join(pages_text)
    if not total_text.strip():
        print("\n" + "=" * 60)
        print("DIAGNOSIS: PDF IS IMAGE-BASED (SCANNED)")
        print("=" * 60)
        print("""
pdfplumber extracted zero text from all pages.
Your bank statement is a scanned image, not a text PDF.

SOLUTION OPTIONS:

Option A (easiest): Download as a different format
  Most banks offer both:
  - PDF (sometimes scanned image) 
  - Excel/CSV download
  Use the Excel/CSV — it works perfectly with this app.
  ICICI: Login -> Accounts -> Download Statement -> Excel

Option B: OCR the PDF first
  Install: pip install pytesseract pdf2image pillow
  Also install tesseract: sudo apt-get install tesseract-ocr
  Then run: python debug_pdf.py your_statement.pdf --ocr

Option C: Copy-paste from PDF
  Open PDF in browser, select all text, paste into a .txt file,
  upload the .txt file instead.
""")
        return

    print("=" * 60)
    print("STEP 2: Do the lines look like transactions?")
    print("=" * 60)

    # Show all lines that contain a date pattern
    date_pat = re.compile(
        r"\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}"
        r"|\d{2}\s+\w{3}\s+\d{4}|\d{2}-\w{3}-\d{4}"
    )
    amount_pat = re.compile(r"[\d,]+\.\d{2}")

    tx_lines = []
    for page_text in pages_text:
        for line in page_text.splitlines():
            if date_pat.search(line) and amount_pat.search(line):
                tx_lines.append(line)

    print(f"Lines that look like transactions (have date + amount): {len(tx_lines)}")
    print("\nFirst 20 such lines:")
    for line in tx_lines[:20]:
        print(f"  {line}")

    if len(tx_lines) == 0:
        print("\n*** WARNING: No lines match the date+amount pattern ***")
        print("Possible reasons:")
        print("  1. Date format is unusual (e.g. '01 Apr 2026' — add to date_pat)")
        print("  2. Amount format is unusual (e.g. '1,234' without decimals)")
        print("  3. Date and amount are on separate lines (multi-line transactions)")
        print("\nShowing raw text of first page for manual inspection:")
        print(pages_text[0][:2000] if pages_text else "No text")

    print()
    print("=" * 60)
    print("STEP 3: Check if Ollama is running")
    print("=" * 60)

    import requests as req
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama is running. Available models: {models}")
        ollama_ok = True
    except Exception as e:
        print(f"Ollama NOT reachable: {e}")
        print("Start it with: ollama serve")
        ollama_ok = False

    if ollama_ok and pages_text:
        print()
        print("=" * 60)
        print("STEP 3b: Reassembled page text (what LLM actually sees)")
        print("=" * 60)

        # Import and test the reassembly logic
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from transaction_parser import _extract_pages_from_pdf
            with open(pdf_path, "rb") as f:
                reassembled_pages = _extract_pages_from_pdf(f)
            print(f"Reassembled into {len(reassembled_pages)} pages")
            for i, page in enumerate(reassembled_pages[:3]):
                lines = [l for l in page.splitlines() if l.strip()]
                tx_lines_r = [l for l in lines if date_pat.search(l) or amount_pat.search(l)]
                print(f"\nPage {i+1}: {len(lines)} lines, {len(tx_lines_r)} look like transactions")
                print("First 15 lines after reassembly:")
                for l in lines[:15]:
                    print(f"  {l}")
        except Exception as e:
            print(f"Could not test reassembly: {e}")

        print()
        print("=" * 60)
        print("STEP 4: What does the LLM extract from first transaction page?")
        print("=" * 60)

        # Find first page with actual transactions (skip header/ad pages)
        test_page_text = None
        test_page_num = 0
        try:
            from transaction_parser import _extract_pages_from_pdf
            with open(pdf_path, "rb") as f:
                reassembled = _extract_pages_from_pdf(f)
            for i, p in enumerate(reassembled):
                if date_pat.search(p) and amount_pat.search(p):
                    test_page_text = p
                    test_page_num = i
                    break
        except Exception:
            pass

        if not test_page_text:
            test_page_text = next((p for p in pages_text if date_pat.search(p) and amount_pat.search(p)), pages_text[0])

        print(f"Testing LLM on page {test_page_num + 1} ({len(test_page_text)} chars)...")
        model = models[0] if models else "qwen2.5:7b"
        print(f"Using model: {model}")

        prompt = f"""You are an expert at reading Indian bank statements, especially ICICI Bank.

ICICI format: DATE | MODE | PARTICULARS | DEPOSITS | WITHDRAWALS | BALANCE
- DEPOSITS = Credit (money in), WITHDRAWALS = Debit (money out)
- Lines joined with " | " belong to one transaction
- Skip B/F, closing balance, SGST/CGST tax lines, header rows

Return ONLY a JSON array:
[{{"date": "YYYY-MM-DD", "description": "merchant name", "amount": 0.00, "type": "Credit|Debit"}}]

Page:
{test_page_text}"""

        try:
            resp = req.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 4000, "num_ctx": 8192}
                },
                timeout=120
            )
            raw = resp.json().get("response", "").strip()
            raw_clean = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\[.*\]", raw_clean, re.DOTALL)

            if match:
                txs = json.loads(match.group())
                print(f"\nLLM extracted {len(txs)} transactions from page 1:")
                for tx in txs:
                    print(f"  {tx.get('date','?')} | {tx.get('type','?'):6} | "
                          f"₹{tx.get('amount',0):>10,.2f} | {str(tx.get('description',''))[:50]}")
            else:
                print("\nLLM returned no valid JSON. Raw response:")
                print(raw[:1000])

        except Exception as e:
            print(f"LLM call failed: {e}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pages in PDF:              {len(pages_text)}")
    print(f"Total text extracted:      {len(total_text)} characters")
    print(f"Transaction-looking lines: {len(tx_lines)}")
    if ollama_ok:
        print("Ollama:                    Running")
    else:
        print("Ollama:                    NOT running — using regex fallback only")
    print()
    print("If transaction count looks wrong, share the output above")
    print("and we can diagnose further.")


if __name__ == "__main__":
    main()
