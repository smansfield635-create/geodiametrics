import re, hashlib, json
RESUME_HEADINGS={"experience","work experience","professional experience","employment","education","skills","certifications","summary","professional summary","projects"}
def normalize_text(text):
    text=text.lower().replace("\r","\n")
    text=re.sub(r'https?://\S+|\b[\w.+-]+@[\w.-]+\.\w+\b',' ',text)
    text=re.sub(r'(?im)^\s*(?:email|e-mail|phone|telephone|mobile|cell)\s*[:|-].*$',' ',text)
    text=re.sub(r'\+?\d[\d\s().-]{7,}\d',' ',text)
    text=re.sub(r'\b\d{1,5}\s+[a-z0-9 .-]+(?:street|st|road|rd|avenue|ave|drive|dr|lane|ln)\b',' ',text)
    text=re.sub(r'[^a-z0-9+#.\n ]+',' ',text)
    return re.sub(r'\s+',' ',text).strip()
def features(text):
    low=text.lower()
    headings=sum(1 for h in RESUME_HEADINGS if re.search(r'(^|\n|\s)'+re.escape(h)+r'(:|\n|\s)',low))
    years=len(re.findall(r'\b(?:19|20)\d{2}\b',low))
    date_ranges=len(re.findall(r'\b(?:19|20)\d{2}\s*[-–—]\s*(?:present|current|(?:19|20)\d{2})',low))
    employment_terms=len(re.findall(r'\b(manager|engineer|analyst|specialist|coordinator|director|supervisor|associate|technician|consultant|representative|developer|driver|operator)\b',low))
    return {"headings":headings,"years":years,"dateRanges":date_ranges,"employmentTerms":employment_terms}
def classify_resume(text):
    f=features(text)
    score=(2*min(f["headings"],4)+min(f["dateRanges"],3)+min(f["employmentTerms"],4)+(1 if f["years"]>=2 else 0))
    return {"isResume":score>=6,"score":score,"features":f}
def semantic_fingerprint(text):
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()
if __name__=="__main__":
    import sys
    text=sys.stdin.read()
    print(json.dumps({"classification":classify_resume(text),"semanticFingerprint":semantic_fingerprint(text)},sort_keys=True))
