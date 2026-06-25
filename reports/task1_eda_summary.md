# Task 1 EDA Summary

## Key Findings

The full CFPB complaint dataset contains approximately 9.6 million records across all financial products. Only about **31%** of complaints include a consumer narrative; the remaining **69%** contain metadata only (product, issue, company) with no free-text description. This makes narrative filtering essential before any NLP or RAG pipeline, since models cannot learn from empty text fields.

After restricting to the four CrediTrust product categories — Credit Card, Personal Loan, Savings Account, and Money Transfer — and removing empty or very short narratives (≤ 3 words after cleaning), the working dataset contains roughly **456,000** complaints. Credit cards dominate the filtered set (~41%), followed by savings/checking accounts (~31%), money transfers (~22%), and personal loans (~6%). Narrative length varies widely: most cleaned complaints fall between 50–300 words, with a long tail of detailed multi-paragraph submissions.

## Preprocessing Decisions

Text cleaning lowercases all narratives, strips common complaint boilerplate (e.g., "I am writing to file a complaint"), removes redacted `XXXX` placeholders and special characters, and normalizes whitespace. Product mapping consolidates CFPB sub-product labels into four standardized categories aligned with CrediTrust's business lines. The cleaned output is saved to `data/filtered_complaints.csv` (Task 1 deliverable) and mirrored to `data/processed/complaints_clean.csv` for downstream chunking and embedding tasks.

## Implications for RAG

The heavy class imbalance (credit cards and savings accounts vs. personal loans) means stratified sampling is required for embedding experiments. Long narratives benefit from chunking rather than single-vector embedding. The high share of narrative-free records in the raw data confirms that retrieval quality depends entirely on the filtered subset, making Task 1 preprocessing a critical foundation for the complaint chatbot.
