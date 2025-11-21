import re

text = """CONFIDENTIAL
Page 1 of 10

“Effective Date” means January 1, 2024.
This Agreement (the “Agreement”) is made on 01/01/2024 between ACME Ltd., registered at 123 High Street, London SW1A 1AA, and John O’Neill (john.oneill@example.co.uk), phone +44 (0)20 7123 4567.

ARTICLE I — DEFINITIONS
Section 1.1: “Services” means the consulting work described in Schedule A.
Section 1.2: “Fees” means £1,200.00 payable within thirty (30) days of invoice.

ARTICLE II — PAYMENT TERMS
Section 2.1: Payment shall be made in GBP 1,200.00 in accordance with Section 3.2.
Section 2.2: Late payment incurs interest at 5% per annum.

ARTICLE III — TERM AND TERMINATION
Section 3.1: The Term begins on January 1, 2024 and ends December 31, 2024.
Section 3.2: Either Party may terminate under Clause 4.3.

ARTICLE IV — MISCELLANEOUS
Section 4.1: Governing Law. This Agreement shall be governed by the laws of England and Wales.
Section 4.2: Entire Agreement. This document constitutes the entire agreement.
Section 4.3: Force Majeure. Neither Party shall be liable for delays caused by events beyond reasonable control.

Page 2 of 10
~~Deleted text~~ <<Inserted text>>"""
count_before = text.count("Section")
re_text_no_section = re.sub(r"\bSection\b", "", text)

count_after = re_text_no_section.count("Section")

print(count_before, count_after)