#!/usr/bin/env python3
"""
Generate email triage dataset with multi-step action sequences.

Output:
    docs/train_data.jsonl   (~80 %)
    docs/test_data.jsonl    (~20 %)

Schema per line (JSON):
{
  "id": <int>,
  "email_text": <str>,
  "category": <str>,   # work | spam | personal | promotion | urgent
  "steps": [
    {"action": <str>, "value": <str>},   # 'value' only for reply / route_to_department
    ...
  ]
}

Actions:
  high-priority "crisis"   –  first step for critical emergencies
  mark_important           –  flag as important (no value)
  mark_spam                –  flag as spam / scam (no value)
  ignore                   –  low-priority, take no action (no value)
  route_to_department      –  value = department name
  reply                    –  value = suggested reply text

Departments:
  HR | Management | Tech Support | Billing & Finance | Legal |
  Business Team | Customer Support | Sales | Operations | Security
"""

import json
import random
from pathlib import Path

random.seed(42)

CRISIS = 'high-priority "crisis"'

DOCS = Path(__file__).parent.parent / "docs"
DOCS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper to build step lists concisely
# ---------------------------------------------------------------------------
def crisis_steps(dept: str, add_reply: str = "") -> list:
    steps = [
        {"action": CRISIS},
        {"action": "mark_important"},
        {"action": "route_to_department", "value": dept},
    ]
    if add_reply:
        steps.append({"action": "reply", "value": add_reply})
    return steps


def route_steps(dept: str, important: bool = True, reply: str = "") -> list:
    steps = []
    if important:
        steps.append({"action": "mark_important"})
    steps.append({"action": "route_to_department", "value": dept})
    if reply:
        steps.append({"action": "reply", "value": reply})
    return steps


def reply_steps(text: str, important: bool = False) -> list:
    if important:
        return [{"action": "mark_important"}, {"action": "reply", "value": text}]
    return [{"action": "reply", "value": text}]


ACK = "We have received your message and will address it as a priority."
ROUTE_ACK = "Your request has been forwarded to the appropriate department."
REPLY_ACK = "Thank you for reaching out. I will get back to you shortly."
MEET_ACK = "Happy to connect! Let me check my calendar and share some availability."
STAT_ACK = "Thanks for following up. I will look into this and update you by end of day."
FWD_ACK = "Noted. I have forwarded your concern to the relevant team."

# ---------------------------------------------------------------------------
# All examples: (email_text, category, steps)
# ---------------------------------------------------------------------------
ALL: list[tuple[str, str, list]] = [

    # ==================================================================
    # CRISIS EMAILS (urgent) — 70 examples
    # Steps ALWAYS start with high-priority "crisis"
    # ==================================================================

    # ---- Security incidents (15) ----
    (
        "CRITICAL: Production database breached. 2 million customer records exposed. Exfiltration detected.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Ransomware attack in progress. Files encrypting across all company servers right now.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Unauthorized root access detected on all cloud servers. Attacker has admin credentials.",
        "urgent", crisis_steps("Security")
    ),
    (
        "SQL injection attack on our main API. Live data exfiltration confirmed by logs.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "DDoS attack targeting payment infrastructure. 95% packet loss. Revenue halted.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Corporate credentials compromised via targeted phishing attack on exec team.",
        "urgent", crisis_steps("Security")
    ),
    (
        "Supply chain attack: our build pipeline is compromised. All recent releases are suspect.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Customer PII posted on dark web. 100k credit card numbers visible in data dump.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Insider threat: departing employee mass-downloaded all client data before exit.",
        "urgent", crisis_steps("Security")
    ),
    (
        "API keys accidentally committed to public GitHub repo 3 hours ago. Keys not yet rotated.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Crypto mining malware deployed on 60+ company machines, degrading all services.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Zero-day exploit used against our customer portal. Patch unavailable from vendor.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "SSH brute force succeeded on production server at 02:00 AM. Backdoor installed.",
        "urgent", crisis_steps("Security")
    ),
    (
        "GDPR breach notification required: sensitive health data of EU users exposed last night.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "CEO email account compromised. Fraudulent wire-transfer requests sent to finance team.",
        "urgent", crisis_steps("Security")
    ),

    # ---- System outages (15) ----
    (
        "P0 INCIDENT: Payment gateway completely down. No transactions processing. $30k/min loss.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Complete AWS us-east-1 region outage. All primary services down for 500k users.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Database cluster failure. 6 hours of uncommitted transactions potentially lost.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Core network switch failed. Entire London office offline — 300 employees affected.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Mobile app serving 500k users is completely broken after bad canary deployment.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Email server crashed. Zero outbound communications possible company-wide.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Kubernetes control plane unresponsive. Cannot deploy, scale, or restart pods.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "CDN began serving malformed JS injecting ads into our site. Taken offline immediately.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "SSL certificate expired at midnight. All HTTPS connections failing for customers.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Redis cluster in split-brain state. Silent data corruption detected in production.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "CI/CD pipeline injected malicious code into latest release shipped to customers.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Backup system corrupted. Last verified clean backup is 8 days old.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "DNS misconfiguration has our domain resolving to a competitor's IP for 1 hour.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Memory leak crashing prod servers every 90 seconds. Services constantly restarting.",
        "urgent", crisis_steps("Tech Support")
    ),
    (
        "Load balancer misconfigured to route 100% traffic to single overloaded node.",
        "urgent", crisis_steps("Tech Support")
    ),

    # ---- Physical / safety emergencies (10) ----
    (
        "EMERGENCY: Fire in server room B. Halon suppression activated. Data center at risk.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Building lockdown: active security threat reported on Floor 2. All staff to shelter.",
        "urgent", crisis_steps("Security")
    ),
    (
        "Gas leak detected on Floor 4. Full building evacuation in progress.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Employee collapsed in boardroom. Awaiting paramedics. Management must be notified.",
        "urgent", crisis_steps("HR")
    ),
    (
        "Flooding in basement data center. 3 rack units submerged. Emergency shutdown initiated.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Main power failure. Generator failed to start. 40 minutes of UPS backup remaining.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Chemical spill in R&D lab. Building evacuated on east wing. HAZMAT team en route.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Workplace assault reported. Security called. Police on the way to the office.",
        "urgent", crisis_steps("HR")
    ),
    (
        "Roof collapsed on east wing due to heavy snow accumulation. Injuries reported.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Elevator malfunction with 6 employees trapped. Fire brigade contacted.",
        "urgent", crisis_steps("Operations")
    ),

    # ---- Legal / regulatory crises (10) ----
    (
        "Court injunction issued. All operations related to Project Nexus must stop immediately.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Regulatory authority conducting surprise audit NOW. 3 inspectors are in the lobby.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Class action lawsuit filed by 800 customers over the data breach from last month.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Government subpoena received. All records must be preserved and produced within 48 hours.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Competitor secured emergency patent injunction to halt our flagship product sales.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "GDPR investigation formally opened by EU DPA. 30-day response deadline starts today.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Tax authority demanding access to 7 years of financial records by tomorrow morning.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Whistleblower filed formal SEC complaint about financial irregularities in Q2 report.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Media about to publish exposé on our practices. Legal response required within 2 hours.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Employment tribunal starts tomorrow AM. Need all HR records and communications by tonight.",
        "urgent", crisis_steps("Legal")
    ),

    # ---- Financial fraud / crises (10) ----
    (
        "FRAUD ALERT: $3.2M wire transfer executed to unknown overseas account 20 minutes ago.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Payroll system hacked. Employee salaries rerouted to fraudulent bank accounts.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Main operating account drained overnight. $500k missing. Bank fraud team contacted.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Business email compromise scam succeeded. Finance AP team wired $180k to scammers.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "All company credit lines and accounts frozen by bank due to suspicious activity.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Vendor invoice fraud: $400k paid to fake supplier accounts over the past 6 months.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Internal audit reveals $2.5M discrepancy in accounts receivable — possible embezzlement.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Insurance claim for $5M rejected. Dispute must be resolved before month end.",
        "urgent", crisis_steps("Billing & Finance")
    ),
    (
        "Company unable to make payroll this Friday. Cash flow crisis — investor call needed.",
        "urgent", crisis_steps("Management")
    ),
    (
        "Major customer disputing $1.8M invoice. Threatening legal action if not resolved today.",
        "urgent", crisis_steps("Billing & Finance")
    ),

    # ---- Business / reputational crises (10) ----
    (
        "Enterprise client canceled $10M annual contract citing repeated SLA breaches.",
        "urgent", crisis_steps("Management")
    ),
    (
        "Product defect video going viral: 3M views in 6 hours. Brand crisis unfolding.",
        "urgent", crisis_steps("Management")
    ),
    (
        "Key CTO has resigned effective immediately. Engineering team in disarray.",
        "urgent", crisis_steps("Management")
    ),
    (
        "Product recall required: safety defect found causing injury to users. Legal on standby.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Merger announcement leaked to press before IPO lock-up period. Shares halted.",
        "urgent", crisis_steps("Management")
    ),
    (
        "Critical supplier filed for bankruptcy. We have zero raw materials for next 4 weeks.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "40% of engineering team submitted resignations today. Citing toxic management culture.",
        "urgent", crisis_steps("HR")
    ),
    (
        "Regulatory approval for our product withdrawn. All sales must halt immediately.",
        "urgent", crisis_steps("Legal")
    ),
    (
        "Major data centre provider terminating our contract in 72 hours — no backup plan.",
        "urgent", crisis_steps("Operations")
    ),
    (
        "Press release published false financial figures. Trading halted by exchange.",
        "urgent", crisis_steps("Management")
    ),

    # ==================================================================
    # ROUTE TO DEPARTMENT (work / urgent) — 85 examples
    # Steps: [mark_important] + route_to_department + optional reply
    # ==================================================================

    # ---- HR (20) ----
    (
        "A team member has filed a sexual harassment complaint against their direct manager.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Salary dispute: Sarah claims she was verbally promised a promotion 6 months ago with no follow-up.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Employee requesting extended medical leave for severe burnout and mental health treatment.",
        "work", route_steps("HR")
    ),
    (
        "Anonymous report of bullying behavior by a team lead toward junior developers.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Maternity leave request — expecting mother needs HR guidance on entitlements.",
        "work", route_steps("HR")
    ),
    (
        "Discrimination complaint: employee alleges racial bias in the recent promotion cycle.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Employee requesting reasonable workplace accommodation for a visual impairment.",
        "work", route_steps("HR")
    ),
    (
        "Performance improvement plan needed for consistently underperforming team member.",
        "work", route_steps("HR")
    ),
    (
        "Interview scheduling request for the senior data scientist position — 4 candidates.",
        "work", route_steps("HR")
    ),
    (
        "Benefits enrollment question: does the dental plan cover orthodontic treatment for adults?",
        "work", route_steps("HR", important=False)
    ),
    (
        "Onboarding paperwork and IT access setup for new hire Marcus joining next Monday.",
        "work", route_steps("HR")
    ),
    (
        "Exit interview scheduling for senior engineer departing at end of this month.",
        "work", route_steps("HR")
    ),
    (
        "Team offsite request for 25 people in Q3 — need budget approval and venue guidance.",
        "work", route_steps("HR")
    ),
    (
        "Escalating conflict between two senior engineers. Mediation requested by both parties.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Policy question: can an employee legally work remotely from Spain for 2 months?",
        "work", route_steps("HR")
    ),
    (
        "Request for internal transfer: employee wants to move to the Singapore office.",
        "work", route_steps("HR")
    ),
    (
        "Overtime compensation question — employee claims unpaid overtime for last 3 months.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Code of conduct violation reported: employee found to be sharing confidential info with competitor.",
        "work", route_steps("HR", important=True)
    ),
    (
        "Background check results received for engineering candidate — needs HR review before offer.",
        "work", route_steps("HR")
    ),
    (
        "Request for relocation assistance benefits for team member moving to new office city.",
        "work", route_steps("HR")
    ),

    # ---- Management (15) ----
    (
        "Board meeting next Thursday: 25-slide investor deck needs CEO review and sign-off.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Strategic partnership proposal from Series B fintech startup — requires executive review.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Q4 department headcount and budget request needs VP-level approval this week.",
        "work", route_steps("Management", important=True)
    ),
    (
        "M&A due diligence underway — target company financial data needs executive sign-off.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Company-wide reorganization plan ready for leadership review ahead of announcement.",
        "work", route_steps("Management", important=True)
    ),
    (
        "New remote work policy proposal needs C-suite approval before HR publishes it.",
        "work", route_steps("Management")
    ),
    (
        "Emergency capital expenditure of $750k for critical infrastructure needs board approval.",
        "work", route_steps("Management", important=True)
    ),
    (
        "VP of Engineering hire: compensation package is significantly above salary band.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Major shareholder requesting urgent investor relations call this week.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Press release must be approved by CEO before tomorrow's 9 AM embargo-lift.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Board approval needed for expansion into three new international markets.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Succession planning conversation needed: CTO has announced planned retirement in 6 months.",
        "work", route_steps("Management")
    ),
    (
        "Acquisition target identified by strategy team — executive decision required this quarter.",
        "work", route_steps("Management", important=True)
    ),
    (
        "Annual performance review calibration session for director-level and above.",
        "work", route_steps("Management")
    ),
    (
        "Emergency leadership meeting needed: key product pillar has been sunset by biggest partner.",
        "work", route_steps("Management", important=True)
    ),

    # ---- Business Team (12) ----
    (
        "Fortune 500 company submitted a $6M RFP — needs business team response within 5 days.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "Joint venture proposal from European distribution partner for Q1 rollout.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "Sales team needs custom pricing guidance for an enterprise deal worth $2M ARR.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "Top client up for contract renewal negotiation starting in 3 weeks.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "Market feasibility study needed for Southeast Asia expansion — deadline next month.",
        "work", route_steps("Business Team")
    ),
    (
        "Strategic account losing ground to competitor. Business team intervention needed.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "New reseller partnership opportunity in APAC requires business team evaluation.",
        "work", route_steps("Business Team")
    ),
    (
        "Customer success metrics show 30% decline in product usage — churn risk identified.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "Pricing strategy overhaul request for new enterprise product tier launching next quarter.",
        "work", route_steps("Business Team")
    ),
    (
        "Two enterprise clients requesting a co-branded case study — business team to coordinate.",
        "work", route_steps("Business Team")
    ),
    (
        "New channel partnership with system integrator could unlock $10M pipeline.",
        "work", route_steps("Business Team", important=True)
    ),
    (
        "Client escalation requires senior business team involvement to salvage the relationship.",
        "work", route_steps("Business Team", important=True)
    ),

    # ---- Billing & Finance (15) ----
    (
        "Invoice #7821 is overdue by 90 days — client unresponsive to three payment reminders.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Expense report for $12,000 exceeds the standard approval threshold — needs finance review.",
        "work", route_steps("Billing & Finance")
    ),
    (
        "Payroll discrepancy: employee claims September paycheck was $2,400 short.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Vendor threatening legal action if invoice of $35k not paid within 5 business days.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Budget reforecast needed for Q4 following unexpected $200k infrastructure cost.",
        "work", route_steps("Billing & Finance")
    ),
    (
        "Enterprise software license renewal worth $180k/year — evaluation needed before auto-renew.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Tax filing deadline in 3 business days — required documents not yet submitted.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Client charging dispute: $55k in contested charges must be reviewed and resolved.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Bank statement shows $15,000 in unrecognized international charges this month.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Company credit line increase application — CFO approval needed before bank submission.",
        "work", route_steps("Billing & Finance")
    ),
    (
        "Audit committee requesting all Q3 financial documents for tomorrow's review session.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Capital expenditure approval for new GPU cluster — $420k hardware procurement pending.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Annual commercial insurance premium renewal due in 4 business days.",
        "work", route_steps("Billing & Finance", important=True)
    ),
    (
        "Payroll tax compliance question: new office in Texas has different withholding rules.",
        "work", route_steps("Billing & Finance")
    ),
    (
        "Grant funding application for R&D initiative — finance must verify eligibility criteria.",
        "work", route_steps("Billing & Finance")
    ),

    # ---- Tech Support (12) ----
    (
        "Laptop won't power on. Have critical client presentation in 90 minutes. Please help.",
        "work", route_steps("Tech Support", important=True)
    ),
    (
        "Unable to connect to company VPN from home office since yesterday afternoon.",
        "work", route_steps("Tech Support")
    ),
    (
        "Outlook crashes every time I open emails with large PDF attachments.",
        "work", route_steps("Tech Support")
    ),
    (
        "Software provisioning request for 3 new hires starting Monday — licenses needed.",
        "work", route_steps("Tech Support")
    ),
    (
        "Company Slack workspace locked out after MFA device was lost. Entire team affected.",
        "work", route_steps("Tech Support", important=True)
    ),
    (
        "Office printer network connection failed. Department of 30 unable to print.",
        "work", route_steps("Tech Support")
    ),
    (
        "Excel macro stopped working after the Office 365 automatic update yesterday.",
        "work", route_steps("Tech Support")
    ),
    (
        "Zoom microphone and camera not working — all-hands meeting in 45 minutes.",
        "work", route_steps("Tech Support", important=True)
    ),
    (
        "Two-factor authentication app lost — cannot access any company systems remotely.",
        "work", route_steps("Tech Support", important=True)
    ),
    (
        "Corporate Wi-Fi fails to authenticate on the new M3 MacBooks issued last week.",
        "work", route_steps("Tech Support")
    ),
    (
        "Database access credentials expired. Data science team can't run any queries.",
        "work", route_steps("Tech Support", important=True)
    ),
    (
        "Development environment Docker images corrupted — entire engineering team blocked.",
        "work", route_steps("Tech Support", important=True)
    ),

    # ---- Security (8) ----
    (
        "Suspicious vehicle photographing our office building windows for 3 consecutive days.",
        "work", route_steps("Security", important=True)
    ),
    (
        "Access badge found unattended in public lobby area. No name or ID visible.",
        "work", route_steps("Security")
    ),
    (
        "After-hours CCTV motion detected in restricted server room at 3:17 AM.",
        "work", route_steps("Security", important=True)
    ),
    (
        "Physical access logs show unidentified person entering restricted R&D wing twice.",
        "work", route_steps("Security", important=True)
    ),
    (
        "Tailgating incident recorded at secure entrance — badge policy being circumvented.",
        "work", route_steps("Security")
    ),
    (
        "Security camera on Floor 5 has been offline for 72 hours. Blind spot in coverage.",
        "work", route_steps("Security")
    ),
    (
        "Online threat against our COO posted on a professional forum — needs investigation.",
        "work", route_steps("Security", important=True)
    ),
    (
        "Suspicious unmarked package received with no return address. Security screening needed.",
        "work", route_steps("Security", important=True)
    ),

    # ---- Legal (8) ----
    (
        "NDA review and sign-off required before data sharing session with new vendor tomorrow.",
        "work", route_steps("Legal", important=True)
    ),
    (
        "Multi-year contractor agreement requires legal approval — signing ceremony on Friday.",
        "work", route_steps("Legal", important=True)
    ),
    (
        "IP ownership dispute with ex-employee claiming rights to core algorithm.",
        "work", route_steps("Legal", important=True)
    ),
    (
        "Employment contract template needs legal update for new jurisdictions before hiring starts.",
        "work", route_steps("Legal")
    ),
    (
        "Terms of service rewrite required before the new product feature launches next week.",
        "work", route_steps("Legal", important=True)
    ),
    (
        "Licensing compliance audit identified 3 open-source libraries used in violation of GPL.",
        "work", route_steps("Legal", important=True)
    ),
    (
        "Potential anti-competition concern raised by legal counsel regarding planned acquisition.",
        "work", route_steps("Legal", important=True)
    ),
    (
        "Data processing agreement update needed to comply with new CCPA regulations.",
        "work", route_steps("Legal")
    ),

    # ---- Customer Support (10) ----
    (
        "Customer threatening public review on Trustpilot over unresolved 3-week support ticket.",
        "work", route_steps("Customer Support", important=True)
    ),
    (
        "Enterprise customer cannot export their data — SLA breach if not fixed in 2 hours.",
        "work", route_steps("Customer Support", important=True)
    ),
    (
        "Customer accidentally deleted their entire account and needs recovery assistance.",
        "work", route_steps("Customer Support")
    ),
    (
        "Billing error charged customer $5,000 instead of $50 — refund and apology needed.",
        "work", route_steps("Customer Support", important=True)
    ),
    (
        "VIP customer requesting dedicated onboarding support for their 200-seat deployment.",
        "work", route_steps("Customer Support")
    ),
    (
        "Customer reporting product not working after update — revenue impact for their business.",
        "work", route_steps("Customer Support", important=True)
    ),
    (
        "Refund request for annual subscription due to feature being discontinued.",
        "work", route_steps("Customer Support")
    ),
    (
        "Customer data migration from legacy system taking 3x longer than promised.",
        "work", route_steps("Customer Support", important=True)
    ),
    (
        "API rate limits blocking a customer's production integration — business blocked.",
        "work", route_steps("Customer Support", important=True)
    ),
    (
        "Customer requesting invoice with specific legal entity name for compliance purposes.",
        "work", route_steps("Customer Support")
    ),

    # ==================================================================
    # REPLY EMAILS (work / personal) — 75 examples
    # Steps: [reply] or [mark_important, reply]
    # ==================================================================

    # ---- Meeting requests (25) ----
    (
        "Hi, can we find time this week to go over the Q3 product roadmap together?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Would you be available for a 30-min catch-up call on Friday afternoon?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Can we schedule a technical architecture review session for the new API design?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Please let me know your availability for a product alignment meeting next week.",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "I'd like to set up a 1:1 to discuss my career development goals this quarter.",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Team standup time needs to change — what times work for your schedule?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Can we do a quick call to debrief on how yesterday's client demo went?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Proposing a weekly sync between our teams — would Tuesday afternoons work?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "I'd like to walk you through the new service architecture diagram — can we meet?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "When are you free to review the interactive prototype with the design team?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Our investor would like a 20-minute introductory call — what's your schedule next week?",
        "work", reply_steps(MEET_ACK, important=True)
    ),
    (
        "Please RSVP for the all-hands company meeting on Thursday at 2 PM.",
        "work", reply_steps("Thank you for the reminder. I have added it to my calendar and will attend.")
    ),
    (
        "Can we reschedule our interview? I have an unavoidable conflict on the original date.",
        "work", reply_steps("Of course! Please share your availability and we can find a new time.")
    ),
    (
        "Let's plan a sprint retrospective. What time works best for you next week?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Can we block 2 hours for a cross-team strategy planning session early next week?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "I'm requesting a stakeholder review meeting for the upcoming feature release.",
        "work", reply_steps(MEET_ACK, important=True)
    ),
    (
        "Are you available for a budget discussion with the finance team before month-end?",
        "work", reply_steps(MEET_ACK, important=True)
    ),
    (
        "Could we connect to discuss a potential collaboration between our research teams?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "We need to do a joint roadmap planning session. What's your calendar like next month?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Can we set up a code review session for the new authentication module?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "I'd like to invite you to chair our internal AI ethics panel discussion next month.",
        "work", reply_steps("Thank you for the invite! I am happy to participate and will confirm the date shortly.")
    ),
    (
        "Are you attending the company offsite in Barcelona? Would love to grab time there.",
        "work", reply_steps("Yes, I will be there! Let's find some time over the conference schedule.")
    ),
    (
        "Could you join us for a brief knowledge transfer session before your handover?",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "I'd like to schedule a competitor analysis review with the product team.",
        "work", reply_steps(MEET_ACK)
    ),
    (
        "Can we set up a formal project kickoff meeting for the new client engagement?",
        "work", reply_steps(MEET_ACK)
    ),

    # ---- Status updates & follow-ups (25) ----
    (
        "Can you give me a quick status update on the backend migration project timeline?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Just checking in on the status of the pending client proposal submission.",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Have the API integration issues between our systems been fully resolved?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "What's the current status of the hiring process for the senior DevOps engineer role?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Can you confirm last night's database migration completed successfully?",
        "work", reply_steps("The migration completed without errors. All services are running normally.")
    ),
    (
        "Any updates on the vendor evaluation exercise we kicked off last month?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Just following up — did you get a chance to review the design mockups I sent?",
        "personal", reply_steps(STAT_ACK)
    ),
    (
        "Has the client approved the final contract terms? We need to proceed with procurement.",
        "work", reply_steps(STAT_ACK, important=True)
    ),
    (
        "Checking if the technical documentation is ready for the external developer portal.",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "What's the ETA on the refund for invoice #6203? Finance is following up.",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Has the security audit findings report been finalized and shared with the board?",
        "work", reply_steps(STAT_ACK, important=True)
    ),
    (
        "Are we still on track for the Q4 product launch? Any blockers I should know about?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Can you share the action items and notes from yesterday's planning session?",
        "work", reply_steps("Sure! I will share the meeting notes doc in the team Slack channel shortly.")
    ),
    (
        "Any progress on the performance optimization tasks assigned last sprint?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "When can we expect the preliminary results from the user research study?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Has the data migration to the new warehouse finished? I need access to the new tables.",
        "work", reply_steps(STAT_ACK, important=True)
    ),
    (
        "Could you confirm whether the vendor contract has been finalized? Procurement is waiting.",
        "work", reply_steps(STAT_ACK, important=True)
    ),
    (
        "Just following up on the partnership discussion we had in the Q3 business review.",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Has the post-incident report from last week's outage been reviewed and closed?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "What's the delivery timeline for the custom analytics dashboard we commissioned?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Just circling back on the budget proposal I submitted two weeks ago — any decision?",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Has the new office Wi-Fi infrastructure been installed? Team needs it by Monday.",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "I wanted to check if the onboarding materials are ready for the 5 new analysts joining.",
        "work", reply_steps(STAT_ACK)
    ),
    (
        "Can you let me know when the additional server capacity will be provisioned?",
        "work", reply_steps(STAT_ACK, important=True)
    ),
    (
        "I'm following up on the user access request I submitted for the production environment.",
        "work", reply_steps(STAT_ACK)
    ),

    # ---- General professional / personal (25) ----
    (
        "I'd love to get your feedback on the new feature prototype before we share it with users.",
        "work", reply_steps("Happy to take a look! Send it over and I will share my thoughts by tomorrow.")
    ),
    (
        "Would you be willing to mentor a junior team member on our team this quarter?",
        "personal", reply_steps("I would be glad to help. Let's set up an intro call with the team member.")
    ),
    (
        "Congratulations on your promotion to VP! Looking forward to working under your leadership.",
        "personal", reply_steps("Thank you so much for the kind words! Excited for what's ahead.")
    ),
    (
        "I'm writing to introduce myself as your new account manager for the renewal.",
        "work", reply_steps("Great to meet you! Looking forward to working together on the account.")
    ),
    (
        "Could you review and sign the attached NDA at your convenience before our demo?",
        "work", reply_steps("Of course. I will review it and come back to you within 24 hours.", important=True)
    ),
    (
        "Our team would love your thoughts on the beta release before the public launch.",
        "work", reply_steps("Happy to help! I will test it this week and send detailed feedback.")
    ),
    (
        "Are you interested in participating as a subject in our UX research study?",
        "personal", reply_steps("Sure, happy to participate. Please share the session details.")
    ),
    (
        "I have a few questions about the technical specification document for the new API.",
        "work", reply_steps("Feel free to send over your questions and I will answer them promptly.")
    ),
    (
        "Could you share the report template you used for the quarterly business review?",
        "work", reply_steps("Of course! I will send the template link to you shortly.")
    ),
    (
        "I'm the new intern joining the backend team. Could you walk me through the codebase?",
        "work", reply_steps("Welcome! Let's schedule a walkthrough session—I'll reach out to set it up.")
    ),
    (
        "Thank you for the warm introduction! Happy to grab a virtual coffee and learn more.",
        "personal", reply_steps("Absolutely! Let's find 30 minutes on the calendar this week.")
    ),
    (
        "Do you have any recommendations for project management tools for a remote team of 15?",
        "work", reply_steps("Great question! I have a few favourites I can share — I will send a short write-up.")
    ),
    (
        "I'd like to request a professional reference letter for a job application I'm pursuing.",
        "personal", reply_steps("Happy to write one for you. Please send the role details and I'll get started.")
    ),
    (
        "Can you advise on the best approach for the microservices migration we're planning?",
        "work", reply_steps("Happy to share thoughts. Let's schedule a technical discussion to go through the options.")
    ),
    (
        "Would you like to co-author a technical blog post about our open source project?",
        "work", reply_steps("That sounds great! I'd love to collaborate — let's outline the article together.")
    ),
    (
        "I'm preparing a conference presentation and would appreciate your expert input on the topic.",
        "work", reply_steps("Happy to help! Share the draft when ready and I'll provide detailed feedback.")
    ),
    (
        "Can you review this pull request before it goes to production? It's time-sensitive.",
        "work", reply_steps("On it! I will review it within the hour.", important=True)
    ),
    (
        "Is there a standard format we use for engineering status reports sent to leadership?",
        "work", reply_steps("Yes, I'll share the template and style guide — give me a moment.")
    ),
    (
        "How should I handle the escalation from the unhappy enterprise client?",
        "work", reply_steps("Good question. Let's align before you respond — I'll call you in 10 minutes.", important=True)
    ),
    (
        "I have a few follow-up questions from the onboarding training session yesterday.",
        "work", reply_steps("Of course! Write them all out and I'll answer each one thoroughly.")
    ),
    (
        "Your expertise would be incredibly valuable at our upcoming design system sprint.",
        "work", reply_steps("Thank you for the invite! Happy to contribute — please share the sprint schedule.")
    ),
    (
        "I wanted to reach out and thank you for the career advice during the last mentorship session.",
        "personal", reply_steps("It was my pleasure. Keep up the great work and feel free to reach out anytime!")
    ),
    (
        "We are hosting a hackathon next month and would love you to be one of our judges.",
        "work", reply_steps("I'd be honored to judge! Please share the event details and judging criteria.")
    ),
    (
        "Hi, I saw your conference talk and had some questions about the ML pipeline you described.",
        "personal", reply_steps("Thank you for your interest! Please go ahead and ask — happy to explain further.")
    ),
    (
        "Can you share your thoughts on the new company brand guidelines for the website redesign?",
        "work", reply_steps("Absolutely. I'll review the document and send you my feedback by end of week.")
    ),

    # ==================================================================
    # SPAM EMAILS — 95 examples (1 step: mark_spam)
    # ==================================================================

    # ---- Phishing (20) ----
    *(
        (email, "spam", [{"action": "mark_spam"}])
        for email in [
            "Your Microsoft 365 account will be suspended! Verify your credentials at this link immediately.",
            "URGENT: Unusual sign-in detected on your account. Click here to secure it now.",
            "Your PayPal account has been limited. Provide your details to restore full access.",
            "IT department notice: your password has expired. Reset it now using this secure portal.",
            "Your company email storage is 99% full. Click here to upgrade your quota today.",
            "Security alert: someone is trying to hack your Gmail account. Verify now to prevent it.",
            "HR update: payroll details must be re-entered by Friday or salary will be withheld.",
            "Apple ID locked. Verify your identity within 24h to avoid permanent suspension.",
            "FINAL NOTICE: Your DocuSign document requires your immediate signature before it expires.",
            "Dropbox shared a file with you. Click to view it before the link expires in 2 hours.",
            "Your bank requires you to re-confirm details due to a system upgrade — act now.",
            "Your login was used to access your account from an unknown device in Nigeria.",
            "Zoom invitation from CEO — join this emergency all-hands meeting now.",
            "Your Office 365 license has expired — click to renew and avoid data loss.",
            "Important: your company VPN certificate has expired. Install the new one using this link.",
            "Your Netflix subscription is expiring today. Update payment details to continue.",
            "Amazon: your order could not be shipped due to incorrect address. Update now.",
            "LinkedIn: someone viewed your profile 47 times today. Click to see who.",
            "Your cloud backup failed. Recover your files immediately using this secure portal.",
            "IT Security: mandatory 2FA app update required within 6 hours. Follow this guide.",
        ]
    ),

    # ---- Lottery / prize scams (20) ----
    *(
        (email, "spam", [{"action": "mark_spam"}])
        for email in [
            "Congratulations! You have been randomly selected as our £1,000,000 sweepstakes winner!",
            "You have won a brand new iPhone 16 Pro! Claim your prize by clicking the link below.",
            "WINNER NOTIFICATION: Your email won the international online lottery — claim now.",
            "You are the lucky 10,000th visitor to our site! Claim your $500 Amazon gift card.",
            "We tried to deliver your prize but you weren't home — schedule redelivery now.",
            "You have been selected as a finalist for our $50,000 cash giveaway. Act fast!",
            "Your email address was selected in our special draw. You have won a free cruise.",
            "Claim your Google reward of £850 — your email was drawn as this month's winner.",
            "You have won an all-expenses-paid trip to Maldives! Respond within 48 hours.",
            "IMPORTANT: PayPal is giving away $200 to 500 lucky users — you are one of them.",
            "Microsoft is rewarding loyal users with a $500 voucher. Your email was selected!",
            "Samsung grand prize draw: you've won a Galaxy S25 Ultra. Claim before it expires.",
            "Your company email was entered in our global raffle — you've won $10,000 cash!",
            "Congratulations! You qualify for a complimentary luxury car test drive and $200 voucher.",
            "You have been selected for a free home appliance package — shipping info needed.",
            "URGENT: your unclaimed prize worth $2,500 will expire midnight. Click to claim.",
            "You won! Our automated system selected your email for a special anniversary prize.",
            "Collect your exclusive reward: 3-night hotel stay for two — no purchase necessary.",
            "You have been pre-selected for our VIP customer appreciation gift of $1,000.",
            "Exclusive: you have won lifetime Premium access to our platform. Activate it now.",
        ]
    ),

    # ---- Investment / money scams (20) ----
    *(
        (email, "spam", [{"action": "mark_spam"}])
        for email in [
            "Earn 300% returns on your investment with our AI-powered trading bot. Join now.",
            "Bitcoin is about to surge — invest $500 today and earn $5,000 by next week.",
            "Our hedge fund has averaged 45% annual returns. Limited spots available — invest today.",
            "Nigerian prince needs your help moving $45M. You keep 30% for your assistance.",
            "Work from home and earn $800/day with our proven system — no experience needed.",
            "Make $5,000 a week trading forex. Our robot does all the work — just deposit now.",
            "Guaranteed passive income: invest in our crypto fund and earn 10% monthly.",
            "Exclusive: pre-IPO shares in a unicorn startup available for just $1,000.",
            "You have been approved for a $15,000 business loan with zero credit check.",
            "Earn unlimited income by sharing our product links. No selling required!",
            "Our MLM opportunity lets you earn $3,000 a month working just 2 hours a day.",
            "Send $200 in gift cards to claim your inheritance package worth $2.5M.",
            "Your dormant bank account in our institution holds $800,000. Contact us to claim it.",
            "Double your money in 30 days with our guaranteed safe investment program.",
            "Exclusive NFT drop — early investors made 500% gains. Buy yours before it sells out.",
            "Day trading secrets the banks don't want you to know. Buy our course for $99.",
            "Government grant available for small business owners — apply and receive $25,000.",
            "Turn your $1,000 into $12,000 in just 2 months using this one weird trick.",
            "Our Ponzi-proof investment scheme pays 15% weekly. Wire funds today.",
            "AI stock picker predicted 10 winners in a row — subscribe for $29/month.",
        ]
    ),

    # ---- Job / recruitment scams (15) ----
    *(
        (email, "spam", [{"action": "mark_spam"}])
        for email in [
            "Work from home as a mystery shopper and earn $500 per task — no experience required.",
            "You have been shortlisted for a remote job paying $120k — no interview needed.",
            "We reviewed your profile and are offering you a job without any application process.",
            "Part-time data entry work from home: earn $50/hour, flexible hours, start today.",
            "Congratulations! You're hired as our brand ambassador — $200/day from home.",
            "Urgent vacancy: earn $5,000/week working as our company's payment processor.",
            "Work as an online English tutor with no certification required — earn $80/hour.",
            "We are looking for people to test apps from home — earn up to $300 per day.",
            "High-paying remote job: re-ship products from home and earn $2,500/week.",
            "Be your own boss! Our turnkey dropshipping business earns $10k/month guaranteed.",
            "Earn $250/survey by sharing your opinion with our research firms. No skills needed.",
            "Amazon hiring work-from-home customer agents. Apply using this unofficial portal.",
            "Your LinkedIn profile was selected for our executive talent program — $200k roles.",
            "Be a product reviewer: keep free products and get paid $50 each — no experience.",
            "Immediate hire: be our financial intermediary and keep 5% of transactions processed.",
        ]
    ),

    # ---- Other scams / fake notices (20) ----
    *(
        (email, "spam", [{"action": "mark_spam"}])
        for email in [
            "Your parcel is stuck in customs — pay a £2.99 release fee to receive it today.",
            "FINAL TAX NOTICE: You owe £3,400. Pay now to avoid criminal prosecution.",
            "Your computer is infected with a virus — call this number immediately to remove it.",
            "Verify your account or it will be permanently deleted within 24 hours.",
            "Confirm your identity to receive your COVID-19 relief payment of $1,400.",
            "Your DVLA driving licence has expired — renew immediately at this unofficial portal.",
            "Important update from HMRC: click to file your self-assessment before the deadline.",
            "Warning: your social security number is being used for illegal activities.",
            "Your streaming account has been hacked. Urgent action required — click here.",
            "Free credit score check now — just enter your full card details to proceed.",
            "Your cloud storage is leaking private files — click to secure it immediately.",
            "Medicare is issuing new cards — verify your information to receive yours.",
            "Claim your PPI payment — our records show you are owed £4,200 in compensation.",
            "URGENT: Your domain is expiring tomorrow. Renew here or lose it permanently.",
            "Please send iTunes gift cards to recover your hacked business account.",
            "Your antivirus license expired 3 days ago. Download renewal from this secure link.",
            "We noticed unauthorized access to your pension account — verify to lock it down.",
            "Confidential debt cancellation program — your $50,000 debt can be erased legally.",
            "You qualify for a free solar panel installation — no fees, government scheme.",
            "Click here to cancel your unauthorized $199 subscription before it renews tonight.",
        ]
    ),

    # ==================================================================
    # IGNORE EMAILS (promotion) — 95 examples (1 step: ignore)
    # ==================================================================

    # ---- Newsletters / digests (20) ----
    *(
        (email, "promotion", [{"action": "ignore"}])
        for email in [
            "This week in tech: AI breakthroughs, new chip announcements, and startup funding rounds.",
            "Your weekly digest from ProductHunt — 10 trending tools curated just for you.",
            "The Morning Brew: business news, markets update, and today's top stories.",
            "Weekly roundup: the best articles we read this week on leadership and growth.",
            "Your GitHub Trending digest — top repositories this week in Python and TypeScript.",
            "Hacker News Weekly: most upvoted stories from the past 7 days.",
            "Newsletter: our editorial team's top picks for design inspiration this month.",
            "Your monthly reading list from our editorial team — 8 must-read articles.",
            "TechCrunch Roundup: this week's biggest funding rounds, exits, and launches.",
            "Substack Weekly Digest: stories from creators you follow — 12 new articles.",
            "The Data Science Weekly: trending papers, tutorials, and frameworks.",
            "Your curated newsletter: remote work tips, productivity hacks, and career advice.",
            "Weekly AI recap: GPT updates, image generation breakthroughs, agent developments.",
            "Your favorite creator just published 3 new posts — catch up now.",
            "The Sunday Briefing: what happened in the world of startups this week.",
            "Your personalised news feed is ready — 15 articles matched to your interests.",
            "This Week in Open Source: new releases, contributors, and community highlights.",
            "Product updates digest: all new features released by your SaaS tools in July.",
            "Developer weekly: new tools, libraries, and best practices from the community.",
            "Cloud Computing Monthly: AWS, GCP, Azure news and benchmark comparisons.",
        ]
    ),

    # ---- Product promotions / sales (25) ----
    *(
        (email, "promotion", [{"action": "ignore"}])
        for email in [
            "Summer sale: up to 60% off on all electronics. Shop now before stock runs out!",
            "Flash sale! 24 hours only — buy 2, get 1 free on our entire clothing range.",
            "Limited edition sneakers dropping tomorrow at 10 AM — set your reminder now.",
            "Exclusive member price: MacBook Pro 16\" for $500 off this weekend only.",
            "Our biggest sale of the year starts NOW. Up to 70% off sitewide — shop fast.",
            "Introducing our new summer collection — fresh styles, bold colours, shop now.",
            "Members-only early access: new season arrivals available 24h before launch.",
            "New arrivals this week: check out the latest in home decor and furniture.",
            "Your wishlist items are back in stock! Don't miss out — grab them before they sell out.",
            "Unbeatable prices on laptops, tablets, and accessories — deals end midnight.",
            "Huge gym membership discounts this month — join for just £15 and get fit!",
            "Top picks for your home office upgrade — ergonomic chairs, monitors, and more.",
            "Buy one, get one 50% off on all skincare products this week. T&Cs apply.",
            "Weekend deals are here — tech, fashion, and beauty at prices you'll love.",
            "Back to school essentials at incredible prices — stock up now before it ends.",
            "Handpicked deals just for you based on your browsing history — shop now.",
            "Clearance sale: save up to 80% on last season's inventory. Limited stock.",
            "Happy birthday! Here's an exclusive 25% discount valid for the next 48 hours.",
            "Our loyalty rewards double this month — place an order and earn 2x points.",
            "We've matched the competitor price — your favourite item is now even cheaper.",
            "End-of-season sale: up to 40% off outdoor furniture and garden accessories.",
            "Your cart is about to expire — complete your purchase and save an extra 10%.",
            "New product launch: introducing the next generation smartwatch — pre-order now.",
            "Father's Day gift ideas curated for you — shop our top seller gift sets.",
            "Sitewide sale: use code SAVE20 at checkout for an extra 20% off everything.",
        ]
    ),

    # ---- Events & webinars (20) ----
    *(
        (email, "promotion", [{"action": "ignore"}])
        for email in [
            "You're invited: free webinar on mastering productivity with AI tools — register now.",
            "Join our virtual summit on the future of remote work — 20 expert speakers.",
            "Free online workshop: how to launch your side hustle in 30 days. Spots limited.",
            "Join 5,000 professionals at our annual SaaS conference in San Francisco.",
            "Webinar alert: building resilient microservices on Kubernetes — Thursday at 3 PM.",
            "You are invited to our product launch event — RSVP for early access and swag.",
            "Leadership masterclass featuring 3 Fortune 500 CEOs — free to attend online.",
            "Our monthly community meetup is happening next Wednesday — all are welcome.",
            "Hackathon: build with our API and win $10,000 in prizes. Registration open now.",
            "Panel discussion: the ethics of AI in hiring — join the live debate this Friday.",
            "Workshop: introduction to large language models — no prior ML experience required.",
            "Cloud architecture bootcamp — 3-day virtual event, hands-on labs included.",
            "Join our live Q&A with the product team — ask anything about the new features.",
            "Annual developer conference: 50+ talks on web, mobile, and cloud. Register today.",
            "Networking event for startup founders — meet investors and fellow entrepreneurs.",
            "Book club invitation: we are reading 'Zero to One' this month — join the discussion.",
            "Podcast live recording: join us in-studio or watch the live stream Friday evening.",
            "Industry awards ceremony: your company has been nominated — RSVP for the gala.",
            "Professional development day: CPD-accredited workshops running all day Saturday.",
            "Lunch and learn: HR compliance updates for tech companies — free online session.",
        ]
    ),

    # ---- Generic marketing / social media (30) ----
    *(
        (email, "promotion", [{"action": "ignore"}])
        for email in [
            "Your Spotify Wrapped is here — see your top songs and artists of the year.",
            "Don't forget to rate your recent Uber ride — your feedback shapes the experience.",
            "You have 3 unread notifications on LinkedIn. See what your network is up to.",
            "Our app now supports dark mode — update now for the new look and feel.",
            "Did you know you have 4,200 reward points waiting to be redeemed? Shop now.",
            "New features just launched — check out what's new in the latest app update.",
            "We've refreshed our mobile app. Update now for a smoother, faster experience.",
            "Your subscription renews in 30 days — here's what's included in your plan.",
            "You haven't logged in for 30 days. Here's what's new since your last visit.",
            "We value your feedback! Take our 2-minute survey and help us improve.",
            "Someone liked your post on X — see who's engaging with your content.",
            "Your annual account summary is ready — here's a look back at your year with us.",
            "We noticed you abandoned your cart. Come back and complete your order today!",
            "Your favourite creator just went live — tune in before the stream ends.",
            "Trending this week on our platform: top videos, posts, and creators to follow.",
            "We've updated our privacy policy — here's a clear summary of what changed.",
            "Thank you for being a valued customer. Here's a special note from our CEO.",
            "Our end-of-year recap: highlights from the product, team, and community in 2025.",
            "You have unused credits in your account! Apply them to your next purchase.",
            "Our AI has curated a playlist just for you based on your recent activity.",
            "Join 2M users who upgraded to Premium — enhanced features, no ads, priority support.",
            "Your referral earned you $20 in credits! Redeem them on your next order.",
            "We'd love to see you back! Here are 5 new things we've added since you left.",
            "Top picks this week: hand-curated recommendations from our editorial team.",
            "Your annual subscription savings: you saved $240 vs monthly billing this year.",
            "New partnership announcement: we've integrated with your favourite tool.",
            "Redesign preview: our new UI is coming — sneak peek inside for members.",
            "Community spotlight: see how teams like yours are using our platform.",
            "App of the week: you've been featured in our recommended apps list.",
            "Seasonal greeting from our team — wishing you a wonderful holiday period.",
        ]
    ),
]


# ---------------------------------------------------------------------------
# Deduplicate, assign IDs, shuffle, split
# ---------------------------------------------------------------------------
seen = set()
unique: list[tuple] = []
for item in ALL:
    key = item[0][:60]
    if key not in seen:
        seen.add(key)
        unique.append(item)

random.shuffle(unique)

n = len(unique)
split_idx = int(n * 0.80)
train_raw = unique[:split_idx]
test_raw = unique[split_idx:]

train_path = DOCS / "train_data.jsonl"
test_path = DOCS / "test_data.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for i, (email_text, category, steps) in enumerate(train_raw, start=1):
        record = {"id": i, "email_text": email_text, "category": category, "steps": steps}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

with open(test_path, "w", encoding="utf-8") as f:
    for i, (email_text, category, steps) in enumerate(test_raw, start=1):
        record = {"id": i, "email_text": email_text, "category": category, "steps": steps}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Total unique examples : {n}")
print(f"Train set             : {len(train_raw)} examples → {train_path}")
print(f"Test  set             : {len(test_raw)}  examples → {test_path}")

# Quick distribution summary
from collections import Counter

def step_action_counts(rows):
    first_actions = Counter(r[2][0]["action"] for r in rows)
    return dict(first_actions.most_common())

print("\nTrain — first-step action distribution:")
for action, count in step_action_counts(train_raw).items():
    print(f"  {action:<28} {count}")

print("\nTest — first-step action distribution:")
for action, count in step_action_counts(test_raw).items():
    print(f"  {action:<28} {count}")
