import html
import re
from typing import Optional, Tuple


# Regex pattern to identify acute self-harm, suicidal ideation, or immediate danger
CRISIS_PATTERNS = [
    r"\b(suicid\w*|(kill|killing)\s+(myself|me)|(end|ending)\s+my\s+life|want\s+to\s+die|(hang|hanging)\s+myself)\b",
    r"\b((cut|cutting)\s+myself|(harm|harming)\s+myself|self[\s-]harm|(hurt|hurting)\s+myself|(bleed|bleeding)\s+out)\b",
    r"\b((take|taking)\s+all\s+my\s+pills|overdos\w*|(slit|slitting)\s+my\s+wrists)\b",
    r"\b(no\s+reason\s+to\s+live|better\s+off\s+dead|can't\s+go\s+on\s+living)\b"
]

CRISIS_COMPILED = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

CRISIS_RESPONSE_TEXT = (
    "I hear how much pain you are in right now, and I care about your safety. "
    "Because I am an automated conversational guide and cannot provide crisis or medical care, "
    "please reach out for immediate human support:\n\n"
    "• If you are in immediate danger or feel you cannot keep yourself safe, please contact your local "
    "emergency services (such as 911, 112, 999, or your country's local emergency response) immediately.\n"
    "• Reach out right now to a trusted person—a friend, family member, doctor, or local healthcare professional.\n"
    "• In many countries, free and confidential crisis lines are available 24/7 (such as dialing 988 in the US and Canada, "
    "or reaching out to your local hospital or crisis center).\n\n"
    "You do not have to carry this alone. Please connect with someone who can be with you right now."
)

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"you\s+are\s+now\s+(a\s+doctor|dan|jailbroken|unrestricted)",
    r"system\s*prompt\s*override",
    r"disregard\s+all\s+safety\s+rules",
    r"pretend\s+you\s+have\s+a\s+medical\s+license"
]

INJECTION_COMPILED = re.compile("|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)


class SafetyGuardrail:
    """
    Deterministic safety and crisis intervention engine.
    Ensures safe handling of user input prior to LLM or RAG processing.
    """

    @classmethod
    def check_crisis(cls, text: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluates input text for immediate danger or self-harm signals.
        Returns (is_crisis, crisis_message).
        """
        if not text:
            return False, None

        if CRISIS_COMPILED.search(text):
            return True, CRISIS_RESPONSE_TEXT

        return False, None

    @classmethod
    def sanitize_user_input(cls, text: str) -> str:
        """
        Sanitizes user input to mitigate prompt injection and XML boundary escaping.
        Neutralizes overt injection attempts while preserving user sentiment.
        """
        if not text:
            return ""

        # Neutralize XML tag injection
        sanitized = html.escape(text.strip())

        # Check for injection attempt
        if INJECTION_COMPILED.search(sanitized):
            sanitized = re.sub(
                INJECTION_COMPILED,
                "[User input segment omitted for system safety]",
                sanitized
            )

        return sanitized

    @classmethod
    def audit_model_output(cls, output_text: str) -> str:
        """
        Post-processes model generation to guarantee non-clinical guardrails.
        Replaces any accidental diagnostic declarations or medical claims.
        """
        # Ensure model does not state "I diagnose you with"
        bad_phrases = [
            (r"\b(i\s+diagnose\s+you\s+with)\b", "it sounds like you may be experiencing symptoms of"),
            (r"\b(i\s+prescribe)\b", "a supportive practice often recommended is"),
            (r"\b(as\s+your\s+therapist)\b", "as an emotional-support guide"),
            (r"\b(as\s+a\s+licensed\s+therapist)\b", "as an automated emotional-support guide")
        ]

        cleaned = output_text
        for pattern, replacement in bad_phrases:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        return cleaned
