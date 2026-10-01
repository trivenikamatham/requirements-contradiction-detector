import abc
import json
import logging
import re
from typing import Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters an error during analysis."""
    pass


class BaseLLMProvider(abc.ABC):
    """
    Abstract interface for LLM-based requirement conflict reasoning.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the human-readable name of the LLM provider."""
        pass

    @abc.abstractmethod
    def analyze_pair(
        self,
        req_a_text: str,
        req_b_text: str,
        meta_a: Optional[Dict[str, Any]] = None,
        meta_b: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a pair of requirements.
        Returns a dictionary with:
            - 'classification': 'CONTRADICTORY', 'OVERLAPPING', or 'RELATED'
            - 'conflict_type': str
            - 'confidence': float (0.0 to 1.0)
            - 'explanation': str
            - 'suggested_resolution': str
        """
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """
    Examines requirement pairs using OpenAI's Chat Completion API with JSON response format.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', '')
        self.model = model or getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')

        if not self.api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured.")

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def analyze_pair(
        self,
        req_a_text: str,
        req_b_text: str,
        meta_a: Optional[Dict[str, Any]] = None,
        meta_b: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            doc_a = meta_a.get('doc_name', 'Doc A') if meta_a else 'Doc A'
            doc_b = meta_b.get('doc_name', 'Doc B') if meta_b else 'Doc B'

            system_prompt = (
                "You are an expert Systems Requirements Engineer and Formal Methods Analyst. "
                "Analyze two software requirements for potential contradictions, overlaps, or compatibility.\n\n"
                "Classifications allowed:\n"
                "1. 'CONTRADICTORY': Mutually exclusive constraints, incompatible performance/security requirements, opposing modalities, or unresolvable architectural clashes.\n"
                "2. 'OVERLAPPING': Redundant statements, duplicate functional scope, or overlapping specifications that should be unified.\n"
                "3. 'RELATED': Complementary requirements in the same domain that co-exist without conflict.\n\n"
                "You MUST output valid JSON with exactly the following keys:\n"
                "{\n"
                "  \"classification\": \"CONTRADICTORY\" | \"OVERLAPPING\" | \"RELATED\",\n"
                "  \"conflict_type\": \"<concise category, e.g. Direct Contradiction, SLA Inconsistency, Security Access Conflict, Redundant Specification, Compatible>\",\n"
                "  \"confidence\": <float between 0.0 and 1.0>,\n"
                "  \"explanation\": \"<thorough explanation citing specific clauses and why they clash or overlap>\",\n"
                "  \"suggested_resolution\": \"<actionable engineering or business resolution recommendation>\"\n"
                "}"
            )

            user_prompt = (
                f"Requirement 1 (from {doc_a}):\n\"{req_a_text}\"\n\n"
                f"Requirement 2 (from {doc_b}):\n\"{req_b_text}\"\n\n"
                "Provide your formal analysis in JSON."
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)

            # Validate and normalize classification
            cls = parsed.get("classification", "RELATED").upper()
            if cls not in ("CONTRADICTORY", "OVERLAPPING", "RELATED"):
                cls = "RELATED"

            return {
                "classification": cls,
                "conflict_type": parsed.get("conflict_type", "General Assessment"),
                "confidence": float(parsed.get("confidence", 0.85)),
                "explanation": parsed.get("explanation", "Analyzed by LLM."),
                "suggested_resolution": parsed.get("suggested_resolution", "Review both requirements with project stakeholders.")
            }
        except Exception as e:
            logger.error(f"OpenAI analysis call failed: {e}")
            # Fall back to SafeMockLLMProvider on error
            fallback = SafeMockLLMProvider()
            return fallback.analyze_pair(req_a_text, req_b_text, meta_a, meta_b)


class GroqLLMProvider(BaseLLMProvider):
    """
    Examines requirement pairs using Groq's high-speed inference API.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or getattr(settings, 'GROQ_API_KEY', '')
        self.model = model or getattr(settings, 'GROQ_MODEL', 'llama-3.3-70b-versatile')

        if not self.api_key:
            raise LLMProviderError("GROQ_API_KEY is not configured.")

    @property
    def name(self) -> str:
        return f"Groq ({self.model})"

    def analyze_pair(
        self,
        req_a_text: str,
        req_b_text: str,
        meta_a: Optional[Dict[str, Any]] = None,
        meta_b: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)

            doc_a = meta_a.get('doc_name', 'Doc A') if meta_a else 'Doc A'
            doc_b = meta_b.get('doc_name', 'Doc B') if meta_b else 'Doc B'

            system_prompt = (
                "You are an expert Systems Requirements Engineer. "
                "Analyze two software requirements for potential contradictions, overlaps, or compatibility.\n"
                "Classifications: 'CONTRADICTORY', 'OVERLAPPING', or 'RELATED'.\n"
                "Return JSON with keys: classification, conflict_type, confidence, explanation, suggested_resolution."
            )

            user_prompt = (
                f"Requirement 1 ({doc_a}): \"{req_a_text}\"\n"
                f"Requirement 2 ({doc_b}): \"{req_b_text}\"\n"
                "Respond strictly with valid JSON."
            )

            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            parsed = json.loads(completion.choices[0].message.content)
            cls = parsed.get("classification", "RELATED").upper()
            if cls not in ("CONTRADICTORY", "OVERLAPPING", "RELATED"):
                cls = "RELATED"

            return {
                "classification": cls,
                "conflict_type": parsed.get("conflict_type", "Requirements Assessment"),
                "confidence": float(parsed.get("confidence", 0.85)),
                "explanation": parsed.get("explanation", "Evaluated via Groq."),
                "suggested_resolution": parsed.get("suggested_resolution", "Review with business and architecture teams.")
            }
        except Exception as e:
            logger.error(f"Groq analysis call failed: {e}")
            fallback = SafeMockLLMProvider()
            return fallback.analyze_pair(req_a_text, req_b_text, meta_a, meta_b)


class SafeMockLLMProvider(BaseLLMProvider):
    """
    Safe offline heuristic & semantic conflict evaluator.
    Operates without API keys or internet connection.
    Detects:
      - Modality / Negation contradictions (shall vs shall not, permitted vs prohibited)
      - Numerical / Threshold discrepancies (timeouts, response times, password lengths, retention days)
      - Architectural / Protocol incompatibilities (cloud vs on-prem, OAuth vs SAML, PostgreSQL vs NoSQL)
      - Redundant & overlapping statements
      - Complementary non-conflicting pairs
    """

    @property
    def name(self) -> str:
        return "Safe Heuristic LLM Mock (Offline Engine)"

    def analyze_pair(
        self,
        req_a_text: str,
        req_b_text: str,
        meta_a: Optional[Dict[str, Any]] = None,
        meta_b: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        text_a_lower = req_a_text.lower()
        text_b_lower = req_b_text.lower()

        doc_a_name = meta_a.get('doc_name', 'Doc 1') if meta_a else 'Doc 1'
        doc_b_name = meta_b.get('doc_name', 'Doc 2') if meta_b else 'Doc 2'

        # 1. Check for Direct Modality / Negation Incompatibilities
        contradiction_patterns = [
            (
                r'\b(shall|must|will|requires)\b',
                r'\b(shall not|must not|will not|cannot|prohibited|forbidden|never)\b',
                "Direct Modality Negation Contradiction",
                "One requirement mandates an action/capability while the other explicitly forbids or restricts it.",
                "Consult the Product Owner to determine if this action is strictly prohibited or mandatory in this operating context."
            ),
            (
                r'\b(allow|allows|permit|permits|support|supports)\b',
                r'\b(disallow|disallows|prohibit|prohibits|prevent|restricts?|deny)\b',
                "Access / Permission Contradiction",
                "One requirement permits user access/behavior while the other denies or restricts it.",
                "Harmonize the permission matrix and role-based access control (RBAC) policy."
            ),
        ]

        for pos_pat, neg_pat, ctype, expl_template, res_template in contradiction_patterns:
            has_pos_a = bool(re.search(pos_pat, text_a_lower)) and not bool(re.search(neg_pat, text_a_lower))
            has_neg_b = bool(re.search(neg_pat, text_b_lower))
            has_pos_b = bool(re.search(pos_pat, text_b_lower)) and not bool(re.search(neg_pat, text_b_lower))
            has_neg_a = bool(re.search(neg_pat, text_a_lower))

            # Also check if they share common context topic (at least 2 significant words)
            words_a = set(re.findall(r'\b[a-z]{4,}\b', text_a_lower)) - {'shall', 'must', 'will', 'system', 'user', 'should'}
            words_b = set(re.findall(r'\b[a-z]{4,}\b', text_b_lower)) - {'shall', 'must', 'will', 'system', 'user', 'should'}
            shared_topic = len(words_a & words_b) >= 2

            if shared_topic and ((has_pos_a and has_neg_b) or (has_pos_b and has_neg_a)):
                shared_words = ", ".join(list(words_a & words_b)[:3])
                return {
                    "classification": "CONTRADICTORY",
                    "conflict_type": ctype,
                    "confidence": 0.92,
                    "explanation": (
                        f"Contradictory modality detected regarding topic '{shared_words}'. "
                        f"{expl_template} '{doc_a_name}' specifies mandatory behavior, whereas '{doc_b_name}' "
                        f"states a prohibition or negative constraint."
                    ),
                    "suggested_resolution": res_template
                }

        # 2. Check for Numerical & SLA Inconsistencies (e.g., timeouts, response times, lengths, retention)
        num_conflict = self._check_numerical_conflict(text_a_lower, text_b_lower, doc_a_name, doc_b_name)
        if num_conflict:
            return num_conflict

        # 3. Check for Domain-Specific Technical Clashes
        tech_conflict = self._check_tech_incompatibilities(text_a_lower, text_b_lower, doc_a_name, doc_b_name)
        if tech_conflict:
            return tech_conflict

        # 4. Check for Overlapping / Duplicate Specifications
        words_a_all = set(re.findall(r'\b[a-z]{3,}\b', text_a_lower))
        words_b_all = set(re.findall(r'\b[a-z]{3,}\b', text_b_lower))
        if words_a_all and words_b_all:
            jaccard = len(words_a_all & words_b_all) / len(words_a_all | words_b_all)
            if jaccard >= 0.55:
                return {
                    "classification": "OVERLAPPING",
                    "conflict_type": "Redundant / Duplicate Specification",
                    "confidence": round(0.70 + min(0.25, jaccard * 0.3), 2),
                    "explanation": (
                        f"High semantic and lexical overlap ({int(jaccard * 100)}% vocabulary overlap). "
                        f"Both requirements appear to define the same functional expectation with slight phrasing differences."
                    ),
                    "suggested_resolution": (
                        "Consolidate these into a single authoritative requirement statement in the Master SRS/BRD "
                        "and deprecate redundant versions to avoid drift."
                    )
                }

        # 5. Default to Related but Non-Conflicting
        return {
            "classification": "RELATED",
            "conflict_type": "Complementary Specification",
            "confidence": 0.80,
            "explanation": (
                "Both requirements relate to the same system domain or component, but their constraints "
                "are compatible and can be satisfied concurrently without contradiction."
            ),
            "suggested_resolution": "No conflict detected. Keep both specifications aligned during implementation."
        }

    def _check_numerical_conflict(self, text_a: str, text_b: str, doc_a: str, doc_b: str) -> Optional[Dict[str, Any]]:
        """Identifies conflicting numerical thresholds (timeouts, latency, retention, character lengths)."""
        metrics = [
            # Session timeouts
            (
                r'\b(session|sessions|inactivity|idle)\b',
                r'\b(timeout|expire|expires|expiry|logout|terminate|terminates|inactivity)\b',
                r'(\d+)\s*(minutes?|mins?|seconds?|secs?|hours?)',
                "Numerical / Session Timeout Inconsistency",
                "Align session security timeout with organizational compliance guidelines (e.g. NIST SP 800-63B)."
            ),
            # Response time / latency SLAs
            (
                r'\b(search|query|queries|response|latency|render|records|download|export)\b',
                r'\b(within|latency|response\s*time|sla|render|complete)\b',
                r'(\d+(?:\.\d+)?)\s*(seconds?|secs?|ms|milliseconds?)',
                "Numerical / Performance Response Time SLA Conflict",
                "Establish a unified Service Level Agreement (SLA) defining p95 and p99 response time limits."
            ),
            # Password length
            (
                r'\b(password|passwords|passphrase)\b',
                r'\b(characters?|length|minimum|least)\b',
                r'(?:at least|minimum of|min|contain)\s*(\d+)\s*(characters?)',
                "Numerical / Password Complexity Mismatch",
                "Standardize minimum password length policy across authentication modules."
            ),
            # Data retention
            (
                r'\b(retention|retain|retained|archived?|purge|purged|audit\s*logs?|trails?)\b',
                r'\b(period|stored?|retained?|preserve|preserved|archiving|purging|years?|days?|months?)\b',
                r'(\d+)\s*(days?|months?|years?)',
                "Numerical / Data Retention Policy Inconsistency",
                "Confirm statutory and regulatory data retention requirements with the legal/compliance officer."
            ),
        ]

        for topic_pat1, topic_pat2, num_pat, conflict_title, resolution in metrics:
            if (re.search(topic_pat1, text_a) and re.search(topic_pat2, text_a) and
                re.search(topic_pat1, text_b) and re.search(topic_pat2, text_b)):
                match_a = re.search(num_pat, text_a)
                match_b = re.search(num_pat, text_b)
                if match_a and match_b:
                    groups_a = match_a.groups()
                    groups_b = match_b.groups()
                    val_a = groups_a[0]
                    unit_a = groups_a[1] if len(groups_a) > 1 else ''
                    val_b = groups_b[0]
                    unit_b = groups_b[1] if len(groups_b) > 1 else ''
                    if val_a != val_b or unit_a != unit_b:
                        return {
                            "classification": "CONTRADICTORY",
                            "conflict_type": conflict_title,
                            "confidence": 0.95,
                            "explanation": (
                                f"Incompatible numeric values detected for {conflict_title.lower()}. "
                                f"'{doc_a}' specifies '{val_a} {unit_a}'.strip(), whereas '{doc_b}' specifies '{val_b} {unit_b}'.strip()."
                            ),
                            "suggested_resolution": resolution
                        }
        return None

    def _check_tech_incompatibilities(self, text_a: str, text_b: str, doc_a: str, doc_b: str) -> Optional[Dict[str, Any]]:
        """Identifies architectural, protocol, or technology contradictions."""
        clashes = [
            (
                r'\b(cloud|aws|azure|gcp|public cloud)\b',
                r'\b(on-premise|on prem|air-gapped|local datacenter|private infrastructure)\b',
                "Hosting / Deployment Architecture Contradiction",
                "One requirement mandates cloud infrastructure while the other mandates isolated on-premise deployment.",
                "Decide whether a hybrid-cloud architecture is permitted, or establish a definitive single hosting model."
            ),
            (
                r'\b(oauth\s*2\.?0?|openid\s*connect|jwt)\b',
                r'\b(saml\s*2\.?0?|ldap\s*only|kerberos)\b',
                "Identity & Authentication Protocol Divergence",
                "One specification dictates modern OAuth 2.0 / OIDC while the other prescribes enterprise SAML/LDAP.",
                "Clarify identity provider (IdP) integration standards or implement an identity gateway supporting federated auth."
            ),
            (
                r'\b(?:mandatory\s+)?(multi-factor|mfa|2fa|two-factor)\b',
                r'\b(single-factor|password\s+only|without\s+(?:requiring\s+)?(?:mfa|multi-factor)|bypass\s+mfa)\b',
                "Authentication Security Mandate Conflict",
                "One document requires mandatory MFA for all users, whereas the other allows single-factor password authentication.",
                "Enforce MFA across all external access and document any approved emergency break-glass exemptions."
            ),
            (
                r'\b(relational|postgresql|mysql|sql database)\b',
                r'\b(nosql|mongodb|dynamodb|document store)\b',
                "Database Technology Architecture Contradiction",
                "Conflicting database persistence tiers specified (Relational SQL vs NoSQL Document Store).",
                "Review data modeling requirements and select a primary persistence engine."
            ),
            (
                r'\b(english\s+only|single\s+language)\b',
                r'\b(multilingual|multiple\s+languages|i18n|internationalization|localization)\b',
                "Internationalization (i18n) Scope Contradiction",
                "One document scopes the application to English only, while the other requires multi-language localization.",
                "Confirm product internationalization roadmap with stakeholder steering committee."
            ),
        ]

        for pat_a, pat_b, title, expl, res in clashes:
            if (re.search(pat_a, text_a) and re.search(pat_b, text_b)) or \
               (re.search(pat_b, text_a) and re.search(pat_a, text_b)):
                return {
                    "classification": "CONTRADICTORY",
                    "conflict_type": title,
                    "confidence": 0.90,
                    "explanation": f"Architectural discrepancy: {expl}",
                    "suggested_resolution": res
                }
        return None


def get_llm_provider(provider_type: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory function to retrieve the configured LLM reasoning provider.
    Automatically and safely falls back to SafeMockLLMProvider if API keys are absent.
    """
    provider_name = (provider_type or getattr(settings, 'LLM_PROVIDER', 'mock')).lower()
    openai_key = getattr(settings, 'OPENAI_API_KEY', '')
    groq_key = getattr(settings, 'GROQ_API_KEY', '')

    if provider_name == 'openai':
        if openai_key and openai_key.strip():
            try:
                return OpenAILLMProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAILLMProvider: {e}. Falling back to SafeMockLLMProvider.")
                return SafeMockLLMProvider()
        else:
            logger.info("OPENAI_API_KEY is not set. Using SafeMockLLMProvider.")
            return SafeMockLLMProvider()

    elif provider_name == 'groq':
        if groq_key and groq_key.strip():
            try:
                return GroqLLMProvider()
            except Exception as e:
                logger.warning(f"Failed to initialize GroqLLMProvider: {e}. Falling back to SafeMockLLMProvider.")
                return SafeMockLLMProvider()
        else:
            logger.info("GROQ_API_KEY is not set. Using SafeMockLLMProvider.")
            return SafeMockLLMProvider()

    else:
        # Default is safe offline mock heuristic provider
        return SafeMockLLMProvider()
