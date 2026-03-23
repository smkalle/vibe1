"""
clawcard_langchain_agent.py — Full autonomous LangChain agent with Clawcard identity.

This example wires all Clawcard capabilities into LangChain tools so an LLM can:
  - Sign up for external services using its provisioned email / phone
  - Create virtual cards and complete checkout flows
  - Store and retrieve secrets from the encrypted vault
  - Pay 402-gated APIs gaslessly via x402 or MPP
  - Read and act on incoming verification emails / SMS codes

Prerequisites:
    pip install langchain langchain-openai
    npm install -g @clawcard/cli
    clawcard signup          # one-time — enter invite code
    clawcard billing topup   # fund your balance ($5+ minimum)
    clawcard keys create --name "langchain-agent"

Environment variables:
    OPENAI_API_KEY   — OpenAI key for the LLM backbone
    # No CLAWCARD key needed — CLI reads from ~/.clawcard/.env

Run:
    python examples/clawcard_langchain_agent.py
    # or supply a task via stdin:
    echo "Sign me up for a free Mailchimp account and report back." | \
      python examples/clawcard_langchain_agent.py
"""

import os
import sys
import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from clawcard_tool import ClawcardAgent, ClawcardError

# ---------------------------------------------------------------------------
# Shared Clawcard agent instance (one per process)
# ---------------------------------------------------------------------------
_cw = ClawcardAgent()


# ---------------------------------------------------------------------------
# LangChain tools — each wraps a ClawcardAgent method
# ---------------------------------------------------------------------------

@tool
def get_agent_identity() -> str:
    """
    Return the agent's provisioned identity: email address, phone number,
    wallet address, remaining card budget (in cents), and on-chain ERC-8004
    NFT URL.  Call this first to learn which email/phone to use for sign-ups.
    """
    try:
        return json.dumps(_cw.info(), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def check_inbox(unread_only: bool = True, limit: int = 10) -> str:
    """
    List emails in the agent's Clawcard inbox.  Use unread_only=True (default)
    to see only new messages.  Returns a JSON list with sender, subject, preview,
    and message ID.  Call get_email_body(id) to read a full email.
    """
    try:
        emails = _cw.emails(unread=unread_only, limit=limit)
        return json.dumps(emails, indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def get_email_body(email_id: str) -> str:
    """
    Fetch the full body and all links from an email given its ID.
    Use this to retrieve verification links, activation URLs, or
    any content from a sign-up confirmation email.
    """
    try:
        return json.dumps(_cw.email_get(email_id), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def get_sms_messages(limit: int = 5) -> str:
    """
    Return the most recent SMS messages (including 2FA / OTP codes) received
    on the agent's provisioned phone number.
    """
    try:
        return json.dumps(_cw.sms(limit=limit), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def create_virtual_card(
    amount_cents: int,
    card_type: str = "single_use",
    memo: str = "",
    merchant: str = "",
) -> str:
    """
    Create a virtual Mastercard for checkout.

    Args:
        amount_cents: Spending limit in US cents (e.g. 2000 = $20.00).
        card_type: "single_use" for one-time purchases; "merchant_locked" for
                   subscriptions/recurring charges at a specific merchant.
        memo: Human-readable label to identify the card later.
        merchant: Required for merchant_locked cards — the merchant domain
                  (e.g. "stripe.com").

    Returns card ID.  Use get_card_details(card_id) once to retrieve PAN/CVV.
    """
    try:
        result = _cw.cards_create(
            amount_cents=amount_cents,
            card_type=card_type,
            memo=memo or None,
            merchant=merchant or None,
        )
        return json.dumps(result, indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def get_card_details(card_id: str) -> str:
    """
    Retrieve the PAN (card number), CVV, and expiry for a virtual card.
    WARNING: This is a one-time reveal — store the details immediately in
    the vault using store_credential if you need them again.
    """
    try:
        return json.dumps(_cw.cards_details(card_id), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def store_credential(service: str, key: str, value: str) -> str:
    """
    Store a secret (API key, password, card number, etc.) in the AES-256
    encrypted Clawcard vault.  Credentials persist across agent restarts and
    are scoped to this agent's key.

    Args:
        service: Logical service name, e.g. "stripe", "openai", "github".
        key: Credential key name, e.g. "api_key", "password", "card_pan".
        value: The plaintext secret to encrypt and store.
    """
    try:
        return json.dumps(_cw.creds_set(service, key, value), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def retrieve_credential(service: str, key: str) -> str:
    """
    Retrieve a previously stored credential from the encrypted vault.

    Args:
        service: Logical service name (must match what was used in store_credential).
        key: Credential key name.

    Returns the plaintext value.
    """
    try:
        value = _cw.creds_get(service, key)
        return value if value else "Not found"
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def list_credentials(service: str = "") -> str:
    """
    List credential keys stored in the vault.  Optionally filter by service name.
    Values are never returned in bulk — use retrieve_credential to read one.
    """
    try:
        return json.dumps(_cw.creds_list(service or None), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def check_wallet_balance() -> str:
    """
    Return USDC balance on Base and pathUSD balance on Tempo for this agent's
    wallet.  Both are stablecoins (1:1 USD peg).
    """
    try:
        return json.dumps(_cw.wallet_balance(), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def pay_api_endpoint(url: str, protocol: str = "x402", amount_usd: float = 0.0) -> str:
    """
    Pay a 402-gated API endpoint gaslessly using the agent's wallet.

    The agent wallet handles the entire payment flow:
      1. Sends request to the URL
      2. Receives HTTP 402 with payment details
      3. Signs and broadcasts the payment (USDC on Base or pathUSD on Tempo)
      4. Retries the original request with payment proof

    Args:
        url: The API endpoint that returns HTTP 402.
        protocol: "x402" for Coinbase/Base payments, "mpp" for Tempo/Stripe.
        amount_usd: Override payment amount in USD; omit to use stated price.

    Returns the API response after successful payment.
    """
    try:
        kwargs = {"url": url, "protocol": protocol}
        if amount_usd > 0:
            kwargs["amount_usd"] = amount_usd
        return json.dumps(_cw.wallet_send(**kwargs), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def freeze_wallet() -> str:
    """
    Emergency freeze of all outbound wallet transactions.  Use if the agent
    is behaving unexpectedly or if you suspect a compromise.
    """
    try:
        return json.dumps(_cw.wallet_freeze(), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


@tool
def get_activity_log(limit: int = 20) -> str:
    """
    Fetch the last N entries from the agent's activity audit log.
    Includes all email reads, card creations, payments, and vault operations.
    """
    try:
        return json.dumps(_cw.activity(limit=limit), indent=2)
    except ClawcardError as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Agent assembly
# ---------------------------------------------------------------------------

CLAWCARD_TOOLS = [
    get_agent_identity,
    check_inbox,
    get_email_body,
    get_sms_messages,
    create_virtual_card,
    get_card_details,
    store_credential,
    retrieve_credential,
    list_credentials,
    check_wallet_balance,
    pay_api_endpoint,
    freeze_wallet,
    get_activity_log,
]

SYSTEM_PROMPT = """You are a fully autonomous AI agent with a complete, isolated identity
provided by Clawcard.sh.  You have:

  - A dedicated email inbox for sign-ups and verifications
  - A real US phone number for SMS / 2FA
  - On-demand virtual Mastercards (single-use or merchant-locked)
  - A stablecoin wallet (USDC on Base, pathUSD on Tempo)
  - Gasless API payments via x402 and MPP protocols
  - An AES-256 encrypted credential vault
  - An ERC-8004 on-chain identity NFT on Base

## Operating rules

1. ALWAYS start by calling get_agent_identity() to know your email and phone.
2. When signing up for services, use your Clawcard email — NEVER ask the human for theirs.
3. After submitting a sign-up form, poll check_inbox() until the verification email arrives,
   then call get_email_body(id) to extract and act on the verification link.
4. For SMS 2FA, call get_sms_messages() after triggering the SMS.
5. Create single-use virtual cards for one-off purchases; use merchant_locked for subscriptions.
6. ALWAYS store credentials (API keys, passwords) via store_credential() immediately after
   obtaining them — do not rely on conversational memory across sessions.
7. For paid APIs that return HTTP 402, use pay_api_endpoint() — do not ask the human to pay.
8. Report the outcome clearly: what service was created, what credentials were stored,
   and what the agent's current wallet / budget balance is.
9. If any operation fails, report the exact error and suggest a remediation step.
"""


def build_agent(model: str = "gpt-4o", verbose: bool = True) -> AgentExecutor:
    """Construct and return a LangChain AgentExecutor with all Clawcard tools."""
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, CLAWCARD_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=CLAWCARD_TOOLS, verbose=verbose)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_TASK = (
    "Check my agent identity and wallet balance, then report a summary of "
    "what this agent can do and what resources it currently has."
)


def main():
    task = sys.stdin.read().strip() if not sys.stdin.isatty() else DEFAULT_TASK

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Export it before running:\n"
            "  export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n[Task] {task}\n{'─' * 60}")
    executor = build_agent()
    result = executor.invoke({"input": task})
    print(f"\n{'─' * 60}\n[Result]\n{result['output']}")


if __name__ == "__main__":
    main()
