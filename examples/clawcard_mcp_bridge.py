"""
clawcard_mcp_bridge.py — MCP bridge that exposes Clawcard agent capabilities
as native Claude Desktop / Claude Code tools via the Model Context Protocol.

Add this to your claude_desktop_config.json and every Claude session gets
live access to your agent's identity, inbox, cards, vault, and wallet — no
separate backend needed.

Usage (stdio transport — Claude Desktop / Claude Code):
    python examples/clawcard_mcp_bridge.py

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "clawcard": {
          "command": "python",
          "args": ["/absolute/path/to/examples/clawcard_mcp_bridge.py"]
        }
      }
    }

After adding, restart Claude Desktop.  Say "Check my Clawcard inbox" and Claude
will call the mcp tool directly — no manual copy-paste required.
"""

import json
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from clawcard_tool import ClawcardAgent, ClawcardError

mcp = FastMCP("clawcard-agent")
_cw = ClawcardAgent()


def _ok(data: Any) -> str:
    return json.dumps(data, indent=2)


def _err(exc: Exception) -> str:
    return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@mcp.tool()
def clawcard_info() -> str:
    """
    Return the agent's provisioned identity: email, phone number, wallet
    address, card budget (cents), and ERC-8004 on-chain NFT URL.
    """
    try:
        return _ok(_cw.info())
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_activity(limit: int = 20) -> str:
    """
    Fetch the last `limit` entries from the agent's full audit log.
    Covers emails read, cards created, payments made, and vault ops.
    """
    try:
        return _ok(_cw.activity(limit=limit))
    except ClawcardError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

@mcp.tool()
def clawcard_emails(unread: bool = True, limit: int = 10) -> str:
    """
    List emails in the agent's Clawcard inbox.
    Set unread=True (default) to filter to unread messages only.
    Returns sender, subject, snippet, and message ID.
    """
    try:
        return _ok(_cw.emails(unread=unread, limit=limit))
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_email_get(email_id: str) -> str:
    """
    Fetch the full body and all hyperlinks from an email by ID.
    Use this after clawcard_emails() to read verification links.
    """
    try:
        return _ok(_cw.email_get(email_id))
    except ClawcardError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# SMS
# ---------------------------------------------------------------------------

@mcp.tool()
def clawcard_sms(limit: int = 5) -> str:
    """
    Return the most recent SMS messages on the agent's phone, including
    2FA / OTP codes from external services.
    """
    try:
        return _ok(_cw.sms(limit=limit))
    except ClawcardError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Virtual cards
# ---------------------------------------------------------------------------

@mcp.tool()
def clawcard_card_create(
    amount_cents: int,
    card_type: str = "single_use",
    memo: str = "",
    merchant: str = "",
) -> str:
    """
    Create a virtual Mastercard.

    amount_cents: spending limit in US cents (2000 = $20).
    card_type: "single_use" (one purchase) or "merchant_locked" (recurring).
    memo: label for your own reference.
    merchant: required for merchant_locked — e.g. "aws.amazon.com".

    Returns the card ID.  Call clawcard_card_details(card_id) once to get PAN/CVV.
    """
    try:
        return _ok(_cw.cards_create(
            amount_cents=amount_cents,
            card_type=card_type,
            memo=memo or None,
            merchant=merchant or None,
        ))
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_card_details(card_id: str) -> str:
    """
    One-time reveal of PAN, CVV, and expiry for a virtual card.
    Store the result in the vault immediately via clawcard_creds_set().
    """
    try:
        return _ok(_cw.cards_details(card_id))
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_cards_list() -> str:
    """List all virtual cards and their current status / spend."""
    try:
        return _ok(_cw.cards_list())
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_card_freeze(card_id: str) -> str:
    """Immediately freeze a card to prevent further charges."""
    try:
        return _ok(_cw.cards_freeze(card_id))
    except ClawcardError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Credential vault
# ---------------------------------------------------------------------------

@mcp.tool()
def clawcard_creds_set(service: str, key: str, value: str) -> str:
    """
    Store a secret in the AES-256 encrypted vault.

    service: logical name, e.g. "stripe" or "github".
    key: credential field, e.g. "api_key" or "password".
    value: plaintext secret to encrypt and store.
    """
    try:
        return _ok(_cw.creds_set(service, key, value))
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_creds_get(service: str, key: str) -> str:
    """
    Retrieve a plaintext credential from the encrypted vault.

    service: must match what was used in clawcard_creds_set.
    key: credential field name.
    """
    try:
        value = _cw.creds_get(service, key)
        return _ok({"value": value})
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_creds_list(service: str = "") -> str:
    """
    List stored credential keys (values are never returned in bulk).
    Filter by service name or omit to list all.
    """
    try:
        return _ok(_cw.creds_list(service or None))
    except ClawcardError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Wallet
# ---------------------------------------------------------------------------

@mcp.tool()
def clawcard_wallet_balance() -> str:
    """Return USDC (Base) and pathUSD (Tempo) wallet balances."""
    try:
        return _ok(_cw.wallet_balance())
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_wallet_send(
    url: str,
    protocol: str = "x402",
    amount_usd: float = 0.0,
) -> str:
    """
    Pay a 402-gated API endpoint gaslessly from the agent wallet.

    url: endpoint that returns HTTP 402.
    protocol: "x402" (Coinbase/Base USDC) or "mpp" (Tempo/Stripe pathUSD).
    amount_usd: override stated price; leave 0.0 to use the API's price.

    Returns the full API response after the payment is confirmed.
    """
    try:
        return _ok(_cw.wallet_send(
            url=url,
            protocol=protocol,
            amount_usd=amount_usd if amount_usd > 0 else None,
        ))
    except ClawcardError as exc:
        return _err(exc)


@mcp.tool()
def clawcard_wallet_freeze() -> str:
    """Emergency freeze of all outbound wallet transactions."""
    try:
        return _ok(_cw.wallet_freeze())
    except ClawcardError as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
