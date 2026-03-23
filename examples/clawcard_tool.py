"""
clawcard_tool.py — Reusable Python wrapper around the `clawcard agent` CLI.

Drop this into any agent project (LangChain, LitServe, AutoGPT, etc.) to give
your agent a complete, isolated identity + payments stack without ever touching
raw credentials.

Usage:
    from clawcard_tool import ClawcardAgent

    agent = ClawcardAgent()
    info   = agent.info()                        # email, phone, wallet, budget
    emails = agent.emails(unread=True)           # check inbox
    card   = agent.cards_create(amount_cents=2000, card_type="single_use")
    creds  = agent.creds_set("stripe", "secret_key", "sk_live_...")
    bal    = agent.wallet_balance()
    pay    = agent.wallet_send("https://api.example.com/premium", protocol="x402")
"""

import json
import subprocess
from typing import Any, Optional


class ClawcardError(Exception):
    """Raised when a clawcard CLI call fails."""


def _run(args: list[str]) -> dict[str, Any]:
    """Run `clawcard agent <args> --json` and return parsed JSON."""
    cmd = ["clawcard", "agent"] + args + ["--json"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise ClawcardError(
            "clawcard CLI not found. Install with: npm install -g @clawcard/cli"
        )
    except subprocess.TimeoutExpired:
        raise ClawcardError(f"clawcard command timed out: {' '.join(cmd)}")

    if result.returncode != 0:
        raise ClawcardError(
            f"clawcard exited {result.returncode}: {result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ClawcardError(
            f"Invalid JSON from clawcard: {result.stdout[:200]}"
        ) from exc


class ClawcardAgent:
    """
    Thin Python wrapper around the `clawcard agent` sub-commands.

    Every method maps 1-to-1 to a CLI command and returns parsed JSON.
    The agent's API key is stored in ~/.clawcard/.env — it is never
    exposed in code or process arguments.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """
        Return agent identity: email, phone, wallet, budget, ERC-8004 NFT URL.

        CLI: clawcard agent info --json
        """
        return _run(["info"])

    def activity(self, limit: int = 20) -> list[dict[str, Any]]:
        """
        Fetch audit log of recent agent activity.

        CLI: clawcard agent activity --limit N --json
        """
        return _run(["activity", "--limit", str(limit)])

    # ------------------------------------------------------------------
    # Email inbox
    # ------------------------------------------------------------------

    def emails(
        self,
        unread: bool = False,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        List inbox messages, optionally filtered to unread only.

        CLI: clawcard agent emails [--unread] --limit N --json
        """
        args = ["emails", "--limit", str(limit)]
        if unread:
            args.append("--unread")
        return _run(args)

    def email_get(self, email_id: str) -> dict[str, Any]:
        """
        Fetch a single email by ID (includes full body + links).

        CLI: clawcard agent emails get <id> --json
        """
        return _run(["emails", "get", email_id])

    # ------------------------------------------------------------------
    # SMS / phone
    # ------------------------------------------------------------------

    def sms(self, limit: int = 5) -> list[dict[str, Any]]:
        """
        List recent SMS messages (useful for 2FA codes).

        CLI: clawcard agent sms --limit N --json
        """
        return _run(["sms", "--limit", str(limit)])

    # ------------------------------------------------------------------
    # Virtual cards
    # ------------------------------------------------------------------

    def cards_create(
        self,
        amount_cents: int,
        card_type: str = "single_use",
        memo: Optional[str] = None,
        merchant: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a virtual Mastercard.

        Args:
            amount_cents: Spending limit in US cents (e.g. 2000 = $20).
            card_type: "single_use" or "merchant_locked".
            memo: Human-readable label (e.g. "Stripe subscription").
            merchant: Merchant lock domain for merchant_locked cards.

        CLI: clawcard agent cards create --amount N --type T [--memo M] --json
        """
        args = ["cards", "create", "--amount", str(amount_cents), "--type", card_type]
        if memo:
            args += ["--memo", memo]
        if merchant and card_type == "merchant_locked":
            args += ["--merchant", merchant]
        return _run(args)

    def cards_details(self, card_id: str) -> dict[str, Any]:
        """
        Retrieve PAN, CVV, and expiry for a card (one-time reveal).

        CLI: clawcard agent cards details <id> --json
        """
        return _run(["cards", "details", card_id])

    def cards_list(self) -> list[dict[str, Any]]:
        """
        List all virtual cards and their status.

        CLI: clawcard agent cards list --json
        """
        return _run(["cards", "list"])

    def cards_freeze(self, card_id: str) -> dict[str, Any]:
        """
        Immediately freeze a card (stops all future charges).

        CLI: clawcard agent cards freeze <id> --json
        """
        return _run(["cards", "freeze", card_id])

    # ------------------------------------------------------------------
    # Credential vault
    # ------------------------------------------------------------------

    def creds_set(self, service: str, key: str, value: str) -> dict[str, Any]:
        """
        Store an AES-256-encrypted credential in the vault.

        CLI: clawcard agent creds set --service S --key K --value V --json
        """
        return _run(["creds", "set", "--service", service, "--key", key, "--value", value])

    def creds_get(self, service: str, key: str) -> str:
        """
        Retrieve a credential from the vault (returns plaintext value).

        CLI: clawcard agent creds get --service S --key K --json
        """
        result = _run(["creds", "get", "--service", service, "--key", key])
        return result.get("value", "")

    def creds_list(self, service: Optional[str] = None) -> list[dict[str, Any]]:
        """
        List stored credential keys (values are never returned in bulk).

        CLI: clawcard agent creds list [--service S] --json
        """
        args = ["creds", "list"]
        if service:
            args += ["--service", service]
        return _run(args)

    # ------------------------------------------------------------------
    # Wallet (USDC on Base + pathUSD on Tempo)
    # ------------------------------------------------------------------

    def wallet_balance(self) -> dict[str, Any]:
        """
        Return wallet balance on Base (USDC) and Tempo (pathUSD).

        CLI: clawcard agent wallet balance --json
        """
        return _run(["wallet", "balance"])

    def wallet_fund(self) -> dict[str, Any]:
        """
        Initiate a wallet top-up flow (returns deposit address).

        CLI: clawcard agent wallet fund --json
        """
        return _run(["wallet", "fund"])

    def wallet_send(
        self,
        url: str,
        protocol: str = "x402",
        amount_usd: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Pay a 402-gated API endpoint gaslessly.

        Args:
            url: The endpoint URL that returns HTTP 402.
            protocol: "x402" (Coinbase/Base) or "mpp" (Tempo/Stripe).
            amount_usd: Optional override; omit to use the API's stated price.

        CLI: clawcard agent wallet send --url U --protocol P [--amount A] --json
        """
        args = ["wallet", "send", "--url", url, "--protocol", protocol]
        if amount_usd is not None:
            args += ["--amount", str(amount_usd)]
        return _run(args)

    def wallet_freeze(self) -> dict[str, Any]:
        """
        Emergency freeze of all outbound wallet transactions.

        CLI: clawcard agent wallet freeze --json
        """
        return _run(["wallet", "freeze"])


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python examples/clawcard_tool.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    cw = ClawcardAgent()

    print("=== Agent Info ===")
    try:
        info = cw.info()
        print(json.dumps(info, indent=2))
    except ClawcardError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Wallet Balance ===")
    try:
        bal = cw.wallet_balance()
        print(json.dumps(bal, indent=2))
    except ClawcardError as exc:
        print(f"[error] {exc}", file=sys.stderr)
