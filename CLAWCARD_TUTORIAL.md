# Clawcard Tutorial: Equipping AI Agents with Full Autonomous Identity, Payments, and Infrastructure

> **Perspective:** Senior Software Engineer (SDE3) on Google's AI infrastructure team
> **Stack:** LitServe · LangChain · MCP · Clawcard.sh
> **Time to first autonomous agent:** ~15 minutes

---

## Why This Matters

After years building production agent systems at Google-scale I kept hitting the same wall: agents can reason and plan, but the moment they need to **sign up for a service, verify an email, pay for an API, or remember a credential across restarts**, a human has to step in.

That friction compounds.  An agent pipeline that fires 20 sign-up flows a day blocks 20 engineers.  The "10x autonomous agent" everyone talks about is actually a 1.2x agent with 8.8x human babysitting tax.

**[Clawcard.sh](https://clawcard.sh)** removes this tax with one command:

```
npx @clawcard/cli keys create --name my-agent
```

You get a complete, isolated identity stack — email, phone, virtual Mastercards, stablecoin wallet with x402/MPP gasless payments, and an encrypted credential vault.  The agent signs up.  The agent pays.  The agent stores the credentials.  You get a Slack ping when it's done.

This tutorial integrates Clawcard with the **litserve-mcp-starter** patterns you already know: LitServe APIs, MCP bridges, and LangChain agents.

---

## What Clawcard Provisions (Per Agent)

| Resource | Details |
|---|---|
| Email | `inbox-xxx@mail.clawcard.sh` — real inbox, full MIME |
| Phone | Real US number — receives SMS / 2FA codes |
| Virtual Mastercards | Single-use or merchant-locked, any spend limit |
| Wallet | USDC on Base + auto-bridged pathUSD on Tempo |
| Gasless payments | x402 (Coinbase standard) + MPP (Tempo/Stripe) |
| Credential vault | AES-256 encrypted, scoped to agent key |
| On-chain identity | ERC-8004 NFT on Base — portable reputation |
| Audit ledger | Every action logged, queryable via CLI |

**Pricing:** Free tier covers email, phone, wallet, on-chain ID, and x402/MPP payments.  Virtual cards are the only paid add-on (10% top-up fee, min $5).  No subscriptions.

---

## Prerequisites

- Node.js 18+ (`node --version`)
- Python 3.11+ with this repo's virtualenv activated
- An invite code — request via [clawcard.sh](https://clawcard.sh) or DM [@cblovescode](https://x.com/cblovescode) on X
- $5+ USD for initial top-up (Coinbase card flow — no crypto knowledge required)
- `OPENAI_API_KEY` for LangChain / LitServe examples (OpenAI or compatible)

---

## Step 1: Install the CLI and Sign Up (5 minutes)

```bash
# Install the Clawcard CLI globally
npm install -g @clawcard/cli

# Verify
clawcard --version
```

### Create your account

```bash
clawcard signup
# Enter invite code when prompted
# → Account created. First agent identity provisioned.
```

### Fund your balance

Cards and wallet top-ups draw from your Clawcard balance.

```bash
clawcard billing topup
# → Opens Coinbase card purchase flow
# → Minimum $5 USD.  Your balance is shown in cents.
```

Verify after funding:

```bash
clawcard billing balance
# → { "balanceCents": 500, "currency": "USD" }
```

> **SDE3 note:** The CLI stores your session in `~/.clawcard/.env`.  Add this path to your global `.gitignore` and never commit it.  The file is loaded automatically — agent code never sees the root API key.

---

## Step 2: Create an Isolated Agent Identity

Each agent gets its own key, email, phone, and wallet.  Revoke one without touching others.

```bash
clawcard keys create --name "litserve-tutorial-agent"
```

**Output (shown once — save it now):**

```json
{
  "name": "litserve-tutorial-agent",
  "apiKey": "ak_EXAMPLE_replace_with_real_key",
  "email": "inbox-7xk@mail.clawcard.sh",
  "phone": "+12025551234",
  "walletBase": "0xabc...def",
  "walletTempo": "0x123...456",
  "erc8004Nft": "https://basescan.org/nft/..."
}
```

Store the raw API key immediately:

```bash
# Option A — Google Secret Manager (production)
gcloud secrets create clawcard-agent-key --data-file=<(echo "ak_EXAMPLE_replace_with_real_key")

# Option B — local .env (development, gitignored)
echo "CLAWCARD_AGENT_KEY=ak_EXAMPLE_replace_with_real_key" >> .env
```

Optionally pre-allocate a card budget:

```bash
clawcard agent fund --amount 5000   # $50.00 in cents
```

---

## Step 3: Install the Agent Skill

```bash
clawcard setup
```

This uses [skills.sh](https://skills.sh) to teach every `clawcard agent` sub-command to your agent framework.  After this step:

- **Claude Code / Cursor:** The skill auto-registers.  Say *"Use Clawcard to sign up for Vercel."*
- **Claude Desktop:** Add the MCP bridge (Step 6 below) to your config.
- **LangChain / LangGraph / CrewAI:** Use the tool wrappers in Step 5.
- **Any shell-capable agent:** Call `clawcard agent <sub-command> --json` directly.

---

## Step 4: The `clawcard agent` Command Reference

All agent operations go through `clawcard agent`.  Always append `--json` so your agent can parse structured output.

### Identity

```bash
clawcard agent info --json
```

```json
{
  "email": "inbox-7xk@mail.clawcard.sh",
  "phone": "+12025551234",
  "wallet": "0xabc...def",
  "budgetCents": 5000,
  "erc8004Nft": "https://basescan.org/nft/..."
}
```

### Email inbox

```bash
# List unread emails
clawcard agent emails --unread --json

# Read a specific email (full body + links)
clawcard agent emails get <email_id> --json
```

### SMS / 2FA

```bash
clawcard agent sms --limit 5 --json
# → [{ "from": "+12025550100", "body": "Your OTP is 847293", "ts": "..." }]
```

### Virtual cards

```bash
# Create a $20 single-use card
clawcard agent cards create --amount 2000 --type single_use --memo "Stripe trial" --json

# One-time PAN/CVV reveal (store immediately!)
clawcard agent cards details <card_id> --json

# List all cards
clawcard agent cards list --json

# Freeze a card instantly
clawcard agent cards freeze <card_id> --json
```

### Credential vault

```bash
# Store (AES-256 encrypted)
clawcard agent creds set --service stripe --key secret_key --value "<your-stripe-key>" --json

# Retrieve
clawcard agent creds get --service stripe --key secret_key --json

# List keys (values never returned in bulk)
clawcard agent creds list --service stripe --json
```

### Wallet & gasless payments

```bash
# Check balance (USDC on Base + pathUSD on Tempo)
clawcard agent wallet balance --json

# Pay a 402-gated API (auto-bridges between chains)
clawcard agent wallet send --url "https://api.example.com/premium" --protocol x402 --json
clawcard agent wallet send --url "https://api.tempo.io/v1/complete" --protocol mpp --json

# Emergency freeze
clawcard agent wallet freeze --json
```

### Audit log

```bash
clawcard agent activity --json
```

---

## Step 5: Python Integration Patterns

The repo ships three integration layers in `examples/`.  Pick the one that fits your stack.

### Pattern A — `ClawcardAgent` wrapper class (`examples/clawcard_tool.py`)

A thin Python wrapper around `subprocess` that maps every CLI command to a typed method.  Works with any framework.

```python
from examples.clawcard_tool import ClawcardAgent

cw = ClawcardAgent()

# Identity
info = cw.info()
print(info["email"])   # inbox-7xk@mail.clawcard.sh

# Full signup flow
emails = cw.emails(unread=True)
email  = cw.email_get(emails[0]["id"])
# → email["links"] contains the verification URL

# One-time virtual card
card    = cw.cards_create(amount_cents=2000, memo="Stripe trial")
details = cw.cards_details(card["id"])
# → details: { "pan": "4111...", "cvv": "...", "expiry": "12/27" }
cw.creds_set("stripe", "card_pan", details["pan"])

# Gasless payment
result = cw.wallet_send("https://api.example.com/premium", protocol="x402")
```

**Full class API:**

| Method | CLI equivalent |
|---|---|
| `info()` | `agent info` |
| `emails(unread, limit)` | `agent emails [--unread]` |
| `email_get(id)` | `agent emails get <id>` |
| `sms(limit)` | `agent sms` |
| `cards_create(cents, type, memo, merchant)` | `agent cards create` |
| `cards_details(id)` | `agent cards details <id>` |
| `cards_list()` | `agent cards list` |
| `cards_freeze(id)` | `agent cards freeze <id>` |
| `creds_set(service, key, value)` | `agent creds set` |
| `creds_get(service, key)` | `agent creds get` |
| `creds_list(service?)` | `agent creds list` |
| `wallet_balance()` | `agent wallet balance` |
| `wallet_send(url, protocol, amount_usd?)` | `agent wallet send` |
| `wallet_freeze()` | `agent wallet freeze` |
| `activity(limit)` | `agent activity` |

---

### Pattern B — LitServe HTTP server (`examples/server_clawcard_agent.py`)

Wraps the entire Clawcard API surface as a LitServe endpoint on port 8003.  Drop-in for any HTTP-capable agent.

```bash
python examples/server_clawcard_agent.py
```

All operations via `POST /predict`:

```bash
# Identity
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "info"}'

# Check unread email
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "emails", "unread": true}'

# Create $20 single-use card
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "card_create", "amount_cents": 2000, "memo": "AWS trial"}'

# Store credential
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "creds_set", "service": "aws", "key": "access_key_id", "value": "AKIA..."}'

# Gasless x402 payment
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "wallet_send", "url": "https://api.example.com/premium", "protocol": "x402"}'
```

**Available `action` values:**

```
info | emails | email_get | sms
card_create | card_details | cards_list | card_freeze
creds_set | creds_get | creds_list
wallet_balance | wallet_send | wallet_freeze
activity
```

---

### Pattern C — LangChain autonomous agent (`examples/clawcard_langchain_agent.py`)

The most powerful option: a GPT-4o LangChain agent with all Clawcard capabilities as native tools.  The agent reasons about which tools to call and in what order.

```bash
export OPENAI_API_KEY=sk-...
python examples/clawcard_langchain_agent.py
```

Or pipe a task:

```bash
echo "Sign me up for a free Mailchimp account, verify the email, and store the credentials." | \
  python examples/clawcard_langchain_agent.py
```

**What the agent does automatically:**

1. Calls `get_agent_identity()` — learns its email and phone
2. Navigates to the service (via browser tool or Playwright)
3. Submits the sign-up form with the Clawcard email
4. Calls `check_inbox()` in a loop until the verification email arrives
5. Calls `get_email_body(id)` — extracts the confirmation link
6. Navigates to the link — account verified
7. Calls `store_credential(service, key, value)` — saves login details
8. Reports outcome with wallet/budget status

**Available LangChain tools:**

| Tool | Description |
|---|---|
| `get_agent_identity` | Email, phone, wallet, budget, NFT URL |
| `check_inbox` | List inbox (unread filter) |
| `get_email_body` | Full email body + links by ID |
| `get_sms_messages` | Recent SMS / OTP codes |
| `create_virtual_card` | New Mastercard (single_use / merchant_locked) |
| `get_card_details` | One-time PAN/CVV reveal |
| `store_credential` | AES-256 vault write |
| `retrieve_credential` | Vault read |
| `list_credentials` | Vault key listing |
| `check_wallet_balance` | USDC + pathUSD balances |
| `pay_api_endpoint` | x402 / MPP gasless payment |
| `freeze_wallet` | Emergency freeze |
| `get_activity_log` | Audit log |

**Embedding in your own LangChain setup:**

```python
from examples.clawcard_langchain_agent import CLAWCARD_TOOLS, build_agent

# Mix with your own tools
from langchain.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """My domain-specific tool."""
    return "result"

executor = build_agent()  # uses gpt-4o by default
# or:
from langchain.agents import AgentExecutor, create_openai_tools_agent
all_tools = CLAWCARD_TOOLS + [my_custom_tool]
```

---

## Step 6: Claude Desktop / Claude Code — MCP Bridge

The MCP bridge (`examples/clawcard_mcp_bridge.py`) exposes all Clawcard tools natively inside Claude Desktop and Claude Code sessions.  No Python code needed from the user — just natural language.

### Add to Claude Desktop config

Edit `~/.config/claude/claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "clawcard": {
      "command": "python",
      "args": ["/absolute/path/to/examples/clawcard_mcp_bridge.py"]
    }
  }
}
```

Restart Claude Desktop.  You can now say:

- *"Check my Clawcard inbox for unread messages."*
- *"Create a $30 single-use virtual card for a Stripe trial."*
- *"Store my GitHub token in the Clawcard vault."*
- *"Pay the x402 API at https://api.example.com/premium with my wallet."*

### Run as standalone MCP server (Claude Code)

```bash
python examples/clawcard_mcp_bridge.py
```

**Exposed MCP tools:**

| Tool name | Description |
|---|---|
| `clawcard_info` | Agent identity |
| `clawcard_activity` | Audit log |
| `clawcard_emails` | Inbox listing |
| `clawcard_email_get` | Full email body |
| `clawcard_sms` | SMS / OTP messages |
| `clawcard_card_create` | Create virtual card |
| `clawcard_card_details` | One-time PAN/CVV |
| `clawcard_cards_list` | List all cards |
| `clawcard_card_freeze` | Freeze a card |
| `clawcard_creds_set` | Store encrypted secret |
| `clawcard_creds_get` | Retrieve secret |
| `clawcard_creds_list` | List vault keys |
| `clawcard_wallet_balance` | USDC + pathUSD |
| `clawcard_wallet_send` | x402 / MPP payment |
| `clawcard_wallet_freeze` | Emergency wallet freeze |

---

## Step 7: Full End-to-End Walkthrough

Let's walk through a complete autonomous flow: **sign up for a paid API, pay for it, and store the credentials.**

### 7.1 Start the LitServe Clawcard server

```bash
# Terminal 1
python examples/server_clawcard_agent.py
# → LitServe running on http://0.0.0.0:8003
```

### 7.2 Confirm identity

```bash
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "info"}' | python -m json.tool
```

```json
{
  "email": "inbox-7xk@mail.clawcard.sh",
  "phone": "+12025551234",
  "wallet": "0xabc...def",
  "budgetCents": 5000,
  "erc8004Nft": "https://basescan.org/nft/..."
}
```

### 7.3 Sign up for a service

The agent fills the sign-up form with `inbox-7xk@mail.clawcard.sh`.  Then it polls:

```bash
# Poll until the verification email arrives
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "emails", "unread": true}' | python -m json.tool
```

```json
{
  "emails": [
    {
      "id": "msg_abc123",
      "from": "noreply@someservice.com",
      "subject": "Confirm your email",
      "snippet": "Click here to verify your account...",
      "ts": "2026-03-23T14:01:00Z"
    }
  ]
}
```

### 7.4 Extract the verification link

```bash
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "email_get", "email_id": "msg_abc123"}' | python -m json.tool
```

```json
{
  "id": "msg_abc123",
  "body": "Click here to verify...",
  "links": [
    "https://someservice.com/verify?token=abc123xyz"
  ]
}
```

The agent navigates to that URL.  Account confirmed.

### 7.5 Create a virtual card and pay

If the service requires a credit card:

```bash
# Create single-use card ($25 limit)
CARD=$(curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "card_create", "amount_cents": 2500, "memo": "SomeService signup"}')

CARD_ID=$(echo $CARD | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

# Get PAN/CVV (one-time reveal)
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d "{\"action\": \"card_details\", \"card_id\": \"$CARD_ID\"}" | python -m json.tool
```

```json
{
  "id": "card_xyz",
  "pan": "5204 7300 0000 0000",
  "cvv": "737",
  "expiry": "12/27",
  "billingZip": "94016"
}
```

For **402-gated APIs** (x402/MPP), skip the card entirely:

```bash
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "wallet_send", "url": "https://api.someservice.com/v1/complete", "protocol": "x402"}'
```

The wallet handles the full payment handshake transparently.

### 7.6 Store credentials in the vault

```bash
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "creds_set", "service": "someservice", "key": "api_key", "value": "<your-api-key>"}'

curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "creds_set", "service": "someservice", "key": "account_email", "value": "inbox-7xk@mail.clawcard.sh"}'
```

On the next agent restart:

```bash
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "creds_get", "service": "someservice", "key": "api_key"}'
# → { "value": "<your-api-key>" }
```

### 7.7 Audit everything

```bash
curl -s -X POST localhost:8003/predict \
  -H "content-type: application/json" \
  -d '{"action": "activity", "limit": 10}' | python -m json.tool
```

---

## Step 8: Direct REST API (Non-CLI Agents)

For agents that cannot shell out, use the REST API directly with a Bearer token:

```bash
# Identity
curl -H "Authorization: Bearer $CLAWCARD_AGENT_KEY" \
  https://clawcard.sh/api/agents

# Inbox
curl -H "Authorization: Bearer $CLAWCARD_AGENT_KEY" \
  "https://clawcard.sh/api/agents/emails?unread=true"

# Create card
curl -X POST -H "Authorization: Bearer $CLAWCARD_AGENT_KEY" \
  -H "content-type: application/json" \
  -d '{"amountCents": 2000, "type": "single_use", "memo": "test"}' \
  https://clawcard.sh/api/agents/cards
```

Full API reference: [https://www.clawcard.sh/docs](https://www.clawcard.sh/docs)

**LangGraph custom tool example:**

```python
import httpx
from langchain.tools import tool

CLAWCARD_API = "https://clawcard.sh/api/agents"
HEADERS = {"Authorization": f"Bearer {os.environ['CLAWCARD_AGENT_KEY']}"}

@tool
def get_clawcard_emails(unread: bool = True) -> str:
    """Check the agent's email inbox."""
    r = httpx.get(
        f"{CLAWCARD_API}/emails",
        params={"unread": str(unread).lower()},
        headers=HEADERS,
    )
    return r.text
```

---

## Security & Production Guardrails

As an SDE3, these are non-negotiable:

| Control | Implementation |
|---|---|
| Key isolation | One key per agent — revoke individually |
| Hard spend limits | Per-card caps enforced server-side |
| Instant freeze | `clawcard agent wallet freeze` or `card freeze` |
| Full audit trail | `clawcard agent activity --json` |
| Encrypted vault | AES-256, keys never leave Clawcard HSMs |
| No personal card exposure | Virtual cards die with the agent key |

**If an agent goes rogue:**

```bash
clawcard keys revoke ak_EXAMPLE_replace_with_real_key
# → All resources (email, phone, cards, wallet) instantly deactivated
```

**Production checklist:**
- [ ] Store agent API key in Secret Manager / Vault, not `.env`
- [ ] Set per-card spending limits conservatively (start low, raise if needed)
- [ ] Run `clawcard agent activity` daily; alert on anomalies
- [ ] Use `merchant_locked` cards for recurring charges (prevents drift)
- [ ] Name credentials consistently: `--service <service> --key <field>`
- [ ] Never commit `~/.clawcard/.env` — add to global `.gitignore`

---

## Makefile Integration

Add these targets to the project `Makefile`:

```makefile
run-clawcard:
	python examples/server_clawcard_agent.py

run-clawcard-agent:
	python examples/clawcard_langchain_agent.py

run-clawcard-mcp:
	python examples/clawcard_mcp_bridge.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `clawcard: command not found` | `npm install -g @clawcard/cli` then restart terminal |
| `ClawcardError: clawcard CLI not found` | Same — ensure `npm bin -g` is in `PATH` |
| `clawcard exited 401` | Session expired: re-run `clawcard signup` or re-export key |
| No invite code | DM [@cblovescode](https://x.com/cblovescode) on X |
| SMS delay | `clawcard agent sms --limit 10` — may take up to 60s |
| Card declined | Increase `--amount`; check `clawcard billing balance` |
| Verification email not arriving | `clawcard agent emails --unread --json` after 30s; some services delay |
| `ImportError: No module named clawcard_tool` | Run from repo root: `python -m examples.clawcard_tool` or adjust `PYTHONPATH` |

---

## File Reference

| File | Purpose |
|---|---|
| `examples/clawcard_tool.py` | Python wrapper class for all `clawcard agent` CLI calls |
| `examples/server_clawcard_agent.py` | LitServe HTTP API exposing all Clawcard operations |
| `examples/clawcard_langchain_agent.py` | GPT-4o LangChain agent with 13 Clawcard tools |
| `examples/clawcard_mcp_bridge.py` | MCP bridge — exposes tools natively to Claude Desktop |

---

## Next Steps

1. **Start here:** `clawcard agent info --json` — confirm your identity
2. **First real task:** Tell the LangChain agent to sign up for a free API tier
3. **Scale up:** Add Clawcard tools to your existing LangGraph / CrewAI workflows
4. **Production:** Wire the LitServe server behind your agent orchestration layer

**Resources:**
- Official docs & full API: [https://www.clawcard.sh/docs](https://www.clawcard.sh/docs)
- CLI help: `clawcard help`
- Issues & community: [https://x.com/cblovescode](https://x.com/cblovescode)

---

*This infrastructure is the "Cursor moment" for agent execution.  Five minutes of setup removes months of custom hacks for email verification, payments, and identity.  Ship agents that actually get things done.*
