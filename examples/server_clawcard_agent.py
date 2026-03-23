"""
server_clawcard_agent.py — LitServe server powered by a Clawcard agent identity.

This server demonstrates how to embed Clawcard's autonomous identity + payments
stack directly inside a LitServe API.  Every request can:

  - look up the agent's provisioned email / phone / wallet
  - check the inbox for verification links
  - read SMS 2FA codes
  - create a one-time virtual card on demand
  - retrieve / store credentials from the encrypted vault
  - pay a 402-gated endpoint gaslessly via x402 or MPP

Run:
    python examples/server_clawcard_agent.py

Test:
    # Identity + budget
    curl -s -X POST localhost:8003/predict \
      -H "content-type: application/json" \
      -d '{"action": "info"}'

    # Check unread email
    curl -s -X POST localhost:8003/predict \
      -H "content-type: application/json" \
      -d '{"action": "emails", "unread": true}'

    # Create a $20 single-use card
    curl -s -X POST localhost:8003/predict \
      -H "content-type: application/json" \
      -d '{"action": "card_create", "amount_cents": 2000, "memo": "test card"}'

    # Store a credential
    curl -s -X POST localhost:8003/predict \
      -H "content-type: application/json" \
      -d '{"action": "creds_set", "service": "stripe", "key": "secret_key", "value": "<your-stripe-key>"}'

    # Gasless x402 payment
    curl -s -X POST localhost:8003/predict \
      -H "content-type: application/json" \
      -d '{"action": "wallet_send", "url": "https://api.example.com/premium", "protocol": "x402"}'
"""

import litserve as ls
from clawcard_tool import ClawcardAgent, ClawcardError


class ClawcardLitAPI(ls.LitAPI):
    """
    LitServe API that exposes Clawcard agent capabilities as HTTP endpoints.

    The `action` field in each request routes to the corresponding Clawcard
    operation.  All responses are JSON-serialisable dicts so they pass
    cleanly through LitServe's response pipeline.
    """

    def setup(self, device):
        self.cw = ClawcardAgent()

    def decode_request(self, request):
        return request

    def predict(self, request: dict) -> dict:
        action = request.get("action", "info")

        try:
            if action == "info":
                return self.cw.info()

            elif action == "emails":
                unread = request.get("unread", False)
                limit = int(request.get("limit", 10))
                return {"emails": self.cw.emails(unread=unread, limit=limit)}

            elif action == "email_get":
                email_id = request["email_id"]
                return self.cw.email_get(email_id)

            elif action == "sms":
                limit = int(request.get("limit", 5))
                return {"messages": self.cw.sms(limit=limit)}

            elif action == "card_create":
                amount_cents = int(request.get("amount_cents", 1000))
                card_type = request.get("card_type", "single_use")
                memo = request.get("memo")
                merchant = request.get("merchant")
                return self.cw.cards_create(
                    amount_cents=amount_cents,
                    card_type=card_type,
                    memo=memo,
                    merchant=merchant,
                )

            elif action == "card_details":
                card_id = request["card_id"]
                return self.cw.cards_details(card_id)

            elif action == "cards_list":
                return {"cards": self.cw.cards_list()}

            elif action == "card_freeze":
                card_id = request["card_id"]
                return self.cw.cards_freeze(card_id)

            elif action == "creds_set":
                return self.cw.creds_set(
                    service=request["service"],
                    key=request["key"],
                    value=request["value"],
                )

            elif action == "creds_get":
                value = self.cw.creds_get(
                    service=request["service"],
                    key=request["key"],
                )
                return {"value": value}

            elif action == "creds_list":
                return {"credentials": self.cw.creds_list(request.get("service"))}

            elif action == "wallet_balance":
                return self.cw.wallet_balance()

            elif action == "wallet_send":
                return self.cw.wallet_send(
                    url=request["url"],
                    protocol=request.get("protocol", "x402"),
                    amount_usd=request.get("amount_usd"),
                )

            elif action == "wallet_freeze":
                return self.cw.wallet_freeze()

            elif action == "activity":
                limit = int(request.get("limit", 20))
                return {"activity": self.cw.activity(limit=limit)}

            else:
                return {
                    "error": f"Unknown action: {action!r}",
                    "available_actions": [
                        "info", "emails", "email_get", "sms",
                        "card_create", "card_details", "cards_list", "card_freeze",
                        "creds_set", "creds_get", "creds_list",
                        "wallet_balance", "wallet_send", "wallet_freeze",
                        "activity",
                    ],
                }

        except ClawcardError as exc:
            return {"error": str(exc)}
        except KeyError as exc:
            return {"error": f"Missing required field: {exc}"}

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    ls.LitServer(ClawcardLitAPI()).run(port=8003)
