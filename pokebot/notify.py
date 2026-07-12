"""Alert delivery: Discord webhook plus a loud console fallback."""

import json
import urllib.request


def send_alert(webhook_url, product):
    console_alert(product)
    if webhook_url:
        try:
            discord_alert(webhook_url, product)
        except Exception as e:
            print(f"Discord alert failed: {e}")


def console_alert(product):
    # \a rings the terminal bell on most terminals
    print("\a" + "=" * 60)
    print(f"  RESTOCK: {product['name']}")
    print(f"  BUY NOW: {product['url']}")
    print("=" * 60 + "\a")


def discord_alert(webhook_url, product):
    payload = {
        "content": f"🚨 **RESTOCK** — {product['name']}\n{product['url']}",
    }
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10).read()
