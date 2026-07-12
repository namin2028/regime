"""Pokemon restock monitor.

Polls retailer product pages at a gentle, human-like cadence and fires an
alert (Discord webhook + console bell) the moment an item flips from
out-of-stock to in-stock, so you can click through and check out immediately.

Usage:
    python monitor.py [path/to/config.json]

See config.example.json and README.md for setup.
"""

import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

from notify import send_alert
from retailers import check_stock

DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_POLL_SECONDS = 60
MIN_POLL_SECONDS = 30  # be polite; hammering pages gets your IP blocked anyway
ALERT_COOLDOWN_SECONDS = 300  # don't re-alert for the same product within 5 min


@dataclass
class ProductState:
    in_stock: bool = False
    last_alert_at: float = 0.0
    consecutive_errors: int = 0
    last_checked: str = field(default="never")


def load_config(path):
    with open(path) as f:
        config = json.load(f)
    if not config.get("products"):
        sys.exit("Config has no products. See config.example.json.")
    return config


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(config):
    poll = max(int(config.get("poll_seconds", DEFAULT_POLL_SECONDS)), MIN_POLL_SECONDS)
    webhook = config.get("discord_webhook_url", "")
    products = config["products"]
    states = {p["url"]: ProductState() for p in products}

    log(f"Watching {len(products)} product(s), checking every ~{poll}s each.")
    for p in products:
        log(f"  - {p['name']}")

    while True:
        for product in products:
            state = states[product["url"]]
            try:
                in_stock = check_stock(product)
                state.consecutive_errors = 0
            except Exception as e:
                state.consecutive_errors += 1
                if state.consecutive_errors in (1, 5, 20):
                    log(f"WARN {product['name']}: check failed ({e})")
                continue
            finally:
                state.last_checked = datetime.now().isoformat(timespec="seconds")
                # small jitter between products so requests don't look like a burst
                time.sleep(random.uniform(1.5, 4.0))

            if in_stock and not state.in_stock:
                now = time.time()
                if now - state.last_alert_at > ALERT_COOLDOWN_SECONDS:
                    log(f"*** IN STOCK: {product['name']} -> {product['url']}")
                    send_alert(webhook, product)
                    state.last_alert_at = now
            elif not in_stock and state.in_stock:
                log(f"{product['name']} went out of stock again.")
            state.in_stock = in_stock

        # jittered sleep so polling doesn't happen on a robotic fixed clock
        time.sleep(poll + random.uniform(-5, 10))


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        sys.exit(
            f"No config found at '{config_path}'. Copy config.example.json to "
            "config.json and fill in your products."
        )
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("\nStopped.")
