# Pokemon Restock Monitor

Watches retailer product pages and alerts you **the moment** an item comes
back in stock — terminal bell + Discord ping with the direct product link, so
you can click through and check out before it sells out again.

This is deliberately *not* an auto-checkout bot. Auto-checkout requires
defeating retailer anti-bot systems, which gets accounts and payment methods
banned. Speed-wise, nearly all of the edge is in knowing about the restock
within seconds — this gives you that.

## Setup

Requires Python 3.9+, no third-party packages.

1. Copy the example config:
   ```
   cp config.example.json config.json
   ```
2. Edit `config.json`:
   - Add each product's **name** and exact **product page URL**.
   - (Optional but recommended) Add a `discord_webhook_url` so alerts hit your
     phone. In any Discord server you control: Server Settings → Integrations
     → Webhooks → New Webhook → Copy URL. Enable Discord mobile notifications.
3. Run it:
   ```
   python monitor.py
   ```
   Leave it running in a terminal (or under `tmux`/`screen`, or as a service).

## How it works

- Checks each product roughly every `poll_seconds` (default 60s, minimum 30s)
  with randomized jitter, which is polite to the retailer and less likely to
  get your IP rate-limited.
- Detects stock by looking for retailer-specific marker text on the page
  ("Add to Cart" vs "Sold Out"). Built-in support: Pokemon Center, Target,
  Best Buy, Walmart, GameStop. Any other store works by setting
  `in_stock_text` / `out_of_stock_text` on the product entry.
- Alerts only on the out-of-stock → in-stock transition, with a 5-minute
  cooldown per product, so you don't get spammed.

## Tips for actually winning the drop

- Keep yourself **logged in** to each retailer with your address and payment
  method saved — that turns checkout into two clicks.
- Run the monitor from your home connection. Retailers are more suspicious of
  cloud/datacenter IPs.
- Some sites (notably Pokemon Center) sometimes serve a bot-check interstitial
  to scripted requests. If a product never flips despite a known restock, the
  page markers may need adjusting — open the product page, find the exact
  sold-out text, and set it as `out_of_stock_text` in the config.
- Don't lower `poll_seconds` aggressively. Faster polling mostly increases the
  odds of getting blocked, not of winning.

## Ideas for later

- eBay Buy API integration for true (and sanctioned) auto-purchase of singles
  and sealed product under a max price.
- Price-comp alerts: flag listings priced under recent market comps.
