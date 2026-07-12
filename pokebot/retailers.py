"""Stock detection for retailer product pages.

Strategy: fetch the product page like a normal browser visit and look for
in-stock / out-of-stock marker text. Each supported retailer has default
markers; any product can override them in the config with `in_stock_text`
/ `out_of_stock_text`, which also lets you watch retailers not listed here.
"""

import urllib.request

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Marker text per retailer, matched case-insensitively against the page HTML.
# out_of_stock markers win over in_stock markers when both appear.
RETAILER_MARKERS = {
    "pokemoncenter": {
        "in_stock": ["add to cart"],
        "out_of_stock": ["sold out", "out of stock", "coming soon"],
    },
    "target": {
        "in_stock": ["\"availability_status\":\"IN_STOCK\"", "add to cart"],
        "out_of_stock": ["\"availability_status\":\"OUT_OF_STOCK\"", "sold out"],
    },
    "bestbuy": {
        "in_stock": ["\"buttonstate\":\"ADD_TO_CART\"", "add to cart"],
        "out_of_stock": ["\"buttonstate\":\"SOLD_OUT\"", "sold out", "coming soon"],
    },
    "walmart": {
        "in_stock": ["add to cart"],
        "out_of_stock": ["out of stock", "currently unavailable"],
    },
    "gamestop": {
        "in_stock": ["add to cart"],
        "out_of_stock": ["not available", "sold out"],
    },
}


def detect_retailer(url):
    for name in RETAILER_MARKERS:
        if name in url.replace("-", "").replace(".", ""):
            return name
    return None


def fetch_page(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace").lower()


def check_stock(product):
    """Return True if the product looks purchasable right now."""
    html = fetch_page(product["url"])

    markers = RETAILER_MARKERS.get(detect_retailer(product["url"]), {})
    in_markers = [m.lower() for m in product.get("in_stock_text", markers.get("in_stock", []))]
    out_markers = [m.lower() for m in product.get("out_of_stock_text", markers.get("out_of_stock", []))]

    if not in_markers and not out_markers:
        raise ValueError(
            "Unknown retailer and no in_stock_text/out_of_stock_text configured "
            f"for {product['url']}"
        )

    if any(m in html for m in out_markers):
        return False
    if any(m in html for m in in_markers):
        return True
    # Neither marker found — page layout may have changed or we got an
    # interstitial page. Treat as out of stock rather than false-alerting.
    return False
