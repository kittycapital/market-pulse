#!/usr/bin/env python3
"""
HerdVibe Market Pulse — 매시간 자동 실행되는 마켓 뉴스 파이프라인
CryptoCompare + Finnhub → FxTwitter → Gemini → JSON → HerdVibe 대시보드
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

# ─── 설정 ────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "market-pulse.json")
MAX_ITEMS_PER_SOURCE = 15
MAX_TOTAL_ITEMS = 30
MAX_HISTORY_HOURS = 72

KST = timezone(timedelta(hours=9))


def fetch_json(url, headers=None):
    """URL에서 JSON 데이터를 가져온다."""
    req = urllib.request.Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    req.add_header("User-Agent", "HerdVibe-MarketPulse/1.0")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:300]
        except Exception:
            pass
        print(f"[ERROR] HTTP {e.code} for {url[:80]}")
        if body:
            print(f"[ERROR] Response: {body[:150]}")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# ─── 뉴스 수집 ──────────────────────────────────────────

def fetch_cryptocompare_news():
    """CryptoCompare에서 크립토 뉴스를 가져온다."""
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    print("  [CryptoCompare] Fetching...")
    data = fetch_json(url)

    posts = []
    if data and data.get("Data"):
        print(f"  [CryptoCompare] Got {len(data['Data'])} articles")
        for item in data["Data"][:MAX_ITEMS_PER_SOURCE]:
            post = parse_cryptocompare_item(item)
            if post:
                posts.append(post)
    else:
        print("  [CryptoCompare] No results")
    return posts


def fetch_finnhub_news():
    """Finnhub에서 미국 주식/매크로 뉴스를 가져온다."""
    if not FINNHUB_API_KEY:
        print("  [Finnhub] API key not set, skipping")
        return []

    url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
    print("  [Finnhub] Fetching general news...")
    data = fetch_json(url)

    posts = []
    if data and isinstance(data, list):
        print(f"  [Finnhub] Got {len(data)} articles")
        for item in data[:MAX_ITEMS_PER_SOURCE]:
            post = parse_finnhub_item(item)
            if post:
                posts.append(post)
    else:
        print("  [Finnhub] No results")
    return posts


def parse_cryptocompare_item(item):
    """CryptoCompare 뉴스 아이템을 파싱한다."""
    try:
        title = item.get("title", "")
        body = item.get("body", "")
        source_url = item.get("url", "") or item.get("guid", "")
        published = item.get("published_on", 0)
        source_name = item.get("source_info", {}).get("name", item.get("source", ""))
        image_url = item.get("imageurl", "")
        categories = item.get("categories", "")

        published_iso = ""
        if published:
            published_iso = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()

        category = determine_category(title, body, categories)

        twitter_url = None
        tweet_id = None
        if source_url and ("twitter.com" in source_url or "x.com" in source_url):
            twitter_url = source_url
            tweet_id = extract_tweet_id(source_url)

        coins = []
        if categories:
            for cat in categories.split("|"):
                cat = cat.strip()
                if cat.isupper() and len(cat) <= 5:
                    coins.append(cat)

        return {
            "title": title,
            "body_preview": body[:300] if body else "",
            "source_url": source_url,
            "source_name": source_name,
            "twitter_url": twitter_url,
            "tweet_id": tweet_id,
            "published_at": published_iso,
            "category": category,
            "currencies": coins[:4],
            "media_url": image_url if image_url and image_url.startswith("http") else None,
        }
    except Exception as e:
        print(f"[WARN] CryptoCompare parse error: {e}")
        return None


def parse_finnhub_item(item):
    """Finnhub 뉴스 아이템을 파싱한다."""
    try:
        title = item.get("headline", "")
        body = item.get("summary", "")
        source_url = item.get("url", "")
        published = item.get("datetime", 0)
        source_name = item.get("source", "")
        image_url = item.get("image", "")
        related = item.get("related", "")

        published_iso = ""
        if published:
            published_iso = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()

        category = determine_category(title, body, related)

        return {
            "title": title,
            "body_preview": body[:300] if body else "",
            "source_url": source_url,
            "source_name": source_name,
            "twitter_url": None,
            "tweet_id": None,
            "published_at": published_iso,
            "category": category,
            "currencies": [],
            "media_url": image_url if image_url and image_url.startswith("http") else None,
        }
    except Exception as e:
        print(f"[WARN] Finnhub parse error: {e}")
        return None


def determine_category(title, body, extra=""):
    """뉴스 카테고리를 판별한다."""
    text = (title + " " + body + " " + extra).lower()

    macro_keywords = [
        "fed", "fomc", "cpi", "ppi", "gdp", "inflation", "interest rate",
        "treasury", "yield", "bond", "unemployment", "nonfarm", "payroll",
        "powell", "ecb", "boj", "tariff", "trade war", "recession",
        "dollar", "dxy", "gold", "oil", "crude", "geopolitical",
    ]
    crypto_keywords = [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain",
        "defi", "nft", "solana", "sol", "xrp", "dogecoin", "altcoin",
        "binance", "coinbase", "mining", "halving", "stablecoin",
    ]
    stock_keywords = [
        "stock", "nasdaq", "s&p", "spy", "qqq", "nvda", "nvidia",
        "apple", "aapl", "tesla", "tsla", "microsoft", "msft",
        "earnings", "revenue", "eps", "ipo", "sec filing",
        "market cap", "dow jones", "wall street", "nyse",
        "amazon", "amzn", "google", "alphabet", "meta",
        "quarterly", "guidance", "buyback", "dividend",
    ]

    for kw in macro_keywords:
        if kw in text:
            return "macro"
    for kw in crypto_keywords:
        if kw in text:
            return "crypto"
    for kw in stock_keywords:
        if kw in text:
            return "stocks"

    return "stocks"


def extract_tweet_id(url):
    match = re.search(r"/status/(\d+)", url)
    return match.group(1) if match else None


# ─── FxTwitter 보강 ──────────────────────────────────────

def fetch_fxtwitter_data(tweet_id, username="i"):
    if not tweet_id:
        return None
    url = f"https://api.fxtwitter.com/{username}/status/{tweet_id}"
    data = fetch_json(url)
    if data and data.get("code") == 200 and "tweet" in data:
        tweet = data["tweet"]
        media = tweet.get("media", {})
        media_url = None
        if media:
            photos = media.get("photos", [])
            if photos:
                media_url = photos[0].get("url")
            else:
                videos = media.get("videos", [])
                if videos:
                    media_url = videos[0].get("thumbnail_url")
        return {
            "text": tweet.get("text", ""),
            "author_name": tweet.get("author", {}).get("name", ""),
            "author_handle": tweet.get("author", {}).get("screen_name", ""),
            "author_avatar": tweet.get("author", {}).get("avatar_url", ""),
            "likes": tweet.get("likes", 0),
            "retweets": tweet.get("retweets", 0),
            "media_url": media_url,
        }
    return None


# ─── Gemini 번역 ────────────────────────────────────────

def translate_with_gemini(items):
    """Gemini API로 한글 100자 요약을 생성한다."""
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set, skipping translation")
        for item in items:
            item["korean_summary"] = ""
        return items

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    # 배치로 나누기 (한번에 15개씩 → 응답 잘림 방지)
    batch_size = 15
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        prompt = build_translation_prompt(batch)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 16384},
        }

        max_retries = 3
        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    text = (
                        result.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    parse_gemini_response(text, batch)
                    print(f"  → Batch {i//batch_size + 1} OK")
                    break
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")[:200]
                except Exception:
                    pass
                print(f"[ERROR] Gemini API: HTTP {e.code} (attempt {attempt+1}/{max_retries})")
                if body:
                    print(f"[ERROR] {body[:150]}")
                if e.code == 429 and attempt < max_retries - 1:
                    wait = 60 * (attempt + 1)
                    print(f"  → Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    for item in batch:
                        item.setdefault("korean_summary", "")
                    break
            except Exception as e:
                print(f"[ERROR] Gemini API: {e}")
                for item in batch:
                    item.setdefault("korean_summary", "")
                break

        time.sleep(3)  # 배치 간 대기

    return items


def build_translation_prompt(batch):
    items_text = ""
    for idx, item in enumerate(batch):
        source = item.get("tweet_text", item.get("body_preview", item["title"]))
        items_text += f"\n[{idx}] ({item['category']}) {item['title']}\n내용: {source[:200]}\n"

    return f"""당신은 금융 뉴스 번역가입니다. 아래 영문 뉴스를 한국어로 요약해주세요.

규칙:
1. 각 항목에 대해 korean_summary (중립적 한글 요약, 80~120자)를 작성
2. 객관적이고 중립적인 뉴스 요약 스타일
3. 전문 용어는 한영 병기 (예: 연방준비제도(Fed))
4. 반드시 아래 JSON 형식으로만 응답 (다른 텍스트 없이)

뉴스 목록:
{items_text}

응답 형식 (JSON만):
[
  {{"idx": 0, "korean_summary": "80~120자 한글 요약"}},
  ...
]"""


def parse_gemini_response(text, batch):
    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # 잘린 JSON 복구 시도
        try:
            translations = json.loads(text)
        except json.JSONDecodeError:
            # 마지막 완전한 객체까지만 파싱
            last_brace = text.rfind("}")
            if last_brace > 0:
                truncated = text[:last_brace + 1]
                # 배열 닫기
                if not truncated.rstrip().endswith("]"):
                    truncated = truncated.rstrip().rstrip(",") + "]"
                translations = json.loads(truncated)
                print(f"  [WARN] Recovered {len(translations)} items from truncated response")
            else:
                raise

        for t in translations:
            idx = t.get("idx", -1)
            if 0 <= idx < len(batch):
                batch[idx]["korean_summary"] = t.get("korean_summary", "")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[WARN] Failed to parse Gemini response: {e}")
        print(f"[DEBUG] Response: {text[:500]}")
        for item in batch:
            item.setdefault("korean_summary", "")


# ─── 데이터 저장 ─────────────────────────────────────────

def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"items": [], "last_updated": ""}


def merge_and_dedupe(existing, new_items):
    merged = list(new_items)
    for item in existing["items"]:
        if item["source_url"] not in {i["source_url"] for i in merged}:
            merged.append(item)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_HISTORY_HOURS)
    filtered = []
    for item in merged:
        try:
            pub = item.get("fetched_at", item.get("published_at", ""))
            if pub:
                pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                if pub_dt > cutoff:
                    filtered.append(item)
                    continue
        except Exception:
            pass
        filtered.append(item)
    return filtered[:200]


def save_data(items):
    os.makedirs(DATA_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    output = {
        "last_updated": now,
        "last_updated_kst": now_kst,
        "total_items": len(items),
        "items": items,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved {len(items)} items to {OUTPUT_FILE}")
    print(f"[OK] Last updated: {now_kst}")


# ─── 메인 ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HerdVibe Market Pulse — Fetching latest market news")
    print(f"Time: {datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}")
    print("=" * 60)

    # 1. 뉴스 수집
    print("\n[1/4] Fetching news...")
    crypto_posts = fetch_cryptocompare_news()
    finnhub_posts = fetch_finnhub_news()

    seen_urls = set()
    posts = []
    for post in crypto_posts + finnhub_posts:
        if post["source_url"] not in seen_urls:
            seen_urls.add(post["source_url"])
            posts.append(post)
    posts = posts[:MAX_TOTAL_ITEMS]
    print(f"  → Total: {len(posts)} (crypto src: {len(crypto_posts)}, finnhub src: {len(finnhub_posts)})")

    # 2. FxTwitter 보강
    print("\n[2/4] Enriching with FxTwitter...")
    enriched = 0
    for post in posts:
        if post.get("tweet_id"):
            td = fetch_fxtwitter_data(post["tweet_id"])
            if td:
                post.update({
                    "tweet_text": td["text"], "author_name": td["author_name"],
                    "author_handle": td["author_handle"], "author_avatar": td["author_avatar"],
                    "likes": td["likes"], "retweets": td["retweets"], "media_url": td["media_url"],
                })
                enriched += 1
            time.sleep(0.5)
    print(f"  → Enriched {enriched} tweets")

    # 3. Gemini 한글 요약
    print("\n[3/4] Translating with Gemini...")
    translate_with_gemini(posts)
    translated = sum(1 for p in posts if p.get("korean_summary"))
    print(f"  → Translated {translated} items")

    # 4. 저장
    print("\n[4/4] Saving data...")
    now = datetime.now(timezone.utc).isoformat()
    for post in posts:
        post["fetched_at"] = now
    existing = load_existing_data()
    all_items = merge_and_dedupe(existing, posts)
    save_data(all_items)
    print("\n✅ Market Pulse update complete!")


if __name__ == "__main__":
    main()
