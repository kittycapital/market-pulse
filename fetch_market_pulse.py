#!/usr/bin/env python3
"""
HerdVibe Market Pulse — 매시간 자동 실행되는 마켓 뉴스 파이프라인
CryptoPanic → FxTwitter → Gemini → JSON → HerdVibe 대시보드
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

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "market-pulse.json")
MAX_ITEMS_PER_CATEGORY = 8
MAX_TOTAL_ITEMS = 20
MAX_HISTORY_HOURS = 72  # 최근 72시간 데이터 보관

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
        print(f"[ERROR] HTTP {e.code} for {url}")
        if body:
            print(f"[ERROR] Response: {body}")
        return None
    except Exception as e:
        print(f"[ERROR] {e} for {url}")
        return None


def fetch_news_posts():
    """CryptoCompare News API에서 최근 뉴스를 가져온다. API 키 불필요."""
    posts = []

    # CryptoCompare 뉴스 API (무료, 키 불필요)
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    print(f"  Fetching from CryptoCompare...")
    data = fetch_json(url)

    if data and data.get("Data"):
        print(f"  → Got {len(data['Data'])} articles")
        for item in data["Data"][:MAX_TOTAL_ITEMS]:
            post = parse_news_item(item)
            if post:
                posts.append(post)
    else:
        print(f"  → No results from CryptoCompare")

    return posts[:MAX_TOTAL_ITEMS]


def parse_news_item(item):
    """CryptoCompare 뉴스 아이템을 파싱한다."""
    try:
        title = item.get("title", "")
        body = item.get("body", "")
        source_url = item.get("url", "") or item.get("guid", "")
        published = item.get("published_on", 0)
        source_name = item.get("source_info", {}).get("name", item.get("source", ""))
        image_url = item.get("imageurl", "")
        categories = item.get("categories", "")

        # 타임스탬프 변환
        published_iso = ""
        if published:
            published_iso = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()

        # 카테고리 판별
        category = determine_category_from_text(title, body, categories)

        # 트위터 URL인지 확인
        twitter_url = None
        tweet_id = None
        if source_url and ("twitter.com" in source_url or "x.com" in source_url):
            twitter_url = source_url
            tweet_id = extract_tweet_id(source_url)

        # 관련 코인 추출
        coins = []
        if categories:
            for cat in categories.split("|"):
                cat = cat.strip()
                if cat.isupper() and len(cat) <= 5:
                    coins.append(cat)

        return {
            "title": title,
            "body_preview": body[:200] if body else "",
            "source_url": source_url,
            "source_name": source_name,
            "twitter_url": twitter_url,
            "tweet_id": tweet_id,
            "published_at": published_iso,
            "category": category,
            "importance": 3,
            "currencies": coins[:4],
            "media_url": image_url if image_url and image_url.startswith("http") else None,
        }
    except Exception as e:
        print(f"[WARN] Failed to parse item: {e}")
        return None


def determine_category_from_text(title, body, categories):
    """뉴스 카테고리를 판별한다."""
    text = (title + " " + body + " " + categories).lower()
    
    # 매크로 키워드
    macro_keywords = [
        "fed", "fomc", "cpi", "ppi", "gdp", "inflation", "interest rate",
        "treasury", "yield", "bond", "unemployment", "nonfarm", "payroll",
        "powell", "ecb", "boj", "tariff", "trade war", "recession",
        "dollar", "dxy", "gold", "oil", "crude",
    ]
    
    # 주식 키워드
    stock_keywords = [
        "stock", "nasdaq", "s&p", "spy", "qqq", "nvda", "nvidia",
        "apple", "aapl", "tesla", "tsla", "microsoft", "msft",
        "earnings", "revenue", "eps", "ipo", "sec filing",
        "market cap", "dow jones",
    ]
    
    for kw in macro_keywords:
        if kw in text:
            return "macro"
    
    for kw in stock_keywords:
        if kw in text:
            return "stocks"
    
    return "crypto"  # 기본값


def extract_tweet_id(url):
    """트위터 URL에서 트윗 ID를 추출한다."""
    match = re.search(r"/status/(\d+)", url)
    return match.group(1) if match else None


def fetch_fxtwitter_data(tweet_id, username="i"):
    """FxTwitter API로 트윗 상세 데이터를 가져온다."""
    if not tweet_id:
        return None
    
    url = f"https://api.fxtwitter.com/{username}/status/{tweet_id}"
    data = fetch_json(url)
    
    if data and data.get("code") == 200 and "tweet" in data:
        tweet = data["tweet"]
        return {
            "text": tweet.get("text", ""),
            "author_name": tweet.get("author", {}).get("name", ""),
            "author_handle": tweet.get("author", {}).get("screen_name", ""),
            "author_avatar": tweet.get("author", {}).get("avatar_url", ""),
            "likes": tweet.get("likes", 0),
            "retweets": tweet.get("retweets", 0),
            "replies": tweet.get("replies", 0),
            "created_at": tweet.get("created_at", ""),
            "media_url": extract_media_url(tweet),
        }
    return None


def extract_media_url(tweet):
    """트윗에서 미디어 URL을 추출한다."""
    media = tweet.get("media", {})
    if not media:
        return None
    
    # 사진 우선
    photos = media.get("photos", [])
    if photos:
        return photos[0].get("url", None)
    
    # 비디오 썸네일
    videos = media.get("videos", [])
    if videos:
        return videos[0].get("thumbnail_url", None)
    
    return None


def translate_with_gemini(items):
    """Gemini API로 한글 번역 + 해설을 생성한다."""
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set, skipping translation")
        for item in items:
            item["korean_title"] = item["title"]
            item["korean_analysis"] = ""
        return items

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    # 전체를 한 번에 번역 (API 호출 1회로 최소화)
    prompt = build_translation_prompt(items)
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
        },
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = (
                    result.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                parse_gemini_response(text, items)
                print(f"  → Translation OK")
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
                wait = 60 * (attempt + 1)  # 60s, 120s
                print(f"  → Waiting {wait}s before retry...")
                time.sleep(wait)
                # 새 Request 객체 생성 (이전 것은 소비됨)
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            else:
                print("  → Using original titles (no translation)")
                for item in items:
                    item["korean_title"] = item["title"]
                    item["korean_analysis"] = ""
        except Exception as e:
            print(f"[ERROR] Gemini API: {e}")
            for item in items:
                item["korean_title"] = item["title"]
                item["korean_analysis"] = ""
            break

    return items


def build_translation_prompt(batch):
    """Gemini에 보낼 번역 프롬프트를 생성한다."""
    items_text = ""
    for idx, item in enumerate(batch):
        # 트윗 원문이 있으면 사용, 없으면 제목 사용
        source_text = item.get("tweet_text", item.get("body_preview", item["title"]))
        items_text += f"\n[{idx}] ({item['category']}) {source_text}\n"

    return f"""당신은 금융 시장 전문 번역가입니다. 아래 영문 뉴스/트윗을 한국어로 번역하고 트레이더 관점에서 간단한 해설을 추가해주세요.

규칙:
1. 각 항목에 대해 korean_title (한글 번역, 1줄)과 korean_analysis (왜 중요한지 해설, 1~2줄)를 작성
2. 전문 용어는 한영 병기 (예: 연방준비제도(Fed))
3. 트레이더에게 실질적으로 유용한 맥락 제공
4. 반드시 아래 JSON 형식으로만 응답 (다른 텍스트 없이)

뉴스 목록:
{items_text}

응답 형식 (JSON만):
[
  {{"idx": 0, "korean_title": "한글 번역", "korean_analysis": "왜 중요한지 해설"}},
  ...
]"""


def parse_gemini_response(text, batch):
    """Gemini 응답을 파싱하여 batch에 적용한다."""
    try:
        # JSON 블록 추출
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        
        translations = json.loads(text)
        
        for t in translations:
            idx = t.get("idx", -1)
            if 0 <= idx < len(batch):
                batch[idx]["korean_title"] = t.get("korean_title", batch[idx]["title"])
                batch[idx]["korean_analysis"] = t.get("korean_analysis", "")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[WARN] Failed to parse Gemini response: {e}")
        print(f"[DEBUG] Response text: {text[:500]}")
        for item in batch:
            item.setdefault("korean_title", item["title"])
            item.setdefault("korean_analysis", "")


def load_existing_data():
    """기존 데이터를 로드한다."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"items": [], "last_updated": ""}


def merge_and_dedupe(existing, new_items):
    """기존 데이터와 새 데이터를 병합하고 중복 제거한다."""
    # 기존 아이템의 source_url 셋
    existing_urls = {item["source_url"] for item in existing["items"]}
    
    # 새 아이템 중 중복 아닌 것만 추가
    merged = list(new_items)  # 새 아이템을 앞에
    for item in existing["items"]:
        if item["source_url"] not in {i["source_url"] for i in merged}:
            merged.append(item)
    
    # 72시간 이내 데이터만 보관
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_HISTORY_HOURS)
    filtered = []
    for item in merged:
        try:
            pub = item.get("fetched_at", item.get("published_at", ""))
            if pub:
                # ISO 형식 파싱
                pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                if pub_dt > cutoff:
                    filtered.append(item)
                    continue
        except Exception:
            pass
        filtered.append(item)  # 파싱 실패 시 일단 보관
    
    # 최대 200개까지만 보관
    return filtered[:200]


def save_data(items):
    """데이터를 JSON 파일로 저장한다."""
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


def main():
    print("=" * 60)
    print("HerdVibe Market Pulse — Fetching latest market news")
    print(f"Time: {datetime.now(KST).strftime('%Y-%m-%d %H:%M KST')}")
    print("=" * 60)
    
    # 1. CryptoCompare에서 뉴스 수집
    print("\n[1/4] Fetching news...")
    posts = fetch_news_posts()
    print(f"  → Got {len(posts)} posts")
    
    # 2. 트위터 URL이 있는 경우 FxTwitter로 보강
    print("\n[2/4] Enriching with FxTwitter...")
    enriched_count = 0
    for post in posts:
        if post.get("tweet_id"):
            tweet_data = fetch_fxtwitter_data(post["tweet_id"])
            if tweet_data:
                post["tweet_text"] = tweet_data["text"]
                post["author_name"] = tweet_data["author_name"]
                post["author_handle"] = tweet_data["author_handle"]
                post["author_avatar"] = tweet_data["author_avatar"]
                post["likes"] = tweet_data["likes"]
                post["retweets"] = tweet_data["retweets"]
                post["media_url"] = tweet_data["media_url"]
                enriched_count += 1
            time.sleep(0.5)  # FxTwitter 레이트 리밋 배려
    print(f"  → Enriched {enriched_count} tweets")
    
    # 3. Gemini로 한글 번역 + 해설
    print("\n[3/4] Translating with Gemini...")
    translate_with_gemini(posts)
    translated = sum(1 for p in posts if p.get("korean_analysis"))
    print(f"  → Translated {translated} items")
    
    # 4. 타임스탬프 추가 및 저장
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
