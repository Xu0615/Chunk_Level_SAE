#!/usr/bin/env python3
"""Build the self-contained Keyword-vs-SAE multi-page review dashboard.

The dashboard compares the locked keyword baseline with four independent SAE
retrieval branches.  It embeds the existing Top-20 results and the corresponding
full candidate-page text in one offline HTML file.  No retrieval result is
recomputed or reordered by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from chunk_saes.plot_style import METHOD_COLORS


DASHBOARD_FILENAME = "keyword_vs_sae_multi_page_selection.html"
METHOD_ORDER = ("keyword", "token", "temporal", "mean", "cross")
METHODS = {
    "keyword": {
        "name": "Keyword baseline",
        "short": "Keyword",
        "family": "Literal retrieval",
        "color": "#8b98a8",
        "description": (
            "使用预先锁定的大小写无关短语/布尔规则扫描完整 10,000 页候选池，"
            "按规则条款与出现次数确定性排序。"
        ),
    },
    "token": {
        "name": "BatchTopK SAE",
        "short": "Token",
        "family": "Token SAE",
        "color": METHOD_COLORS["token"],
        "description": (
            "逐 token 经 BatchTopK SAE 编码并阈值化，再聚合为窗口和页面表示；"
            "使用同一组独立 seed 在本特征空间做 cosine Top-20。"
        ),
    },
    "temporal": {
        "name": "Temporal SAE",
        "short": "Temporal",
        "family": "Sequence SAE",
        "color": METHOD_COLORS["temporal"],
        "description": (
            "在 token 序列上进行 Temporal SAE 聚合与阈值化，使用完整 65,536 维字典；"
            "使用同一组独立 seed 做 cosine Top-20。"
        ),
    },
    "mean": {
        "name": "Mean-Chunk SAE",
        "short": "Mean",
        "family": "Chunk SAE",
        "color": METHOD_COLORS["mean"],
        "description": (
            "先对每个窗口的 hidden state 求均值，再经 Mean-Chunk SAE 编码；"
            "使用同一组独立 seed 做 cosine Top-20。"
        ),
    },
    "cross": {
        "name": "Cross-Chunk SAE",
        "short": "Cross",
        "family": "Cross-chunk SAE",
        "color": METHOD_COLORS["cross"],
        "description": (
            "对 chunk 级均值应用 Cross-Chunk SAE，形成偏文档级的稀疏表示；"
            "使用同一组独立 seed 做 cosine Top-20。"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Eval-9 single-file human review dashboard."
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Eval-9 code-page directory containing data/, demo/, and manifests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output HTML path. Defaults to "
            f"<base>/demo/{DASHBOARD_FILENAME}."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum number of results embedded per topic/method.",
    )
    parser.add_argument(
        "--keep-legacy-html",
        action="store_true",
        help=(
            "Keep the obsolete split index/pages HTML files. By default they "
            "are removed after the self-contained dashboard is written."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload: Any) -> str:
    """Return a deterministic digest for a JSON-compatible payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_title(text: str, fallback: str) -> str:
    """Extract a compact display title without interpreting source HTML/Markdown."""

    for raw_line in text.splitlines()[:80]:
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^[#>*+\-\s]+", "", line).strip()
        line = re.sub(r"\s+", " ", line)
        if len(line) >= 4:
            return line[:180]
    return fallback


def safe_script_json(payload: Any) -> str:
    """Serialize JSON so source text cannot terminate the embedding script tag."""

    value = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    return (
        value.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def compact_result(method_id: str, row: dict[str, Any]) -> dict[str, Any]:
    window = row.get("matched_window")
    if window:
        compact_window = {
            "window_index": window.get("window_index"),
            "token_start": window.get("token_start"),
            "token_end": window.get("token_end"),
            "valid_tokens": window.get("valid_tokens"),
            "similarity": window.get("similarity"),
        }
    else:
        compact_window = None

    compact_features = []
    for feature in row.get("shared_features", []):
        compact_features.append(
            {
                "feature_id": feature.get("feature_id"),
                "cosine_contribution": feature.get("cosine_contribution"),
                "query_value": feature.get("query_value"),
                "document_value": feature.get("document_value"),
            }
        )

    if method_id == "keyword":
        contains_keyword = True
        matched_atoms = row.get("matched_atoms", [])
        occurrence_count = row.get("occurrence_count", 0)
        matched_clause_count = row.get("matched_clause_count", 0)
    else:
        contains_keyword = bool(row.get("contains_keyword", False))
        matched_atoms = row.get("keyword_matched_atoms", [])
        occurrence_count = row.get("keyword_occurrence_count", 0)
        matched_clause_count = row.get("keyword_matched_clause_count", 0)

    return {
        "rank": int(row["rank"]),
        "document_index": int(row["document_index"]),
        "snippet": row.get("snippet", ""),
        "similarity": row.get("similarity"),
        "contains_keyword": contains_keyword,
        "matched_atoms": matched_atoms,
        "matched_clause_count": matched_clause_count,
        "occurrence_count": occurrence_count,
        "matched_window": compact_window,
        "shared_features": compact_features,
    }


def build_payload(base: Path, top_k: int) -> dict[str, Any]:
    queries_path = base / "demo" / "queries.locked.json"
    seeds_path = base / "demo" / "seeds.jsonl"
    method_map_path = base / "demo" / "method_map.json"
    summary_path = base / "retrieval_summary.json"
    profile_path = base / "data" / "sample_profile.json"
    sample_manifest_path = base / "data" / "sample_manifest.json"
    feature_manifest_path = base / "data" / "features" / "feature_manifest.json"
    demo_manifest_path = base / "demo" / "retrieval_demo_manifest.json"
    source_jsonl_path = base / "data" / "code_page_hqs8_9_10k.jsonl"

    required = [
        queries_path,
        seeds_path,
        method_map_path,
        summary_path,
        profile_path,
        sample_manifest_path,
        feature_manifest_path,
        demo_manifest_path,
        source_jsonl_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    queries_doc = load_json(queries_path)
    method_map = load_json(method_map_path)
    summary_doc = load_json(summary_path)
    profile = load_json(profile_path)
    sample_manifest = load_json(sample_manifest_path)
    feature_manifest = load_json(feature_manifest_path)
    demo_manifest = load_json(demo_manifest_path)

    query_configs = {row["query_id"]: row for row in queries_doc["queries"]}
    query_order = [row["query_id"] for row in queries_doc["queries"]]
    summary_by_query = {row["query_id"]: row for row in summary_doc["queries"]}
    retrieval_digest = json_digest(
        {
            "queries_locked": sha256_file(queries_path),
            "seeds_jsonl": sha256_file(seeds_path),
            "method_map": sha256_file(method_map_path),
            "results": {
                query_id: sha256_file(
                    base / "demo" / "results" / f"{query_id}.json"
                )
                for query_id in query_order
            },
            "source_jsonl": sample_manifest["files"]["jsonl"]["sha256"],
        }
    )

    seeds_by_query: dict[str, list[dict[str, Any]]] = {
        query_id: [] for query_id in query_order
    }
    with seeds_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = row["query_id"]
            if query_id not in seeds_by_query:
                raise ValueError(
                    f"{seeds_path}:{line_number}: unknown query_id {query_id!r}"
                )
            seeds_by_query[query_id].append(
                {
                    "seed_id": row["seed_id"],
                    "text": row["text"],
                    "source": f"seeds.jsonl:line {line_number}",
                }
            )

    topics: list[dict[str, Any]] = []
    needed_document_indices: set[int] = set()
    embedded_result_count = 0

    for query_id in query_order:
        result_path = base / "demo" / "results" / f"{query_id}.json"
        result_doc = load_json(result_path)
        if int(result_doc.get("candidate_count", -1)) != 10_000:
            raise ValueError(f"{result_path}: unexpected candidate_count")

        methods_payload: dict[str, Any] = {}
        for method_id in METHOD_ORDER:
            if method_id not in result_doc["methods"]:
                raise ValueError(f"{result_path}: missing method {method_id}")
            source_method = result_doc["methods"][method_id]
            rows = source_method.get("results", [])[:top_k]
            compact_rows = [compact_result(method_id, row) for row in rows]
            for row in compact_rows:
                needed_document_indices.add(row["document_index"])
            embedded_result_count += len(compact_rows)

            if method_id == "keyword":
                total_hits = int(
                    source_method.get("total_hits", len(source_method.get("results", [])))
                )
                literal_hits = len(compact_rows)
                mean_similarity = None
                overlap_keyword_top20 = None
            else:
                total_hits = None
                literal_hits = sum(
                    int(row["contains_keyword"]) for row in compact_rows
                )
                similarities = [
                    float(row["similarity"])
                    for row in compact_rows
                    if row["similarity"] is not None
                ]
                mean_similarity = (
                    sum(similarities) / len(similarities) if similarities else None
                )
                overlap_keyword_top20 = (
                    summary_by_query.get(query_id, {})
                    .get("methods", {})
                    .get(method_id, {})
                    .get("overlap_keyword_top20")
                )

            methods_payload[method_id] = {
                "results": compact_rows,
                "total_hits": total_hits,
                "literal_hits": literal_hits,
                "semantic_extensions": len(compact_rows) - literal_hits,
                "mean_similarity": mean_similarity,
                "overlap_keyword_top20": overlap_keyword_top20,
            }

        config = query_configs[query_id]
        topics.append(
            {
                "query_id": query_id,
                "title": config["title"],
                "keyword_rule": config["keyword_rule"],
                "seed_ids": config["seed_ids"],
                "seeds": seeds_by_query[query_id],
                "candidate_count": int(result_doc["candidate_count"]),
                "methods": methods_payload,
            }
        )

    documents: dict[str, dict[str, Any]] = {}
    row_count = 0
    with source_jsonl_path.open("r", encoding="utf-8") as handle:
        for document_index, line in enumerate(handle):
            row_count += 1
            if document_index not in needed_document_indices:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"{source_jsonl_path}:{document_index + 1}: text is not a string"
                )
            documents[str(document_index)] = {
                "title": extract_title(text, f"Document {document_index}"),
                "text": text,
                "characters": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
            }

    if row_count != 10_000:
        raise ValueError(f"{source_jsonl_path}: expected 10,000 rows, got {row_count}")
    missing = sorted(needed_document_indices - {int(key) for key in documents})
    if missing:
        raise ValueError(f"Missing source documents: {missing[:20]}")

    literal_top20 = {
        method_id: sum(
            topic["methods"][method_id]["literal_hits"] for topic in topics
        )
        for method_id in METHOD_ORDER[1:]
    }

    payload = {
        "format": "eval9-code-page-human-review-dashboard-v1",
        "meta": {
            "title": "关键词 vs SAE · 多网页筛选与评审",
            "experiment": "Eval 9 · Qwen3.5-9B-Base · Layer 21",
            "generated_date": "2026-08-28",
            "candidate_documents": int(summary_doc["candidate_documents"]),
            "topics": len(topics),
            "methods": len(METHOD_ORDER),
            "sae_methods": 4,
            "independent_seeds": sum(len(topic["seeds"]) for topic in topics),
            "document_windows": int(feature_manifest["document_windows"]),
            "processed_tokens": int(profile["total_processed_tokens"]),
            "quality_filter": "7 < k2_hqs_score ≤ 9",
            "feature_width": int(feature_manifest["feature_widths"]["token"]),
            "vector_top_k": int(feature_manifest["top_k"]),
            "embedded_top_k": top_k,
            "embedded_results": embedded_result_count,
            "embedded_unique_documents": len(documents),
            "sample_digest": sample_manifest["manifest_payload_sha256"],
            "feature_digest": feature_manifest["artifact_digest"],
            # Keep review state stable when only the HTML wrapper or its
            # manifest changes.  The digest is tied to retrieval inputs,
            # result JSON files, the blind map, and the source corpus.
            "demo_digest": retrieval_digest,
            "source_jsonl_sha256": sample_manifest["files"]["jsonl"]["sha256"],
            "source_jsonl_bytes": sample_manifest["files"]["jsonl"]["bytes"],
            "dashboard_purpose": (
                "人工比较关键词基线与四种 SAE，判断哪种方法更能从同一 10k 候选池中"
                "组织出主题一致、信息互补、足以支撑长程任务生成的多网页素材集合。"
            ),
        },
        "protocol": {
            "independence": (
                "Keyword 仅接收锁定关键词规则；四种 SAE 仅接收同一组预先锁定的独立 "
                "seed page。Keyword 命中不参与 SAE 的 seed 选择、候选过滤或排序。"
            ),
            "ranking": (
                "页面展示原始确定性排序：Keyword 为规则排序；每个 SAE 在自己的 "
                "65,536 维特征空间内对 seed 均值向量做 cosine Top-20。"
            ),
            "interpretation": (
                "这不是无监督聚类，而是围绕主题 query/seed 的五路独立检索。"
                "SAE 的关键词未命中项只是语义扩展候选，不自动等于相关；跨 SAE 的 "
                "cosine 数值也不可直接横向比较。最终优劣由盲评与集合级人工评分决定。"
            ),
        },
        "methods": METHODS,
        "method_order": list(METHOD_ORDER),
        "blind_map": method_map["queries"],
        "topics": topics,
        "documents": documents,
        "automatic_summary": {
            "sae_literal_hits_top20_across_topics": literal_top20,
            "sae_total_results_across_topics": (
                len(topics) * top_k * len(METHOD_ORDER[1:])
            ),
        },
    }
    return payload


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>Keyword vs SAE · 多网页筛选与评审</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071019;
      --bg-2: #091522;
      --panel: rgba(15, 29, 42, .88);
      --panel-solid: #0f1d2a;
      --panel-2: #132435;
      --panel-3: #182c3f;
      --text: #edf5fa;
      --text-soft: #c8d7e2;
      --muted: #8ea5b5;
      --faint: #647b8c;
      --line: rgba(164, 195, 213, .16);
      --line-strong: rgba(164, 195, 213, .29);
      --accent: #5be0c5;
      --accent-2: #58a9ff;
      --good: #58d6a5;
      --bad: #ff7188;
      --warn: #f2c166;
      --shadow: 0 26px 70px rgba(0, 0, 0, .28);
      --radius-xl: 24px;
      --radius-lg: 18px;
      --radius-md: 13px;
      --sidebar: 272px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-width: 320px;
      color: var(--text);
      background:
        radial-gradient(circle at 12% -10%, rgba(56, 133, 194, .24), transparent 34rem),
        radial-gradient(circle at 92% 2%, rgba(37, 188, 158, .13), transparent 30rem),
        linear-gradient(180deg, #071019 0%, #08131e 38%, #061019 100%);
      font-family: Inter, "SF Pro Display", "PingFang SC", "Microsoft YaHei",
                   system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .35;
      background-image:
        linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, black, transparent 78%);
    }
    button, input, select, textarea { font: inherit; }
    button, select { color: inherit; }
    button { cursor: pointer; }
    code, pre, .mono {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    }
    a { color: var(--accent); }
    .shell { width: min(1920px, calc(100% - 30px)); margin: 16px auto 80px; }
    .hero {
      position: relative;
      overflow: hidden;
      min-height: 350px;
      padding: clamp(26px, 4vw, 58px);
      border: 1px solid var(--line);
      border-radius: 30px;
      background:
        linear-gradient(115deg, rgba(19, 52, 75, .94), rgba(10, 28, 41, .94) 52%, rgba(9, 47, 48, .86)),
        var(--panel-solid);
      box-shadow: var(--shadow);
    }
    .hero::after {
      content: "";
      position: absolute;
      width: 520px; height: 520px;
      right: -160px; top: -260px;
      border-radius: 50%;
      border: 1px solid rgba(101, 229, 209, .23);
      box-shadow:
        0 0 0 48px rgba(101, 229, 209, .025),
        0 0 0 104px rgba(83, 164, 244, .018);
    }
    .hero-copy { position: relative; z-index: 1; max-width: 1080px; }
    .eyebrow {
      display: inline-flex; align-items: center; gap: 9px;
      margin-bottom: 18px;
      color: #91f0dc;
      font-size: 12px; font-weight: 800; letter-spacing: .16em;
      text-transform: uppercase;
    }
    .eyebrow::before {
      content: ""; width: 28px; height: 2px; border-radius: 2px;
      background: linear-gradient(90deg, var(--accent), transparent);
    }
    h1, h2, h3, h4, p { margin-top: 0; }
    .hero h1 {
      max-width: 950px;
      margin-bottom: 17px;
      font-size: clamp(35px, 5vw, 68px);
      line-height: 1.04;
      letter-spacing: -.045em;
      font-weight: 760;
    }
    .hero .lead {
      max-width: 960px;
      margin-bottom: 26px;
      color: var(--text-soft);
      font-size: clamp(15px, 1.5vw, 19px);
    }
    .hero strong { color: #fff; }
    .protocol-ribbon {
      position: relative; z-index: 1;
      display: grid; grid-template-columns: auto 1fr auto 1fr;
      align-items: center; gap: 12px;
      max-width: 1050px;
      padding: 14px 16px;
      border: 1px solid rgba(111, 224, 200, .22);
      border-radius: 15px;
      background: rgba(5, 18, 27, .42);
      color: var(--text-soft);
      font-size: 13px;
    }
    .branch-pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(91, 224, 197, .12);
      color: #a5f5e5;
      font-weight: 800;
      white-space: nowrap;
    }
    .independent-mark {
      color: var(--warn); font-weight: 900; letter-spacing: .06em;
      white-space: nowrap;
    }
    .kpi-grid {
      position: relative; z-index: 1;
      display: grid;
      grid-template-columns: repeat(6, minmax(110px, 1fr));
      gap: 10px;
      margin-top: 30px;
    }
    .kpi {
      padding: 16px 17px;
      min-height: 94px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(5, 15, 23, .42);
      backdrop-filter: blur(8px);
    }
    .kpi-value { font-size: 24px; font-weight: 820; letter-spacing: -.025em; }
    .kpi-label { margin-top: 3px; color: var(--muted); font-size: 12px; }
    .view-switcher {
      position: sticky; top: 10px; z-index: 60;
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 8px;
      width: min(720px, calc(100% - 24px));
      margin: 18px auto;
      padding: 7px;
      border: 1px solid var(--line-strong);
      border-radius: 16px;
      background: rgba(8, 20, 30, .94);
      box-shadow: 0 15px 38px rgba(0,0,0,.28);
      backdrop-filter: blur(18px);
    }
    .view-tab {
      min-height: 46px;
      border: 1px solid transparent;
      border-radius: 11px;
      color: var(--text-soft);
      background: transparent;
      font-weight: 820;
    }
    .view-tab:hover { color: #fff; background: rgba(255,255,255,.05); }
    .view-tab.active {
      color: #041817;
      border-color: rgba(91,224,197,.42);
      background: linear-gradient(110deg, var(--accent), #8be9d6);
      box-shadow: 0 8px 24px rgba(91,224,197,.18);
    }
    .blind-review-view { display: grid; gap: 16px; }
    .blind-intro { padding: 24px 26px; }
    .blind-intro-grid {
      display: grid; grid-template-columns: minmax(0,1fr) auto;
      gap: 18px; align-items: start;
    }
    .blind-intro h2 { margin: 5px 0 8px; font-size: clamp(24px, 3vw, 38px); }
    .blind-intro p { max-width: 980px; margin: 0; color: var(--muted); }
    .blind-progress {
      min-width: 122px;
      padding: 13px 15px;
      border: 1px solid rgba(91,224,197,.22);
      border-radius: 14px;
      background: rgba(91,224,197,.07);
      text-align: center;
    }
    .blind-progress strong { display: block; color: var(--accent); font-size: 24px; }
    .blind-progress span { color: var(--muted); font-size: 11px; }
    .blind-controls {
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 17px;
      background: rgba(12,25,37,.92);
    }
    .blind-controls label { color: var(--muted); font-size: 11px; font-weight: 800; }
    .blind-controls select { min-width: 210px; }
    .blind-size-group {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: rgba(255,255,255,.025);
    }
    .blind-size-button {
      min-width: 54px; padding: 7px 10px;
      border: 1px solid transparent;
      border-radius: 8px;
      color: var(--text-soft);
      background: transparent;
      font-size: 12px; font-weight: 800;
    }
    .blind-size-button.active { color: #041817; background: var(--accent); }
    .blind-control-spacer { flex: 1; }
    .blind-round-status { color: var(--faint); font-size: 11px; }
    .blind-groups-shell {
      overflow-x: auto;
      padding: 2px 2px 20px;
      scrollbar-color: rgba(91,224,197,.4) rgba(255,255,255,.04);
    }
    .blind-group-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(320px, 1fr));
      gap: 12px;
      min-width: calc(5 * 320px + 48px);
      align-items: start;
    }
    .blind-group {
      --blind-color: var(--accent);
      overflow: hidden;
      border: 1px solid var(--line);
      border-top: 4px solid var(--blind-color);
      border-radius: 18px;
      background: rgba(13,27,40,.95);
      box-shadow: 0 18px 45px rgba(0,0,0,.18);
      transition: .18s ease;
    }
    .blind-group.selected {
      border-color: var(--blind-color);
      box-shadow: 0 0 0 2px color-mix(in srgb, var(--blind-color) 30%, transparent),
                  0 20px 50px rgba(0,0,0,.25);
    }
    .blind-group-head {
      position: sticky; top: 84px; z-index: 4;
      display: flex; align-items: center; justify-content: space-between; gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(13,27,40,.97);
      backdrop-filter: blur(14px);
    }
    .blind-group-title { margin: 0; font-size: 18px; }
    .blind-group-count { color: var(--muted); font-size: 11px; }
    .blind-sample-list { display: grid; gap: 9px; padding: 10px; }
    .blind-sample {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: linear-gradient(160deg, rgba(255,255,255,.035), rgba(255,255,255,.012));
    }
    .blind-sample-no {
      margin-bottom: 6px;
      color: var(--blind-color);
      font: 800 10px/1.3 ui-monospace, monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .blind-sample h4 {
      margin: 0 0 8px;
      color: #f4f8fb;
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    .blind-sample pre {
      max-height: 230px;
      margin: 0;
      padding: 9px;
      overflow: auto;
      border: 1px solid rgba(164,195,213,.12);
      border-radius: 9px;
      color: #c7d5df;
      background: rgba(2,10,16,.52);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-size: 10.5px;
      line-height: 1.55;
    }
    .blind-sample .ghost-button { margin-top: 8px; padding: 6px 8px; font-size: 10.5px; }
    .blind-group-foot {
      padding: 12px;
      border-top: 1px solid var(--line);
      background: rgba(255,255,255,.018);
    }
    .blind-select-button { width: 100%; }
    .blind-selection-summary {
      display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
      padding: 15px 17px;
      border: 1px solid var(--line);
      border-radius: 15px;
      background: rgba(12,25,37,.9);
    }
    .blind-selection-summary strong { color: var(--accent); }
    .blind-selection-summary .primary-button { margin-left: auto; }
    .reveal-panel { margin-top: 18px; overflow: hidden; }
    .reveal-empty { padding: 0 23px 20px; color: var(--muted); font-size: 12px; }
    .workspace {
      display: grid;
      grid-template-columns: var(--sidebar) minmax(0, 1fr);
      gap: 18px;
      margin-top: 18px;
      align-items: start;
    }
    .sidebar {
      position: sticky; top: 14px;
      max-height: calc(100vh - 28px);
      overflow: auto;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      background: rgba(12, 25, 37, .93);
      box-shadow: 0 18px 48px rgba(0,0,0,.18);
      backdrop-filter: blur(16px);
    }
    .side-head {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 12px;
    }
    .side-title { font-size: 12px; color: var(--muted); font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .side-count { color: var(--accent); font-size: 12px; font-weight: 800; }
    .topic-nav { display: grid; gap: 7px; }
    .topic-button {
      width: 100%;
      display: grid;
      grid-template-columns: 32px 1fr auto;
      align-items: center;
      gap: 9px;
      padding: 10px;
      border: 1px solid transparent;
      border-radius: 12px;
      color: var(--text-soft);
      background: transparent;
      text-align: left;
      transition: .18s ease;
    }
    .topic-button:hover { background: rgba(255,255,255,.035); border-color: var(--line); }
    .topic-button.active {
      color: #fff;
      border-color: rgba(91,224,197,.28);
      background: linear-gradient(110deg, rgba(91,224,197,.13), rgba(88,169,255,.06));
    }
    .topic-no {
      display: grid; place-items: center;
      width: 30px; height: 30px;
      border-radius: 9px;
      background: rgba(255,255,255,.05);
      color: var(--muted);
      font: 700 11px/1 ui-monospace, monospace;
    }
    .topic-button.active .topic-no { color: #061b19; background: var(--accent); }
    .topic-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 720; }
    .topic-progress { color: var(--faint); font-size: 11px; white-space: nowrap; }
    .side-divider { height: 1px; margin: 18px 0; background: var(--line); }
    .side-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .main { min-width: 0; }
    .panel {
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      background: var(--panel);
      box-shadow: 0 18px 48px rgba(0,0,0,.14);
      backdrop-filter: blur(14px);
    }
    .intent-panel { padding: 23px 25px; }
    .section-kicker { color: var(--accent); font-size: 11px; font-weight: 850; letter-spacing: .14em; text-transform: uppercase; }
    .section-title { margin: 5px 0 7px; font-size: clamp(22px, 2.5vw, 34px); letter-spacing: -.025em; }
    .section-copy { max-width: 1080px; margin-bottom: 0; color: var(--muted); }
    .criteria-grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(150px, 1fr));
      gap: 9px;
      margin-top: 18px;
    }
    .criterion {
      padding: 12px 13px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(255,255,255,.022);
    }
    .criterion strong { display: block; margin-bottom: 3px; font-size: 13px; }
    .criterion span { color: var(--muted); font-size: 11.5px; }
    .criterion.negative strong { color: #ff9cab; }
    .summary-panel { margin-top: 18px; overflow: hidden; }
    .summary-head {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
      padding: 20px 23px 12px;
    }
    .summary-head h2 { margin: 3px 0 0; font-size: 21px; }
    .summary-note { max-width: 680px; color: var(--muted); font-size: 12px; text-align: right; }
    .table-wrap { overflow-x: auto; padding: 0 16px 18px; }
    table { width: 100%; min-width: 940px; border-collapse: separate; border-spacing: 0 7px; }
    th {
      padding: 5px 11px;
      color: var(--faint);
      font-size: 10px;
      letter-spacing: .08em;
      text-transform: uppercase;
      text-align: left;
      white-space: nowrap;
    }
    td {
      padding: 11px;
      color: var(--text-soft);
      background: rgba(255,255,255,.026);
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      font-size: 12px;
    }
    td:first-child { border-left: 1px solid var(--line); border-radius: 11px 0 0 11px; }
    td:last-child { border-right: 1px solid var(--line); border-radius: 0 11px 11px 0; }
    .method-dot { display: inline-block; width: 9px; height: 9px; margin-right: 8px; border-radius: 50%; box-shadow: 0 0 14px currentColor; }
    .toolbar {
      position: sticky; top: 10px; z-index: 20;
      display: flex; flex-wrap: wrap; align-items: center; gap: 9px;
      margin: 18px 0;
      padding: 12px;
      border: 1px solid var(--line-strong);
      border-radius: 17px;
      background: rgba(8, 20, 30, .91);
      box-shadow: 0 15px 38px rgba(0,0,0,.25);
      backdrop-filter: blur(18px);
    }
    .control-group {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: rgba(255,255,255,.025);
    }
    .control-label { padding: 0 5px; color: var(--faint); font-size: 10px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
    .control-button, .ghost-button, .primary-button, .icon-button {
      border: 1px solid transparent;
      border-radius: 9px;
      background: transparent;
      color: var(--text-soft);
      font-size: 12px;
      font-weight: 760;
      transition: .16s ease;
    }
    .control-button { min-width: 38px; padding: 7px 9px; }
    .control-button:hover, .ghost-button:hover, .icon-button:hover { color: #fff; background: rgba(255,255,255,.06); }
    .control-button.active { color: #041817; background: var(--accent); box-shadow: 0 5px 18px rgba(91,224,197,.18); }
    .ghost-button { padding: 8px 11px; border-color: var(--line); background: rgba(255,255,255,.025); }
    .primary-button { padding: 9px 13px; color: #041817; background: var(--accent); }
    .primary-button:hover { filter: brightness(1.08); }
    .icon-button { display: inline-grid; place-items: center; width: 34px; height: 34px; border-color: var(--line); }
    select, input[type="search"], textarea {
      border: 1px solid var(--line);
      border-radius: 9px;
      outline: none;
      color: var(--text);
      background: #0b1925;
    }
    select { padding: 7px 30px 7px 9px; font-size: 12px; }
    input[type="search"] { width: min(260px, 28vw); padding: 8px 11px; font-size: 12px; }
    textarea { width: 100%; min-height: 72px; resize: vertical; padding: 9px 10px; font-size: 12px; line-height: 1.5; }
    input:focus, select:focus, textarea:focus { border-color: rgba(91,224,197,.55); box-shadow: 0 0 0 3px rgba(91,224,197,.08); }
    .toolbar-spacer { flex: 1; }
    .save-status { min-width: 72px; color: var(--faint); font-size: 11px; text-align: right; }
    .topic-panel { padding: 25px; margin-bottom: 18px; }
    .topic-heading {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 24px;
    }
    .topic-heading h2 { margin: 2px 0 5px; font-size: clamp(27px, 3vw, 43px); letter-spacing: -.035em; }
    .topic-id { color: var(--faint); font-size: 11px; }
    .topic-stats { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; max-width: 680px; }
    .stat-chip, .badge {
      display: inline-flex; align-items: center; gap: 5px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,.035);
      color: var(--text-soft);
      font-size: 11px;
      font-weight: 720;
      white-space: nowrap;
    }
    .stat-chip { padding: 6px 9px; }
    .badge { padding: 3px 7px; }
    .badge.good { color: #9af0cf; border-color: rgba(88,214,165,.27); background: rgba(88,214,165,.08); }
    .badge.bad { color: #ffa0af; border-color: rgba(255,113,136,.26); background: rgba(255,113,136,.08); }
    .badge.warn { color: #f6d38d; border-color: rgba(242,193,102,.25); background: rgba(242,193,102,.075); }
    .badge.info { color: #a9d4ff; border-color: rgba(88,169,255,.25); background: rgba(88,169,255,.075); }
    .query-branches {
      display: grid; grid-template-columns: minmax(0, .85fr) 40px minmax(0, 1.15fr);
      gap: 10px; align-items: stretch;
      margin-top: 19px;
    }
    .query-card {
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,.024);
    }
    .query-card h3 { margin: 0 0 8px; font-size: 13px; }
    .query-card p { margin: 0; color: var(--muted); font-size: 12px; }
    .branch-separator { display: grid; place-items: center; color: var(--warn); font-weight: 900; font-size: 11px; text-align: center; }
    .rule-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .rule-atom {
      padding: 4px 7px;
      border: 1px solid rgba(139,152,168,.27);
      border-radius: 7px;
      background: rgba(139,152,168,.08);
      color: #d5dce2;
      font: 600 11px/1.3 ui-monospace, monospace;
    }
    details.seed-details {
      margin-top: 13px; border-top: 1px solid var(--line);
      padding-top: 13px;
    }
    details > summary { cursor: pointer; color: var(--text-soft); font-size: 12px; font-weight: 760; }
    .seed-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 9px; margin-top: 10px; }
    .seed {
      min-width: 0; padding: 12px;
      border: 1px solid var(--line); border-radius: 12px;
      background: rgba(3,12,19,.42);
    }
    .seed-title { margin-bottom: 7px; color: #a9d4ff; font-size: 11px; font-weight: 800; }
    .seed pre { max-height: 260px; }
    .decision-panel {
      display: grid; grid-template-columns: minmax(220px, .55fr) 1fr;
      gap: 12px; margin-top: 13px;
    }
    .decision-box {
      padding: 12px; border: 1px solid var(--line); border-radius: 13px;
      background: rgba(255,255,255,.022);
    }
    .decision-box label { display: block; margin-bottom: 6px; color: var(--muted); font-size: 11px; font-weight: 750; }
    .decision-box select { width: 100%; }
    .comparison-shell {
      position: relative;
      overflow-x: auto;
      padding: 2px 2px 22px;
      scrollbar-color: rgba(91,224,197,.4) rgba(255,255,255,.04);
    }
    .method-grid {
      display: grid;
      grid-template-columns: repeat(var(--method-columns, 5), minmax(345px, 1fr));
      gap: 12px;
      min-width: calc(var(--method-columns, 5) * 345px + (var(--method-columns, 5) - 1) * 12px);
      align-items: start;
    }
    .method-panel {
      --method-color: var(--accent);
      min-width: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-top: 3px solid var(--method-color);
      border-radius: 18px;
      background: rgba(13, 27, 40, .94);
      box-shadow: 0 18px 45px rgba(0,0,0,.18);
    }
    .method-head {
      position: sticky; top: 75px; z-index: 5;
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(13,27,40,.96);
      backdrop-filter: blur(14px);
    }
    .method-title-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .method-label { margin: 0; font-size: 18px; letter-spacing: -.02em; }
    .method-family { margin-top: 2px; color: var(--faint); font-size: 10px; letter-spacing: .06em; text-transform: uppercase; }
    .method-rank-count {
      display: grid; place-items: center; min-width: 43px; height: 30px;
      padding: 0 7px; border-radius: 9px;
      color: var(--method-color); background: color-mix(in srgb, var(--method-color) 12%, transparent);
      font: 800 11px/1 ui-monospace, monospace;
    }
    .method-description { min-height: 55px; margin: 10px 0 0; color: var(--muted); font-size: 11.5px; }
    .method-metrics { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
    .set-review {
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.018);
    }
    .set-review-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 9px; }
    .set-review-title { font-size: 11px; font-weight: 820; color: var(--text-soft); }
    .set-review-help { color: var(--faint); font-size: 10px; }
    .set-scores { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
    .set-score { display: grid; grid-template-columns: 1fr 58px; align-items: center; gap: 6px; }
    .set-score label { color: var(--muted); font-size: 10.5px; }
    .set-score select { width: 58px; padding: 5px 4px; }
    .set-score.overall { grid-column: 1 / -1; }
    .set-note { margin-top: 8px; min-height: 54px; }
    .panel-actions { display: flex; gap: 6px; margin-top: 8px; }
    .panel-actions .ghost-button { flex: 1; padding: 6px 7px; font-size: 10.5px; }
    .result-list { display: grid; gap: 9px; padding: 10px; }
    .result-card {
      position: relative;
      min-width: 0;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: linear-gradient(160deg, rgba(255,255,255,.036), rgba(255,255,255,.012));
      transition: border-color .16s ease, transform .16s ease, background .16s ease;
    }
    .result-card:hover { transform: translateY(-1px); border-color: var(--line-strong); background: rgba(255,255,255,.04); }
    .result-card.in-basket { border-color: color-mix(in srgb, var(--method-color) 55%, var(--line)); box-shadow: inset 3px 0 0 var(--method-color); }
    .result-top { display: grid; grid-template-columns: 34px minmax(0,1fr); gap: 9px; align-items: start; }
    .rank {
      display: grid; place-items: center;
      width: 34px; height: 34px; border-radius: 10px;
      color: var(--method-color);
      background: color-mix(in srgb, var(--method-color) 10%, transparent);
      border: 1px solid color-mix(in srgb, var(--method-color) 22%, transparent);
      font: 850 12px/1 ui-monospace, monospace;
    }
    .doc-title {
      display: -webkit-box; overflow: hidden; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
      margin: 0;
      color: #f4f8fb;
      font-size: 13px;
      line-height: 1.38;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .doc-id { margin-top: 3px; color: var(--faint); font: 10px/1.3 ui-monospace, monospace; }
    .badges { display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0 8px; }
    pre.text-preview {
      width: 100%;
      max-height: 250px;
      overflow: auto;
      margin: 0;
      padding: 10px;
      border: 1px solid rgba(164,195,213,.12);
      border-radius: 10px;
      color: #c7d5df;
      background: rgba(2, 10, 16, .56);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      tab-size: 2;
      font-size: 10.5px;
      line-height: 1.55;
    }
    .text-preview.full { max-height: 430px; }
    .card-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
    .card-actions .ghost-button { padding: 6px 8px; font-size: 10.5px; }
    .basket-button.active { color: #061716; border-color: var(--method-color); background: var(--method-color); }
    .review-label { margin-top: 11px; color: var(--faint); font-size: 9.5px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    .binary-scores { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
    .binary-score {
      padding: 5px 7px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: rgba(255,255,255,.018);
      font-size: 9.5px;
      font-weight: 760;
    }
    .binary-score.positive[data-state="1"],
    .binary-score.negative[data-state="0"] {
      color: #071a15; border-color: var(--good); background: var(--good);
    }
    .binary-score.positive[data-state="0"],
    .binary-score.negative[data-state="1"] {
      color: #22060b; border-color: var(--bad); background: var(--bad);
    }
    .result-note { min-height: 48px; margin-top: 7px; }
    .evidence-details {
      margin-top: 9px;
      padding-top: 8px;
      border-top: 1px dashed var(--line);
    }
    .evidence-grid { display: grid; gap: 6px; margin-top: 7px; }
    .evidence-line { color: var(--muted); font-size: 10px; overflow-wrap: anywhere; }
    .feature-row { display: flex; flex-wrap: wrap; gap: 4px; }
    .feature {
      padding: 3px 5px; border-radius: 6px;
      background: rgba(88,169,255,.08); color: #a9d4ff;
      font: 9.5px/1.35 ui-monospace, monospace;
    }
    .empty-state {
      margin: 12px; padding: 24px 14px;
      border: 1px dashed var(--line-strong); border-radius: 13px;
      color: var(--muted); text-align: center; font-size: 12px;
    }
    .blind-banner {
      display: none;
      margin-bottom: 12px; padding: 11px 13px;
      border: 1px solid rgba(242,193,102,.29);
      border-radius: 13px;
      color: #f8d99d;
      background: rgba(242,193,102,.07);
      font-size: 12px;
    }
    .blind-banner.visible { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .modal-backdrop, .drawer-backdrop {
      position: fixed; inset: 0; z-index: 80;
      display: none;
      background: rgba(1, 7, 12, .74);
      backdrop-filter: blur(8px);
    }
    .modal-backdrop.open, .drawer-backdrop.open { display: block; }
    .reader {
      position: absolute;
      inset: 3vh max(18px, calc((100vw - 1240px) / 2));
      display: grid; grid-template-rows: auto minmax(0,1fr);
      overflow: hidden;
      border: 1px solid var(--line-strong);
      border-radius: 22px;
      background: #0a1722;
      box-shadow: 0 35px 100px rgba(0,0,0,.54);
    }
    .reader-head {
      display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
      padding: 17px 19px;
      border-bottom: 1px solid var(--line);
      background: #0e1d2a;
    }
    .reader-head h2 { margin: 0 0 4px; font-size: 18px; }
    .reader-meta { color: var(--muted); font-size: 11px; }
    .reader-actions { display: flex; gap: 7px; }
    .reader pre {
      margin: 0; padding: 24px;
      overflow: auto;
      color: #d2dee6;
      background: #07131d;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      tab-size: 2;
      font-size: 12px;
      line-height: 1.66;
    }
    .drawer {
      position: absolute; top: 0; right: 0; bottom: 0;
      width: min(620px, 95vw);
      display: grid; grid-template-rows: auto minmax(0,1fr) auto;
      border-left: 1px solid var(--line-strong);
      background: #0b1925;
      box-shadow: -30px 0 80px rgba(0,0,0,.43);
    }
    .drawer-head, .drawer-foot { padding: 17px; border-bottom: 1px solid var(--line); }
    .drawer-foot { border-top: 1px solid var(--line); border-bottom: 0; display: flex; gap: 8px; }
    .drawer-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .drawer-head h2 { margin: 0 0 3px; font-size: 21px; }
    .drawer-copy { color: var(--muted); font-size: 11px; }
    .basket-content { overflow: auto; padding: 13px; }
    .basket-method { margin-bottom: 14px; }
    .basket-method-head {
      display: flex; justify-content: space-between; gap: 8px;
      margin-bottom: 7px; padding: 0 3px;
      color: var(--text-soft); font-size: 12px; font-weight: 800;
    }
    .basket-item {
      padding: 11px; margin-bottom: 7px;
      border: 1px solid var(--line); border-radius: 12px;
      background: rgba(255,255,255,.025);
    }
    .basket-item-head { display: flex; justify-content: space-between; gap: 9px; }
    .basket-item-title { color: var(--text-soft); font-size: 11.5px; font-weight: 720; }
    .basket-note { min-height: 48px; margin-top: 7px; }
    .toast {
      position: fixed; z-index: 120; left: 50%; bottom: 24px;
      max-width: min(520px, calc(100vw - 28px));
      padding: 10px 14px;
      border: 1px solid rgba(91,224,197,.28);
      border-radius: 11px;
      color: #dffbf4;
      background: rgba(8, 29, 31, .96);
      box-shadow: 0 18px 50px rgba(0,0,0,.35);
      opacity: 0; transform: translate(-50%, 14px);
      pointer-events: none;
      transition: .2s ease;
      font-size: 12px;
    }
    .toast.show { opacity: 1; transform: translate(-50%, 0); }
    .hidden { display: none !important; }
    noscript {
      display: block; margin: 18px; padding: 15px; border: 1px solid #ff7188;
      color: #ffd4db; background: #301019; border-radius: 12px;
    }
    @media (max-width: 1180px) {
      .kpi-grid { grid-template-columns: repeat(3, 1fr); }
      .criteria-grid { grid-template-columns: repeat(3, 1fr); }
      .workspace { grid-template-columns: 1fr; }
      .sidebar {
        position: relative; top: 0; max-height: none;
        display: grid; grid-template-columns: auto minmax(0,1fr) auto; gap: 12px; align-items: center;
      }
      .side-head { margin: 0; }
      .topic-nav { display: flex; overflow-x: auto; padding-bottom: 4px; }
      .topic-button { min-width: 190px; }
      .side-divider, .side-actions { display: none; }
    }
    @media (max-width: 760px) {
      .shell { width: min(100% - 16px, 1920px); margin-top: 8px; }
      .hero { min-height: 0; padding: 25px 20px; border-radius: 21px; }
      .hero h1 { font-size: 36px; }
      .protocol-ribbon { grid-template-columns: 1fr; }
      .independent-mark { text-align: center; }
      .view-switcher { top: 4px; width: calc(100% - 8px); }
      .blind-intro { padding: 18px; }
      .blind-intro-grid { grid-template-columns: 1fr; }
      .blind-progress { width: 100%; }
      .blind-controls { align-items: stretch; }
      .blind-controls select { width: 100%; }
      .blind-control-spacer { display: none; }
      .blind-selection-summary .primary-button { width: 100%; margin-left: 0; }
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      .criteria-grid { grid-template-columns: 1fr 1fr; }
      .intent-panel, .topic-panel { padding: 18px; }
      .summary-head, .topic-heading { display: block; }
      .summary-note { margin-top: 8px; text-align: left; }
      .topic-stats { justify-content: flex-start; margin-top: 12px; }
      .query-branches { grid-template-columns: 1fr; }
      .branch-separator { min-height: 25px; }
      .decision-panel { grid-template-columns: 1fr; }
      .toolbar { top: 4px; }
      input[type="search"] { width: 100%; }
      .toolbar-spacer { display: none; }
      .reader { inset: 8px; border-radius: 16px; }
      .reader-head { padding: 12px; }
      .reader pre { padding: 15px; }
    }
  </style>
</head>
<body>
<noscript>此评审台需要 JavaScript 才能切换主题、保存人工标签并导出结果。</noscript>
<div class="shell">
  <nav class="view-switcher" aria-label="页面入口">
    <button class="view-tab active" id="view-blind-button" type="button">盲审</button>
    <button class="view-tab" id="view-final-button" type="button">最终结果</button>
  </nav>

  <section class="blind-review-view" id="blind-review-view">
    <section class="panel blind-intro">
      <div class="blind-intro-grid">
        <div>
          <div class="section-kicker">Simple blind review</div>
          <h2>只看匿名网页组，选出你认为最好的一组</h2>
          <p>
            对当前主题，五种方法分别从自己的 Top-20 中随机抽取相同数量的页面，
            匿名显示为候选组 A–E。这里不显示方法名称、原始排名、检索分数或查询提示；
            只需比较哪组页面最同主题、最有信息量、最适合组成长程任务素材。
          </p>
        </div>
        <div class="blind-progress">
          <strong id="blind-progress-value">0 / 9</strong>
          <span>已完成主题</span>
        </div>
      </div>
    </section>

    <div class="blind-controls">
      <label for="blind-topic-select">主题</label>
      <select id="blind-topic-select" aria-label="盲审主题"></select>
      <label>每个候选组随机抽取</label>
      <div class="blind-size-group" aria-label="随机抽样条数">
        <button class="blind-size-button" type="button" data-blind-sample-size="3">3 条</button>
        <button class="blind-size-button active" type="button" data-blind-sample-size="5">5 条</button>
        <button class="blind-size-button" type="button" data-blind-sample-size="10">10 条</button>
      </div>
      <div class="blind-control-spacer"></div>
      <span class="blind-round-status" id="blind-round-status"></span>
      <button class="ghost-button" id="blind-reshuffle" type="button">重新随机抽样</button>
      <button class="ghost-button" id="blind-next-topic" type="button">下一个未评主题</button>
    </div>

    <div class="blind-groups-shell">
      <div class="blind-group-grid" id="blind-group-grid"></div>
    </div>

    <div class="blind-selection-summary">
      <span>当前选择：</span>
      <strong id="blind-current-choice">尚未选择</strong>
      <span class="metadata">选择只记录匿名候选组；方法身份仅在“最终结果”中揭晓。</span>
      <button class="primary-button" id="blind-go-final" type="button">查看最终结果与揭晓</button>
    </div>
  </section>

  <section class="hidden" id="final-result-view">
    <header class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Eval 9 · Keyword vs SAE</div>
        <h1>关键词 vs SAE<br>多网页筛选与评审</h1>
        <p class="lead">
          目标不是找一篇“最像”的网页，而是比较 <strong>Keyword baseline 与四种 SAE</strong>：
          谁更能从同一 10k 候选池中组织出主题一致、信息互补、足以支撑长程 task 生成的多网页素材集合。
        </p>
      </div>
      <div class="protocol-ribbon">
        <span class="branch-pill">Keyword rule → 10k</span>
        <span>锁定字面规则，独立扫描与排序</span>
        <span class="independent-mark">互不依赖</span>
        <span><span class="branch-pill">Locked seeds → 4 SAE → 10k</span> 四种 SAE 共用 seed，但各自在自己的特征空间检索</span>
      </div>
      <div class="kpi-grid" id="kpi-grid"></div>
    </header>

    <section class="panel reveal-panel">
      <div class="summary-head">
        <div>
          <div class="section-kicker">Blind review reveal</div>
          <h2>盲审选择揭晓</h2>
        </div>
        <div class="summary-note">
          下表只揭晓已经完成选择的主题；未选择的主题保持空白。下方保留原有完整 Top‑K 结果和评审工具。
        </div>
      </div>
      <div class="table-wrap"><table>
        <thead><tr>
          <th>主题</th><th>抽样设置</th><th>盲选候选组</th><th>实际方法</th>
        </tr></thead>
        <tbody id="blind-reveal-body"></tbody>
      </table></div>
    </section>

  <div class="workspace">
    <aside class="sidebar">
      <div class="side-head">
        <div class="side-title">Review Topics</div>
        <div class="side-count" id="side-count"></div>
      </div>
      <nav class="topic-nav" id="topic-nav" aria-label="主题导航"></nav>
      <div class="side-divider"></div>
      <div class="side-actions">
        <button class="ghost-button" id="export-json-side">导出 JSON</button>
        <button class="ghost-button" id="export-csv-side">导出 CSV</button>
        <button class="ghost-button" id="import-button">导入评审</button>
        <button class="ghost-button" id="reset-button">清空评审</button>
      </div>
      <input class="hidden" id="import-file" type="file" accept=".json,application/json">
    </aside>

    <main class="main">
      <section class="panel intent-panel">
        <div class="section-kicker">Evaluation target</div>
        <h2 class="section-title">不要把“关键词命中率”误当成答案</h2>
        <p class="section-copy">
          SAE 未命中关键词可能是有价值的语义扩展，也可能是主题漂移；Keyword 命中可能精准，也可能只是字面共现。
          请同时评价单篇页面与整组 Top‑N，尤其关注这组网页能否共同覆盖背景、原理、实现、案例与排错。
        </p>
        <div class="criteria-grid">
          <div class="criterion"><strong>同主题</strong><span>页面是否真正围绕目标主题，而非仅出现术语。</span></div>
          <div class="criterion"><strong>长任务有用</strong><span>是否包含足够事实、步骤、代码或解释，可成为任务素材。</span></div>
          <div class="criterion"><strong>信息互补</strong><span>是否为素材集合增加新角度，而非重复已有内容。</span></div>
          <div class="criterion negative"><strong>重复 / 镜像</strong><span>是否与同组页面高度重复、转载或近似镜像。</span></div>
          <div class="criterion negative"><strong>模板噪声</strong><span>是否主要是导航、样板、无关代码或弱信息页面。</span></div>
        </div>
      </section>

      <section class="panel summary-panel" id="summary-panel">
        <div class="summary-head">
          <div>
            <div class="section-kicker">Human evidence</div>
            <h2>人工评审总览</h2>
          </div>
          <div class="summary-note" id="summary-note"></div>
        </div>
        <div class="table-wrap"><table>
          <thead><tr>
            <th>方法</th><th>完整评审</th><th>同主题率</th><th>长任务有用率</th>
            <th>互补率</th><th>重复率 ↓</th><th>噪声率 ↓</th><th>集合总分</th><th>主题胜出</th>
          </tr></thead>
          <tbody id="summary-body"></tbody>
        </table></div>
      </section>

      <div class="toolbar" aria-label="评审控制栏">
        <div class="control-group">
          <span class="control-label">Top</span>
          <button class="control-button" data-top-n="5">5</button>
          <button class="control-button active" data-top-n="10">10</button>
          <button class="control-button" data-top-n="20">20</button>
        </div>
        <button class="ghost-button hidden" id="blind-toggle">开启盲评 A–E</button>
        <button class="ghost-button hidden" id="reveal-button">揭盲当前主题</button>
        <select id="result-filter" aria-label="结果过滤">
          <option value="all">全部结果</option>
          <option value="semantic">只看 SAE 语义扩展</option>
          <option value="consensus">只看多方法共识</option>
        </select>
        <select id="text-mode" aria-label="正文显示方式">
          <option value="snippet">精炼片段</option>
          <option value="full">完整正文</option>
        </select>
        <input type="search" id="search-input" placeholder="搜索当前主题的标题 / 片段">
        <div class="toolbar-spacer"></div>
        <span class="save-status" id="save-status">本地自动保存</span>
        <button class="primary-button" id="basket-open">素材篮 · 0</button>
      </div>

      <div class="blind-banner hidden" id="blind-banner">
        <span>盲评已开启：算法名、颜色、cosine、关键词命中和 SAE 特征证据均已隐藏；A–E 映射每个主题不同。</span>
        <button class="ghost-button" id="blind-banner-reveal">完成本主题后揭盲</button>
      </div>

      <section class="panel topic-panel">
        <div class="topic-heading">
          <div>
            <div class="section-kicker">Current topic</div>
            <h2 id="topic-title"></h2>
            <div class="topic-id mono" id="topic-id"></div>
          </div>
          <div class="topic-stats" id="topic-stats"></div>
        </div>
        <div class="query-branches">
          <div class="query-card">
            <h3>Keyword 分支输入</h3>
            <p>只使用以下锁定规则，搜索完整 10k 候选池。</p>
            <div id="keyword-rule"></div>
          </div>
          <div class="branch-separator">≠<br>独立</div>
          <div class="query-card">
            <h3>四种 SAE 分支输入</h3>
            <p>四种 SAE 共用同一组独立 seed page；seed 不由 Keyword 结果挑选。</p>
            <div class="rule-row" id="seed-id-row"></div>
          </div>
        </div>
        <details class="seed-details">
          <summary>查看锁定的 SAE seed 正文（仅用于理解 query，不参与 Keyword 排序）</summary>
          <div class="seed-grid" id="seed-grid"></div>
        </details>
        <div class="decision-panel">
          <div class="decision-box">
            <label for="winner-select">本主题哪种方法最适合构造多网页素材包？</label>
            <select id="winner-select"></select>
          </div>
          <div class="decision-box">
            <label for="topic-note">主题级结论 / 失败模式 / 推荐方法</label>
            <textarea id="topic-note" placeholder="例如：Cross 前列主题更集中；Keyword 精准但缺少原理类页面；Mean 扩展过宽……"></textarea>
          </div>
        </div>
      </section>

      <div class="comparison-shell">
        <div class="method-grid" id="method-grid"></div>
      </div>
    </main>
  </div>
  </section>
</div>

<div class="modal-backdrop" id="reader-modal" role="dialog" aria-modal="true" aria-label="完整正文阅读器">
  <div class="reader">
    <div class="reader-head">
      <div><h2 id="reader-title"></h2><div class="reader-meta" id="reader-meta"></div></div>
      <div class="reader-actions">
        <button class="ghost-button" id="reader-copy">复制正文</button>
        <button class="icon-button" id="reader-close" aria-label="关闭">×</button>
      </div>
    </div>
    <pre id="reader-text"></pre>
  </div>
</div>

<div class="drawer-backdrop" id="basket-drawer" role="dialog" aria-modal="true" aria-label="主题素材篮">
  <aside class="drawer">
    <div class="drawer-head">
      <div><h2>主题素材篮</h2><div class="drawer-copy" id="basket-copy"></div></div>
      <button class="icon-button" id="basket-close" aria-label="关闭">×</button>
    </div>
    <div class="basket-content" id="basket-content"></div>
    <div class="drawer-foot">
      <button class="primary-button" id="basket-export">导出当前主题素材包</button>
      <button class="ghost-button" id="basket-clear">清空当前主题</button>
    </div>
  </aside>
</div>

<div class="toast" id="toast"></div>

<script id="eval9-data" type="application/json">__EVAL9_DATA__</script>
<script>
(() => {
  "use strict";

  const DATA = JSON.parse(document.getElementById("eval9-data").textContent);
  const METHOD_ORDER = DATA.method_order;
  const TOPIC_MAP = Object.fromEntries(DATA.topics.map(topic => [topic.query_id, topic]));
  const STORE_KEY = `eval9-review-v2:${DATA.meta.demo_digest}`;
  const LETTER_COLORS = {A:"#73d9c2", B:"#79b9ff", C:"#b694ff", D:"#f2ba68", E:"#f17a91"};
  const DOC_CRITERIA = [
    {id:"relevant", label:"同主题", positive:true},
    {id:"useful", label:"长任务有用", positive:true},
    {id:"complementary", label:"信息互补", positive:true},
    {id:"duplicate", label:"重复/镜像", positive:false},
    {id:"noise", label:"模板噪声", positive:false}
  ];
  const SET_CRITERIA = [
    {id:"cohesion", label:"主题凝聚"},
    {id:"task_support", label:"长任务支撑"},
    {id:"complementarity", label:"集合互补"},
    {id:"low_redundancy", label:"低冗余"},
    {id:"overall", label:"总体推荐"}
  ];

  const $ = id => document.getElementById(id);
  const dom = {
    kpis: $("kpi-grid"),
    viewBlindButton: $("view-blind-button"),
    viewFinalButton: $("view-final-button"),
    blindReviewView: $("blind-review-view"),
    finalResultView: $("final-result-view"),
    blindTopicSelect: $("blind-topic-select"),
    blindGroupGrid: $("blind-group-grid"),
    blindProgressValue: $("blind-progress-value"),
    blindRoundStatus: $("blind-round-status"),
    blindCurrentChoice: $("blind-current-choice"),
    blindReshuffle: $("blind-reshuffle"),
    blindNextTopic: $("blind-next-topic"),
    blindGoFinal: $("blind-go-final"),
    blindRevealBody: $("blind-reveal-body"),
    sideCount: $("side-count"),
    topicNav: $("topic-nav"),
    summaryPanel: $("summary-panel"),
    summaryBody: $("summary-body"),
    summaryNote: $("summary-note"),
    blindToggle: $("blind-toggle"),
    revealButton: $("reveal-button"),
    filter: $("result-filter"),
    textMode: $("text-mode"),
    search: $("search-input"),
    saveStatus: $("save-status"),
    basketOpen: $("basket-open"),
    blindBanner: $("blind-banner"),
    blindBannerReveal: $("blind-banner-reveal"),
    topicTitle: $("topic-title"),
    topicId: $("topic-id"),
    topicStats: $("topic-stats"),
    keywordRule: $("keyword-rule"),
    seedIds: $("seed-id-row"),
    seedGrid: $("seed-grid"),
    winnerSelect: $("winner-select"),
    topicNote: $("topic-note"),
    methodGrid: $("method-grid"),
    readerModal: $("reader-modal"),
    readerTitle: $("reader-title"),
    readerMeta: $("reader-meta"),
    readerText: $("reader-text"),
    readerCopy: $("reader-copy"),
    readerClose: $("reader-close"),
    basketDrawer: $("basket-drawer"),
    basketCopy: $("basket-copy"),
    basketContent: $("basket-content"),
    basketClose: $("basket-close"),
    basketExport: $("basket-export"),
    basketClear: $("basket-clear"),
    toast: $("toast"),
    importFile: $("import-file")
  };

  function blankReview() {
    return {
      format: "eval9-human-review-state-v2",
      demo_digest: DATA.meta.demo_digest,
      document_ratings: {},
      set_ratings: {},
      topic_winners: {},
      topic_notes: {},
      blind_reviews: {},
      blind_rounds: {},
      basket: {}
    };
  }

  function loadReview() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return blankReview();
      const parsed = JSON.parse(raw);
      if (parsed.demo_digest !== DATA.meta.demo_digest) return blankReview();
      return Object.assign(blankReview(), parsed);
    } catch (error) {
      console.warn("Unable to load review state", error);
      return blankReview();
    }
  }

  let review = loadReview();
  let ui = {
    view: "blind",
    blindTopicId: DATA.topics[0].query_id,
    blindSampleSize: 5,
    topicId: DATA.topics[0].query_id,
    topN: 10,
    blind: false,
    revealedTopics: {},
    filter: "all",
    textMode: "snippet",
    search: ""
  };
  let readerContext = null;
  let toastTimer = null;
  let saveTimer = null;

  function create(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function append(parent, ...children) {
    for (const child of children) {
      if (child) parent.appendChild(child);
    }
    return parent;
  }

  function formatInteger(value) {
    return Number(value).toLocaleString("en-US");
  }

  function formatPercent(value, denominator) {
    if (!denominator) return "—";
    return `${(100 * value / denominator).toFixed(1)}%`;
  }

  function formatScore(value) {
    return value === null || value === undefined ? "—" : Number(value).toFixed(3);
  }

  function truncate(value, length) {
    const text = String(value || "");
    return text.length > length ? `${text.slice(0, length)}…` : text;
  }

  function blindReviewTopic() {
    return TOPIC_MAP[ui.blindTopicId];
  }

  function blindRoundKey(topicId, sampleSize) {
    return `${topicId}|${sampleSize}`;
  }

  function currentBlindRound(topicId = ui.blindTopicId, sampleSize = ui.blindSampleSize) {
    return Number(review.blind_rounds[blindRoundKey(topicId, sampleSize)] || 0);
  }

  function hashString(value) {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
      hash ^= value.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function effectiveBlindSampleSize(topic = blindReviewTopic()) {
    return Math.min(
      ui.blindSampleSize,
      ...METHOD_ORDER.map(methodId => topic.methods[methodId].results.length)
    );
  }

  function blindSampleRows(topic, methodId) {
    const rows = [...topic.methods[methodId].results];
    const round = currentBlindRound(topic.query_id, ui.blindSampleSize);
    const random = seededRandom(hashString(
      `${DATA.meta.demo_digest}|${topic.query_id}|${methodId}|${ui.blindSampleSize}|${round}`
    ));
    for (let index = rows.length - 1; index > 0; index -= 1) {
      const swapIndex = Math.floor(random() * (index + 1));
      [rows[index], rows[swapIndex]] = [rows[swapIndex], rows[index]];
    }
    return rows.slice(0, effectiveBlindSampleSize(topic));
  }

  function currentBlindChoice(topicId = ui.blindTopicId) {
    const choice = review.blind_reviews[topicId];
    if (!choice) return null;
    const round = currentBlindRound(topicId, ui.blindSampleSize);
    if (
      Number(choice.sample_size) !== ui.blindSampleSize ||
      Number(choice.round) !== round
    ) {
      return null;
    }
    return choice;
  }

  function currentTopic() {
    return TOPIC_MAP[ui.topicId];
  }

  function isConcealed() {
    return ui.blind && !ui.revealedTopics[ui.topicId];
  }

  function blindLetter(methodId, topicId = ui.topicId) {
    return DATA.blind_map[topicId].method_to_blind[methodId];
  }

  function methodLabel(methodId, topicId = ui.topicId) {
    if (!ui.blind) return DATA.methods[methodId].name;
    const letter = blindLetter(methodId, topicId);
    return ui.revealedTopics[topicId]
      ? `${letter} · ${DATA.methods[methodId].name}`
      : `候选组 ${letter}`;
  }

  function methodColor(methodId) {
    if (!ui.blind) return DATA.methods[methodId].color;
    return LETTER_COLORS[blindLetter(methodId)];
  }

  function orderedMethods(topicId = ui.topicId) {
    if (!ui.blind) return [...METHOD_ORDER];
    const mapping = DATA.blind_map[topicId].blind_to_method;
    return ["A", "B", "C", "D", "E"].map(letter => mapping[letter]);
  }

  function ratingKey(topicId, methodId, documentIndex) {
    return `${topicId}|${methodId}|${documentIndex}`;
  }

  function setKey(topicId, methodId) {
    return `${topicId}|${methodId}`;
  }

  function ensureDocumentRating(topicId, methodId, documentIndex) {
    const key = ratingKey(topicId, methodId, documentIndex);
    if (!review.document_ratings[key]) {
      review.document_ratings[key] = {
        topic_id: topicId,
        method_id: methodId,
        document_index: documentIndex,
        values: {},
        note: ""
      };
    }
    return review.document_ratings[key];
  }

  function ensureSetRating(topicId, methodId) {
    const key = setKey(topicId, methodId);
    if (!review.set_ratings[key]) {
      review.set_ratings[key] = {
        topic_id: topicId,
        method_id: methodId,
        scores: {},
        note: ""
      };
    }
    return review.set_ratings[key];
  }

  function saveReview() {
    clearTimeout(saveTimer);
    dom.saveStatus.textContent = "保存中…";
    saveTimer = setTimeout(() => {
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify(review));
        dom.saveStatus.textContent = "已本地保存";
      } catch (error) {
        console.warn("Unable to save review state", error);
        dom.saveStatus.textContent = "本地保存失败";
      }
      renderSidebar();
      renderSummary();
      renderBlindProgress();
      renderBlindReveal();
    }, 120);
  }

  function showToast(message) {
    dom.toast.textContent = message;
    dom.toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => dom.toast.classList.remove("show"), 2200);
  }

  function addBadge(parent, text, tone) {
    const badge = create("span", `badge${tone ? ` ${tone}` : ""}`, text);
    parent.appendChild(badge);
    return badge;
  }

  function renderKpis() {
    const items = [
      [formatInteger(DATA.meta.candidate_documents), "同一候选网页池"],
      [String(DATA.meta.topics), "预注册技术主题"],
      ["1 + 4", "Keyword + SAE"],
      [formatInteger(DATA.meta.document_windows), "模型处理窗口"],
      [formatInteger(DATA.meta.processed_tokens), "有效 tokens"],
      [String(DATA.meta.independent_seeds), "独立锁定 seeds"]
    ];
    dom.kpis.replaceChildren();
    for (const [value, label] of items) {
      const card = create("div", "kpi");
      append(card, create("div", "kpi-value", value), create("div", "kpi-label", label));
      dom.kpis.appendChild(card);
    }
  }

  function setView(view, {scroll = true} = {}) {
    ui.view = view === "final" ? "final" : "blind";
    const blind = ui.view === "blind";
    if (blind) ui.blindTopicId = ui.topicId;
    else {
      ui.topicId = ui.blindTopicId;
      ui.blind = false;
    }
    dom.blindReviewView.classList.toggle("hidden", !blind);
    dom.finalResultView.classList.toggle("hidden", blind);
    dom.viewBlindButton.classList.toggle("active", blind);
    dom.viewFinalButton.classList.toggle("active", !blind);
    if (blind) renderBlindReview();
    else {
      renderBlindReveal();
      renderAll();
    }
    if (scroll) {
      document.querySelector(".view-switcher").scrollIntoView({
        behavior: "smooth",
        block: "start"
      });
    }
  }

  function renderBlindTopicSelect() {
    dom.blindTopicSelect.replaceChildren();
    DATA.topics.forEach((topic, index) => {
      const choice = review.blind_reviews[topic.query_id];
      const option = create(
        "option",
        "",
        `${String(index + 1).padStart(2, "0")} · ${topic.title}${choice ? " · 已选" : ""}`
      );
      option.value = topic.query_id;
      if (topic.query_id === ui.blindTopicId) option.selected = true;
      dom.blindTopicSelect.appendChild(option);
    });
  }

  function renderBlindProgress() {
    const complete = DATA.topics.filter(
      topic => Boolean(review.blind_reviews[topic.query_id])
    ).length;
    dom.blindProgressValue.textContent = `${complete} / ${DATA.topics.length}`;
  }

  function openBlindReader(topic, methodId, result, letter, sampleIndex) {
    const documentData = DATA.documents[String(result.document_index)];
    readerContext = {
      topicId: topic.query_id,
      methodId,
      documentIndex: result.document_index
    };
    dom.readerTitle.textContent = documentData.title;
    dom.readerMeta.textContent =
      `${topic.title} · 候选组 ${letter} · 样本 ${sampleIndex + 1} · 匿名盲审`;
    dom.readerText.textContent = documentData.text;
    dom.readerModal.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function chooseBlindGroup(topic, letter) {
    const round = currentBlindRound(topic.query_id, ui.blindSampleSize);
    const actualSampleSize = effectiveBlindSampleSize(topic);
    const mapping = DATA.blind_map[topic.query_id].blind_to_method;
    const sampledDocumentIndices = Object.fromEntries(
      ["A", "B", "C", "D", "E"].map(candidateLetter => {
        const candidateMethod = mapping[candidateLetter];
        return [
          candidateLetter,
          blindSampleRows(topic, candidateMethod).map(row => row.document_index)
        ];
      })
    );
    review.blind_reviews[topic.query_id] = {
      topic_id: topic.query_id,
      blind_letter: letter,
      sample_size: actualSampleSize,
      requested_sample_size: ui.blindSampleSize,
      round,
      sampled_document_indices: sampledDocumentIndices,
      selected_at: new Date().toISOString()
    };
    saveReview();
    renderBlindReview();
    showToast(`${topic.title} 已选择候选组 ${letter}`);
  }

  function renderBlindGroups() {
    const topic = blindReviewTopic();
    const mapping = DATA.blind_map[topic.query_id].blind_to_method;
    const selected = currentBlindChoice(topic.query_id);
    dom.blindGroupGrid.replaceChildren();
    for (const letter of ["A", "B", "C", "D", "E"]) {
      const methodId = mapping[letter];
      const rows = blindSampleRows(topic, methodId);
      const panel = create(
        "section",
        `blind-group${selected?.blind_letter === letter ? " selected" : ""}`
      );
      panel.style.setProperty("--blind-color", LETTER_COLORS[letter]);
      const head = create("div", "blind-group-head");
      append(
        head,
        create("h3", "blind-group-title", `候选组 ${letter}`),
        create("span", "blind-group-count", `${rows.length} 条随机样本`)
      );
      panel.appendChild(head);
      const list = create("div", "blind-sample-list");
      rows.forEach((result, sampleIndex) => {
        const documentData = DATA.documents[String(result.document_index)];
        const card = create("article", "blind-sample");
        card.style.setProperty("--blind-color", LETTER_COLORS[letter]);
        card.appendChild(create("div", "blind-sample-no", `sample ${sampleIndex + 1}`));
        card.appendChild(create("h4", "", documentData.title));
        card.appendChild(create("pre", "", result.snippet));
        const read = create("button", "ghost-button", "查看完整正文");
        read.type = "button";
        read.addEventListener("click", () => {
          openBlindReader(topic, methodId, result, letter, sampleIndex);
        });
        card.appendChild(read);
        list.appendChild(card);
      });
      panel.appendChild(list);
      const foot = create("div", "blind-group-foot");
      const choose = create(
        "button",
        `primary-button blind-select-button${selected?.blind_letter === letter ? " active" : ""}`,
        selected?.blind_letter === letter ? `已选择候选组 ${letter}` : `选择候选组 ${letter}`
      );
      choose.type = "button";
      choose.addEventListener("click", () => chooseBlindGroup(topic, letter));
      foot.appendChild(choose);
      panel.appendChild(foot);
      dom.blindGroupGrid.appendChild(panel);
    }
  }

  function renderBlindReview() {
    renderBlindTopicSelect();
    renderBlindProgress();
    document.querySelectorAll("[data-blind-sample-size]").forEach(button => {
      button.classList.toggle(
        "active",
        Number(button.dataset.blindSampleSize) === ui.blindSampleSize
      );
    });
    const round = currentBlindRound();
    const actualSampleSize = effectiveBlindSampleSize();
    dom.blindRoundStatus.textContent =
      `每组 ${actualSampleSize} 条 · 抽样轮次 ${round + 1}`;
    const choice = currentBlindChoice();
    const previousChoice = review.blind_reviews[ui.blindTopicId];
    dom.blindCurrentChoice.textContent = choice
      ? `候选组 ${choice.blind_letter}`
      : previousChoice
        ? `已有旧选择：随机 ${previousChoice.sample_size} 条 / 第 ${Number(previousChoice.round) + 1} 轮 / 候选组 ${previousChoice.blind_letter}；当前抽样尚未选择`
        : "尚未选择";
    renderBlindGroups();
  }

  function renderBlindReveal() {
    dom.blindRevealBody.replaceChildren();
    for (const topic of DATA.topics) {
      const choice = review.blind_reviews[topic.query_id];
      const tr = create("tr");
      tr.appendChild(create("td", "", topic.title));
      tr.appendChild(create(
        "td",
        "",
        choice ? `随机 ${choice.sample_size} 条 · 第 ${Number(choice.round) + 1} 轮` : "—"
      ));
      tr.appendChild(create(
        "td",
        "",
        choice ? `候选组 ${choice.blind_letter}` : "尚未选择"
      ));
      const methodCell = create("td");
      if (choice) {
        const methodId =
          DATA.blind_map[topic.query_id].blind_to_method[choice.blind_letter];
        const dot = create("span", "method-dot");
        dot.style.color = DATA.methods[methodId].color;
        dot.style.background = DATA.methods[methodId].color;
        append(
          methodCell,
          dot,
          document.createTextNode(DATA.methods[methodId].name)
        );
      } else {
        methodCell.textContent = "—";
      }
      tr.appendChild(methodCell);
      dom.blindRevealBody.appendChild(tr);
    }
  }

  function instanceFullyRated(topicId, methodId, documentIndex) {
    const row = review.document_ratings[ratingKey(topicId, methodId, documentIndex)];
    return Boolean(row && DOC_CRITERIA.every(item => row.values[item.id] === 0 || row.values[item.id] === 1));
  }

  function topicReviewProgress(topic) {
    let total = 0;
    let complete = 0;
    for (const methodId of METHOD_ORDER) {
      for (const result of topic.methods[methodId].results) {
        total += 1;
        if (instanceFullyRated(topic.query_id, methodId, result.document_index)) complete += 1;
      }
    }
    return {complete, total};
  }

  function renderSidebar() {
    dom.sideCount.textContent = `${DATA.topics.length} TOPICS`;
    dom.topicNav.replaceChildren();
    DATA.topics.forEach((topic, index) => {
      const progress = topicReviewProgress(topic);
      const button = create("button", `topic-button${topic.query_id === ui.topicId ? " active" : ""}`);
      button.type = "button";
      button.addEventListener("click", () => {
        ui.topicId = topic.query_id;
        ui.search = "";
        dom.search.value = "";
        closeBasket();
        renderAll();
        window.scrollTo({top: document.querySelector(".toolbar").offsetTop - 8, behavior: "smooth"});
      });
      append(
        button,
        create("span", "topic-no", String(index + 1).padStart(2, "0")),
        create("span", "topic-name", topic.title),
        create("span", "topic-progress", `${progress.complete}/${progress.total}`)
      );
      dom.topicNav.appendChild(button);
    });
  }

  function aggregateMethod(methodId) {
    const criteria = Object.fromEntries(DOC_CRITERIA.map(item => [item.id, {ones:0, rated:0}]));
    let complete = 0;
    let total = 0;
    const setScores = [];
    for (const topic of DATA.topics) {
      for (const result of topic.methods[methodId].results) {
        total += 1;
        const row = review.document_ratings[ratingKey(topic.query_id, methodId, result.document_index)];
        if (!row) continue;
        let all = true;
        for (const item of DOC_CRITERIA) {
          const value = row.values[item.id];
          if (value === 0 || value === 1) {
            criteria[item.id].rated += 1;
            criteria[item.id].ones += value;
          } else {
            all = false;
          }
        }
        if (all) complete += 1;
      }
      const setRow = review.set_ratings[setKey(topic.query_id, methodId)];
      const overall = setRow && Number(setRow.scores.overall);
      if (overall >= 1 && overall <= 5) setScores.push(overall);
    }
    const wins = Object.values(review.topic_winners).filter(value => value === methodId).length;
    return {
      criteria, complete, total, wins,
      setAverage: setScores.length ? setScores.reduce((a,b) => a+b, 0) / setScores.length : null,
      setCount: setScores.length
    };
  }

  function renderSummary() {
    dom.summaryPanel.classList.toggle("hidden", ui.blind);
    if (ui.blind) return;
    dom.summaryNote.textContent =
      "百分比只使用已明确标为 0/1 的条目；集合总分为各主题“总体推荐”1–5 分均值。自动关键词覆盖不计入胜负。";
    dom.summaryBody.replaceChildren();
    for (const methodId of METHOD_ORDER) {
      const aggregate = aggregateMethod(methodId);
      const tr = create("tr");
      const methodCell = create("td");
      const dot = create("span", "method-dot");
      dot.style.color = DATA.methods[methodId].color;
      dot.style.background = DATA.methods[methodId].color;
      append(methodCell, dot, document.createTextNode(DATA.methods[methodId].name));
      tr.appendChild(methodCell);
      const completeCell = create("td", "", `${aggregate.complete}/${aggregate.total}`);
      tr.appendChild(completeCell);
      for (const criterionId of ["relevant", "useful", "complementary", "duplicate", "noise"]) {
        const item = aggregate.criteria[criterionId];
        tr.appendChild(create("td", "", formatPercent(item.ones, item.rated)));
      }
      tr.appendChild(create(
        "td", "",
        aggregate.setAverage === null ? "—" : `${aggregate.setAverage.toFixed(2)} / 5 · n=${aggregate.setCount}`
      ));
      tr.appendChild(create("td", "", `${aggregate.wins} / ${DATA.topics.length}`));
      dom.summaryBody.appendChild(tr);
    }
  }

  function renderTopicOverview() {
    const topic = currentTopic();
    const concealed = isConcealed();
    dom.topicTitle.textContent = topic.title;
    dom.topicId.textContent = `query_id: ${topic.query_id}`;
    dom.topicStats.replaceChildren();
    addBadge(dom.topicStats, `${formatInteger(topic.candidate_count)} candidates`, "info");
    if (!concealed) {
      addBadge(dom.topicStats, `Keyword 全池命中 ${topic.methods.keyword.total_hits}`, "");
      for (const methodId of METHOD_ORDER.slice(1)) {
        const method = topic.methods[methodId];
        addBadge(
          dom.topicStats,
          `${DATA.methods[methodId].short} 字面命中 ${method.literal_hits}/${method.results.length}`,
          method.literal_hits >= 10 ? "good" : "warn"
        );
      }
    } else {
      addBadge(dom.topicStats, "盲评：诊断指标已隐藏", "warn");
    }

    dom.keywordRule.replaceChildren();
    for (const group of ["all", "any"]) {
      const atoms = topic.keyword_rule[group] || [];
      if (!atoms.length) continue;
      const row = create("div", "rule-row");
      row.appendChild(create("span", "badge", group === "all" ? "ALL" : "ANY"));
      atoms.forEach(atom => row.appendChild(create("span", "rule-atom", atom)));
      dom.keywordRule.appendChild(row);
    }
    if (!dom.keywordRule.childNodes.length) {
      dom.keywordRule.appendChild(create("span", "badge", "无规则"));
    }

    dom.seedIds.replaceChildren();
    topic.seed_ids.forEach(seedId => dom.seedIds.appendChild(create("span", "rule-atom", seedId)));
    dom.seedGrid.replaceChildren();
    for (const seed of topic.seeds) {
      const card = create("article", "seed");
      card.appendChild(create("div", "seed-title", `${seed.seed_id} · ${seed.source}`));
      const pre = create("pre", "text-preview", seed.text);
      card.appendChild(pre);
      dom.seedGrid.appendChild(card);
    }

    renderWinnerControl();
    dom.topicNote.value = review.topic_notes[topic.query_id] || "";
  }

  function renderWinnerControl() {
    const topic = currentTopic();
    const selected = review.topic_winners[topic.query_id] || "";
    dom.winnerSelect.replaceChildren();
    const empty = create("option", "", "尚未选择");
    empty.value = "";
    dom.winnerSelect.appendChild(empty);
    for (const methodId of orderedMethods()) {
      const option = create("option", "", methodLabel(methodId));
      option.value = methodId;
      if (methodId === selected) option.selected = true;
      dom.winnerSelect.appendChild(option);
    }
  }

  function consensusCounts(topic) {
    const counts = new Map();
    const topN = effectiveTopN(topic);
    for (const methodId of METHOD_ORDER) {
      for (const result of topic.methods[methodId].results.slice(0, topN)) {
        counts.set(result.document_index, (counts.get(result.document_index) || 0) + 1);
      }
    }
    return counts;
  }

  function effectiveTopN(topic = currentTopic()) {
    if (!isConcealed()) return ui.topN;
    return Math.min(
      ui.topN,
      ...METHOD_ORDER.map(methodId => topic.methods[methodId].results.length)
    );
  }

  function visibleResults(topic, methodId, counts) {
    let rows = topic.methods[methodId].results.slice(0, effectiveTopN(topic));
    if (ui.filter === "semantic") {
      if (methodId === "keyword") return [];
      rows = rows.filter(row => !row.contains_keyword);
    } else if (ui.filter === "consensus") {
      rows = rows.filter(row => (counts.get(row.document_index) || 0) >= 2);
    }
    const query = ui.search.trim().toLocaleLowerCase();
    if (query) {
      rows = rows.filter(row => {
        const document = DATA.documents[String(row.document_index)];
        return `${document.title}\n${row.snippet}`.toLocaleLowerCase().includes(query);
      });
    }
    return rows;
  }

  function scoreSelect(value, onChange) {
    const select = create("select");
    const blank = create("option", "", "—");
    blank.value = "";
    select.appendChild(blank);
    for (let score = 1; score <= 5; score += 1) {
      const option = create("option", "", String(score));
      option.value = String(score);
      if (Number(value) === score) option.selected = true;
      select.appendChild(option);
    }
    select.addEventListener("change", () => onChange(select.value ? Number(select.value) : null));
    return select;
  }

  function renderSetReview(topic, methodId) {
    const wrapper = create("div", "set-review");
    const head = create("div", "set-review-head");
    append(
      head,
      create("div", "set-review-title", `整组 Top-${effectiveTopN(topic)} 评分`),
      create("div", "set-review-help", "1 差 · 5 优")
    );
    wrapper.appendChild(head);
    const scoreGrid = create("div", "set-scores");
    const setRow = ensureSetRating(topic.query_id, methodId);
    for (const criterion of SET_CRITERIA) {
      const line = create("div", `set-score${criterion.id === "overall" ? " overall" : ""}`);
      line.appendChild(create("label", "", criterion.label));
      line.appendChild(scoreSelect(setRow.scores[criterion.id], value => {
        if (value === null) delete setRow.scores[criterion.id];
        else setRow.scores[criterion.id] = value;
        saveReview();
      }));
      scoreGrid.appendChild(line);
    }
    wrapper.appendChild(scoreGrid);
    const note = create("textarea", "set-note");
    note.placeholder = "记录本列的主题漂移、覆盖角度、重复模式与长任务适用性…";
    note.value = setRow.note || "";
    note.addEventListener("input", () => {
      setRow.note = note.value;
      saveReview();
    });
    wrapper.appendChild(note);
    const actions = create("div", "panel-actions");
    const addTopFive = create("button", "ghost-button", "前 5 条加入素材篮");
    addTopFive.type = "button";
    addTopFive.addEventListener("click", () => {
      topic.methods[methodId].results.slice(0, Math.min(5, effectiveTopN(topic))).forEach(result => {
        addBasketItem(topic.query_id, methodId, result.document_index);
      });
      saveReview();
      renderMethods();
      updateBasketButton();
      showToast(`${methodLabel(methodId)} 的前 5 条已加入素材篮`);
    });
    const clearMethod = create("button", "ghost-button", "清空本列素材");
    clearMethod.type = "button";
    clearMethod.addEventListener("click", () => {
      if (review.basket[topic.query_id]) {
        delete review.basket[topic.query_id][methodId];
        saveReview();
        renderMethods();
        updateBasketButton();
      }
    });
    append(actions, addTopFive, clearMethod);
    wrapper.appendChild(actions);
    return wrapper;
  }

  function renderMethodHeader(topic, methodId, visibleCount) {
    const concealed = isConcealed();
    const method = DATA.methods[methodId];
    const stats = topic.methods[methodId];
    const topN = effectiveTopN(topic);
    const header = create("div", "method-head");
    const titleRow = create("div", "method-title-row");
    const titleBox = create("div");
    titleBox.appendChild(create("h3", "method-label", methodLabel(methodId)));
    titleBox.appendChild(create(
      "div",
      "method-family",
      concealed ? "Blind candidate set" : method.family
    ));
    append(titleRow, titleBox, create("div", "method-rank-count", `${visibleCount}/${Math.min(topN, stats.results.length)}`));
    header.appendChild(titleRow);
    header.appendChild(create(
      "p",
      "method-description",
      concealed
        ? "原始排序保持不变。完成单篇与整组评分后再揭盲，避免算法先验影响判断。"
        : method.description
    ));
    const metrics = create("div", "method-metrics");
    if (concealed) {
      addBadge(metrics, `等长 Top-${Math.min(topN, stats.results.length)}`, "warn");
      addBadge(metrics, "证据字段隐藏", "");
    } else if (methodId === "keyword") {
      addBadge(metrics, `全池命中 ${stats.total_hits}`, "info");
      addBadge(metrics, `展示 ${Math.min(topN, stats.results.length)}`, "");
    } else {
      const topRows = stats.results.slice(0, topN);
      const hits = topRows.filter(row => row.contains_keyword).length;
      const mean = topRows.length
        ? topRows.reduce((sum, row) => sum + Number(row.similarity || 0), 0) / topRows.length
        : null;
      addBadge(metrics, `字面命中 ${hits}/${topRows.length}`, hits >= topRows.length / 2 ? "good" : "warn");
      addBadge(metrics, `语义扩展 ${topRows.length - hits}`, "info");
      addBadge(metrics, `本空间均值 ${formatScore(mean)}`, "");
    }
    header.appendChild(metrics);
    return header;
  }

  function basketHas(topicId, methodId, documentIndex) {
    return Boolean(
      review.basket[topicId] &&
      review.basket[topicId][methodId] &&
      review.basket[topicId][methodId][String(documentIndex)]
    );
  }

  function addBasketItem(topicId, methodId, documentIndex) {
    review.basket[topicId] ||= {};
    review.basket[topicId][methodId] ||= {};
    review.basket[topicId][methodId][String(documentIndex)] ||= {
      document_index: documentIndex,
      note: ""
    };
  }

  function toggleBasket(topicId, methodId, documentIndex) {
    if (basketHas(topicId, methodId, documentIndex)) {
      delete review.basket[topicId][methodId][String(documentIndex)];
      if (!Object.keys(review.basket[topicId][methodId]).length) {
        delete review.basket[topicId][methodId];
      }
    } else {
      addBasketItem(topicId, methodId, documentIndex);
    }
    saveReview();
    renderMethods();
    updateBasketButton();
  }

  function cycleBinary(button, rating, criterion) {
    const current = rating.values[criterion.id];
    const next = current === undefined ? 1 : current === 1 ? 0 : undefined;
    if (next === undefined) delete rating.values[criterion.id];
    else rating.values[criterion.id] = next;
    updateBinaryButton(button, rating.values[criterion.id], criterion);
    saveReview();
  }

  function updateBinaryButton(button, value, criterion) {
    button.dataset.state = value === undefined ? "" : String(value);
    const marker = value === undefined ? "—" : String(value);
    button.textContent = `${criterion.label} · ${marker}`;
    button.title = "点击循环：未评 → 1 → 0 → 未评";
  }

  function renderEvidence(methodId, result) {
    const details = create("details", "evidence-details");
    const summary = create("summary", "", methodId === "keyword" ? "查看关键词命中证据" : "查看 SAE 检索证据");
    details.appendChild(summary);
    const body = create("div", "evidence-grid");
    if (methodId === "keyword") {
      body.appendChild(create(
        "div", "evidence-line",
        `命中条款 ${result.matched_clause_count} · 总出现次数 ${result.occurrence_count}`
      ));
      body.appendChild(create(
        "div", "evidence-line",
        `命中 atoms：${result.matched_atoms.length ? result.matched_atoms.join(" · ") : "—"}`
      ));
    } else {
      if (result.matched_window) {
        const window = result.matched_window;
        body.appendChild(create(
          "div", "evidence-line",
          `最佳窗口 #${window.window_index} · tokens ${window.token_start}–${window.token_end} · window cosine ${formatScore(window.similarity)}`
        ));
      }
      body.appendChild(create(
        "div", "evidence-line",
        `页面 cosine ${formatScore(result.similarity)} · 字面规则 ${result.contains_keyword ? "命中" : "未命中"}`
      ));
      if (result.matched_atoms.length) {
        body.appendChild(create("div", "evidence-line", `命中 atoms：${result.matched_atoms.join(" · ")}`));
      }
      const features = create("div", "feature-row");
      for (const feature of result.shared_features) {
        features.appendChild(create(
          "span", "feature",
          `f${feature.feature_id} · +${Number(feature.cosine_contribution || 0).toFixed(4)}`
        ));
      }
      if (features.childNodes.length) body.appendChild(features);
    }
    details.appendChild(body);
    return details;
  }

  function openReader(topic, methodId, result) {
    const documentData = DATA.documents[String(result.document_index)];
    readerContext = {topicId: topic.query_id, methodId, documentIndex: result.document_index};
    dom.readerTitle.textContent = documentData.title;
    dom.readerMeta.textContent =
      `${topic.title} · ${methodLabel(methodId)} · rank ${result.rank} · document_index ${result.document_index} · ${formatInteger(documentData.characters)} chars`;
    dom.readerText.textContent = documentData.text;
    dom.readerModal.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeReader() {
    dom.readerModal.classList.remove("open");
    if (!dom.basketDrawer.classList.contains("open")) document.body.style.overflow = "";
    readerContext = null;
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      showToast("已复制到剪贴板");
    } catch (error) {
      const area = create("textarea");
      area.value = text;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
      showToast("已复制到剪贴板");
    }
  }

  function renderResultCard(topic, methodId, result, consensus) {
    const concealed = isConcealed();
    const documentData = DATA.documents[String(result.document_index)];
    const rating = ensureDocumentRating(topic.query_id, methodId, result.document_index);
    const inBasket = basketHas(topic.query_id, methodId, result.document_index);
    const card = create("article", `result-card${inBasket ? " in-basket" : ""}`);
    card.style.setProperty("--method-color", methodColor(methodId));

    const top = create("div", "result-top");
    top.appendChild(create("div", "rank", `#${result.rank}`));
    const titleBox = create("div");
    titleBox.appendChild(create("h4", "doc-title", documentData.title));
    titleBox.appendChild(create(
      "div", "doc-id",
      `doc ${result.document_index} · ${formatInteger(documentData.characters)} chars`
    ));
    top.appendChild(titleBox);
    card.appendChild(top);

    const badges = create("div", "badges");
    if (!concealed) {
      if (methodId === "keyword") {
        addBadge(badges, `${result.occurrence_count} occurrences`, "info");
      } else {
        addBadge(badges, `cos ${formatScore(result.similarity)}`, "info");
        addBadge(
          badges,
          result.contains_keyword ? "同时命中关键词" : "未命中关键词 · 语义扩展",
          result.contains_keyword ? "good" : "warn"
        );
      }
    }
    if (consensus >= 2) addBadge(badges, `${consensus}/5 方法共识`, "good");
    if (inBasket) addBadge(badges, "已入素材篮", "info");
    card.appendChild(badges);

    const previewText = ui.textMode === "full" ? documentData.text : result.snippet;
    card.appendChild(create("pre", `text-preview${ui.textMode === "full" ? " full" : ""}`, previewText));

    const actions = create("div", "card-actions");
    const read = create("button", "ghost-button", "打开完整正文");
    read.type = "button";
    read.addEventListener("click", () => openReader(topic, methodId, result));
    const basket = create("button", `ghost-button basket-button${inBasket ? " active" : ""}`, inBasket ? "移出素材篮" : "加入素材篮");
    basket.type = "button";
    basket.addEventListener("click", () => toggleBasket(topic.query_id, methodId, result.document_index));
    const copyId = create("button", "ghost-button", "复制文档 ID");
    copyId.type = "button";
    copyId.addEventListener("click", () => copyText(String(result.document_index)));
    append(actions, read, basket, copyId);
    card.appendChild(actions);

    card.appendChild(create("div", "review-label", "单篇 0/1 评审 · 点击循环 未评 → 1 → 0"));
    const binary = create("div", "binary-scores");
    for (const criterion of DOC_CRITERIA) {
      const button = create("button", `binary-score ${criterion.positive ? "positive" : "negative"}`);
      button.type = "button";
      updateBinaryButton(button, rating.values[criterion.id], criterion);
      button.addEventListener("click", () => cycleBinary(button, rating, criterion));
      binary.appendChild(button);
    }
    card.appendChild(binary);

    const note = create("textarea", "result-note");
    note.placeholder = "页面内容、互补角色或误召原因…";
    note.value = rating.note || "";
    note.addEventListener("input", () => {
      rating.note = note.value;
      saveReview();
    });
    card.appendChild(note);
    if (!concealed) card.appendChild(renderEvidence(methodId, result));
    return card;
  }

  function renderMethods() {
    const topic = currentTopic();
    const counts = consensusCounts(topic);
    const methods = orderedMethods().filter(methodId => !(ui.filter === "semantic" && methodId === "keyword"));
    dom.methodGrid.style.setProperty("--method-columns", String(methods.length));
    dom.methodGrid.replaceChildren();
    for (const methodId of methods) {
      const rows = visibleResults(topic, methodId, counts);
      const panel = create("section", "method-panel");
      panel.style.setProperty("--method-color", methodColor(methodId));
      panel.appendChild(renderMethodHeader(topic, methodId, rows.length));
      panel.appendChild(renderSetReview(topic, methodId));
      const list = create("div", "result-list");
      if (!rows.length) {
        list.appendChild(create("div", "empty-state", "当前过滤条件下没有结果。"));
      } else {
        for (const result of rows) {
          list.appendChild(renderResultCard(
            topic, methodId, result, counts.get(result.document_index) || 1
          ));
        }
      }
      panel.appendChild(list);
      dom.methodGrid.appendChild(panel);
    }
  }

  function updateBlindUi() {
    const concealed = isConcealed();
    dom.blindToggle.textContent = ui.blind ? "退出盲评" : "开启盲评 A–E";
    dom.revealButton.classList.toggle("hidden", !ui.blind || ui.revealedTopics[ui.topicId]);
    dom.blindBanner.classList.toggle("visible", concealed);
    dom.filter.querySelector('option[value="semantic"]').disabled = concealed;
    if (concealed && ui.filter === "semantic") {
      ui.filter = "all";
      dom.filter.value = "all";
    }
  }

  function basketCount(topicId = ui.topicId) {
    const topicBasket = review.basket[topicId] || {};
    return Object.values(topicBasket).reduce((sum, rows) => sum + Object.keys(rows).length, 0);
  }

  function updateBasketButton() {
    dom.basketOpen.textContent = `素材篮 · ${basketCount()}`;
  }

  function renderAll() {
    updateBlindUi();
    renderSidebar();
    renderSummary();
    renderTopicOverview();
    renderMethods();
    updateBasketButton();
    if (ui.view === "final") renderBlindReveal();
  }

  function revealCurrentTopic() {
    if (!ui.blind || ui.revealedTopics[ui.topicId]) return;
    const okay = window.confirm("确认揭盲当前主题？建议先完成单篇标签、整组评分和优胜方法选择。");
    if (!okay) return;
    ui.revealedTopics[ui.topicId] = true;
    renderAll();
    showToast("当前主题已揭盲；原始 A–E 顺序保持不变");
  }

  function openBasket() {
    renderBasket();
    dom.basketDrawer.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeBasket() {
    dom.basketDrawer.classList.remove("open");
    if (!dom.readerModal.classList.contains("open")) document.body.style.overflow = "";
  }

  function findResult(topic, methodId, documentIndex) {
    return topic.methods[methodId].results.find(row => row.document_index === Number(documentIndex));
  }

  function renderBasket() {
    const topic = currentTopic();
    const topicBasket = review.basket[topic.query_id] || {};
    dom.basketCopy.textContent =
      `${topic.title} · 按方法分别挑选多网页素材，便于比较每一路能否组成长程任务上下文。`;
    dom.basketContent.replaceChildren();
    let any = false;
    for (const methodId of orderedMethods()) {
      const rows = topicBasket[methodId] || {};
      const entries = Object.values(rows);
      if (!entries.length) continue;
      any = true;
      const section = create("section", "basket-method");
      const head = create("div", "basket-method-head");
      append(head, create("span", "", methodLabel(methodId)), create("span", "", `${entries.length} pages`));
      section.appendChild(head);
      entries.sort((a,b) => {
        const ra = findResult(topic, methodId, a.document_index);
        const rb = findResult(topic, methodId, b.document_index);
        return (ra?.rank || 999) - (rb?.rank || 999);
      });
      for (const entry of entries) {
        const result = findResult(topic, methodId, entry.document_index);
        const documentData = DATA.documents[String(entry.document_index)];
        const item = create("article", "basket-item");
        const itemHead = create("div", "basket-item-head");
        const title = create(
          "div", "basket-item-title",
          `#${result ? result.rank : "—"} · doc ${entry.document_index} · ${documentData.title}`
        );
        const remove = create("button", "icon-button", "×");
        remove.type = "button";
        remove.title = "移出素材篮";
        remove.addEventListener("click", () => {
          delete review.basket[topic.query_id][methodId][String(entry.document_index)];
          if (!Object.keys(review.basket[topic.query_id][methodId]).length) {
            delete review.basket[topic.query_id][methodId];
          }
          saveReview();
          renderBasket();
          renderMethods();
          updateBasketButton();
        });
        append(itemHead, title, remove);
        item.appendChild(itemHead);
        const note = create("textarea", "basket-note");
        note.placeholder = "这篇在素材包中的角色：背景 / 原理 / 实现 / 示例 / 排错 / 对比…";
        note.value = entry.note || "";
        note.addEventListener("input", () => {
          entry.note = note.value;
          saveReview();
        });
        item.appendChild(note);
        const read = create("button", "ghost-button", "阅读完整正文");
        read.type = "button";
        read.addEventListener("click", () => {
          closeBasket();
          openReader(topic, methodId, result || {
            document_index: entry.document_index,
            rank: "—"
          });
        });
        item.appendChild(read);
        section.appendChild(item);
      }
      dom.basketContent.appendChild(section);
    }
    if (!any) {
      dom.basketContent.appendChild(create(
        "div", "empty-state",
        "当前主题的素材篮为空。可在任一结果卡片中加入页面，或用“前 5 条加入素材篮”快速构造候选包。"
      ));
    }
  }

  function download(filename, mime, text) {
    const blob = new Blob([text], {type: mime});
    const url = URL.createObjectURL(blob);
    const link = create("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function blindReviewResults() {
    return DATA.topics.map(topic => {
      const choice = review.blind_reviews[topic.query_id];
      if (!choice) {
        return {
          topic_id: topic.query_id,
          topic_title: topic.title,
          reviewed: false
        };
      }
      const methodId =
        DATA.blind_map[topic.query_id].blind_to_method[choice.blind_letter];
      return {
        topic_id: topic.query_id,
        topic_title: topic.title,
        reviewed: true,
        sample_size: choice.sample_size,
        round: Number(choice.round) + 1,
        blind_letter: choice.blind_letter,
        selected_method_id: methodId,
        selected_method_name: DATA.methods[methodId].name,
        sampled_document_indices: choice.sampled_document_indices || null,
        selected_at: choice.selected_at || null
      };
    });
  }

  function exportPayload() {
    return {
      format: "eval9-code-page-human-review-export-v2",
      exported_at: new Date().toISOString(),
      experiment: DATA.meta,
      blind_review_results: blindReviewResults(),
      review
    };
  }

  function exportJson() {
    download(
      `eval9_human_review_${new Date().toISOString().slice(0,10)}.json`,
      "application/json;charset=utf-8",
      JSON.stringify(exportPayload(), null, 2)
    );
    showToast("人工评审 JSON 已导出");
  }

  function csvEscape(value) {
    const text = value === null || value === undefined ? "" : String(value);
    return `"${text.replaceAll('"', '""')}"`;
  }

  function exportCsv() {
    const header = [
      "topic_id","topic_title","method_id","method_name","blind_letter","rank","document_index",
      "contains_keyword","similarity","relevant","useful","complementary","duplicate","noise",
      "document_note","in_basket","basket_note","set_cohesion","set_task_support",
      "set_complementarity","set_low_redundancy","set_overall","set_note","topic_winner","topic_note",
      "blind_sample_size","blind_round","blind_letter","blind_selected_method_id",
      "blind_selected_method_name"
    ];
    const rows = [header];
    for (const topic of DATA.topics) {
      for (const methodId of METHOD_ORDER) {
        const setRow = review.set_ratings[setKey(topic.query_id, methodId)] || {scores:{},note:""};
        const blindChoice = review.blind_reviews[topic.query_id];
        const blindMethodId = blindChoice
          ? DATA.blind_map[topic.query_id].blind_to_method[blindChoice.blind_letter]
          : "";
        for (const result of topic.methods[methodId].results) {
          const docRow = review.document_ratings[ratingKey(topic.query_id, methodId, result.document_index)] || {values:{},note:""};
          const basketRow = review.basket[topic.query_id]?.[methodId]?.[String(result.document_index)];
          rows.push([
            topic.query_id, topic.title, methodId, DATA.methods[methodId].name,
            blindLetter(methodId, topic.query_id), result.rank, result.document_index,
            result.contains_keyword, result.similarity,
            docRow.values.relevant, docRow.values.useful, docRow.values.complementary,
            docRow.values.duplicate, docRow.values.noise, docRow.note,
            Boolean(basketRow), basketRow?.note || "",
            setRow.scores.cohesion, setRow.scores.task_support, setRow.scores.complementarity,
            setRow.scores.low_redundancy, setRow.scores.overall, setRow.note,
            review.topic_winners[topic.query_id] || "", review.topic_notes[topic.query_id] || "",
            blindChoice?.sample_size ?? "",
            blindChoice ? Number(blindChoice.round) + 1 : "",
            blindChoice?.blind_letter || "",
            blindMethodId,
            blindMethodId ? DATA.methods[blindMethodId].name : ""
          ]);
        }
      }
    }
    const csv = "\ufeff" + rows.map(row => row.map(csvEscape).join(",")).join("\r\n");
    download(
      `eval9_human_review_${new Date().toISOString().slice(0,10)}.csv`,
      "text/csv;charset=utf-8",
      csv
    );
    showToast("人工评审 CSV 已导出");
  }

  function exportBasket() {
    const topic = currentTopic();
    const topicBasket = review.basket[topic.query_id] || {};
    const methods = {};
    for (const methodId of METHOD_ORDER) {
      const entries = Object.values(topicBasket[methodId] || {});
      if (!entries.length) continue;
      methods[methodId] = {
        method_name: DATA.methods[methodId].name,
        blind_letter: blindLetter(methodId, topic.query_id),
        pages: entries.map(entry => {
          const result = findResult(topic, methodId, entry.document_index);
          const documentData = DATA.documents[String(entry.document_index)];
          return {
            rank: result?.rank ?? null,
            document_index: entry.document_index,
            title: documentData.title,
            full_text: documentData.text,
            basket_note: entry.note || "",
            result,
            human_rating: review.document_ratings[
              ratingKey(topic.query_id, methodId, entry.document_index)
            ] || null
          };
        })
      };
    }
    const payload = {
      format: "eval9-topic-material-basket-v1",
      exported_at: new Date().toISOString(),
      experiment_demo_digest: DATA.meta.demo_digest,
      topic: {query_id:topic.query_id, title:topic.title},
      methods
    };
    download(
      `eval9_material_basket_${topic.query_id}.json`,
      "application/json;charset=utf-8",
      JSON.stringify(payload, null, 2)
    );
    showToast("当前主题素材包已导出（含完整正文）");
  }

  function importReview(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result));
        const incoming = payload.review || payload;
        if (incoming.demo_digest !== DATA.meta.demo_digest) {
          throw new Error("demo digest 不一致，不能导入到本实验");
        }
        review = Object.assign(blankReview(), incoming);
        localStorage.setItem(STORE_KEY, JSON.stringify(review));
        renderBlindReview();
        renderBlindReveal();
        renderAll();
        showToast("人工评审已导入");
      } catch (error) {
        window.alert(`导入失败：${error.message}`);
      } finally {
        dom.importFile.value = "";
      }
    };
    reader.readAsText(file, "utf-8");
  }

  function bindEvents() {
    dom.viewBlindButton.addEventListener("click", () => setView("blind"));
    dom.viewFinalButton.addEventListener("click", () => setView("final"));
    dom.blindGoFinal.addEventListener("click", () => setView("final"));
    dom.blindTopicSelect.addEventListener("change", () => {
      ui.blindTopicId = dom.blindTopicSelect.value;
      renderBlindReview();
    });
    document.querySelectorAll("[data-blind-sample-size]").forEach(button => {
      button.addEventListener("click", () => {
        ui.blindSampleSize = Number(button.dataset.blindSampleSize);
        renderBlindReview();
      });
    });
    dom.blindReshuffle.addEventListener("click", () => {
      const key = blindRoundKey(ui.blindTopicId, ui.blindSampleSize);
      review.blind_rounds[key] = currentBlindRound() + 1;
      delete review.blind_reviews[ui.blindTopicId];
      saveReview();
      renderBlindReview();
      showToast("已重新随机抽样；请重新选择最佳候选组");
    });
    dom.blindNextTopic.addEventListener("click", () => {
      const currentIndex = DATA.topics.findIndex(
        topic => topic.query_id === ui.blindTopicId
      );
      let next = null;
      for (let offset = 1; offset <= DATA.topics.length; offset += 1) {
        const candidate = DATA.topics[
          (currentIndex + offset) % DATA.topics.length
        ];
        if (!review.blind_reviews[candidate.query_id]) {
          next = candidate;
          break;
        }
      }
      if (!next) {
        next = DATA.topics[(currentIndex + 1) % DATA.topics.length];
        showToast("所有主题都已选择，切换到下一个主题");
      }
      ui.blindTopicId = next.query_id;
      renderBlindReview();
      document.querySelector(".blind-controls").scrollIntoView({
        behavior: "smooth",
        block: "start"
      });
    });
    document.querySelectorAll("[data-top-n]").forEach(button => {
      button.addEventListener("click", () => {
        ui.topN = Number(button.dataset.topN);
        document.querySelectorAll("[data-top-n]").forEach(item => {
          item.classList.toggle("active", Number(item.dataset.topN) === ui.topN);
        });
        renderMethods();
      });
    });
    dom.filter.addEventListener("change", () => {
      ui.filter = dom.filter.value;
      renderMethods();
    });
    dom.textMode.addEventListener("change", () => {
      ui.textMode = dom.textMode.value;
      renderMethods();
    });
    let searchTimer = null;
    dom.search.addEventListener("input", () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        ui.search = dom.search.value;
        renderMethods();
      }, 160);
    });
    dom.winnerSelect.addEventListener("change", () => {
      if (dom.winnerSelect.value) review.topic_winners[ui.topicId] = dom.winnerSelect.value;
      else delete review.topic_winners[ui.topicId];
      saveReview();
    });
    dom.topicNote.addEventListener("input", () => {
      review.topic_notes[ui.topicId] = dom.topicNote.value;
      saveReview();
    });
    dom.basketOpen.addEventListener("click", openBasket);
    dom.basketClose.addEventListener("click", closeBasket);
    dom.basketDrawer.addEventListener("click", event => {
      if (event.target === dom.basketDrawer) closeBasket();
    });
    dom.basketExport.addEventListener("click", exportBasket);
    dom.basketClear.addEventListener("click", () => {
      if (!basketCount()) return;
      if (!window.confirm(`确认清空 ${currentTopic().title} 的全部素材篮选择？`)) return;
      delete review.basket[ui.topicId];
      saveReview();
      renderBasket();
      renderMethods();
      updateBasketButton();
    });
    dom.readerClose.addEventListener("click", closeReader);
    dom.readerModal.addEventListener("click", event => {
      if (event.target === dom.readerModal) closeReader();
    });
    dom.readerCopy.addEventListener("click", () => {
      if (readerContext) copyText(DATA.documents[String(readerContext.documentIndex)].text);
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        if (dom.readerModal.classList.contains("open")) closeReader();
        else if (dom.basketDrawer.classList.contains("open")) closeBasket();
      }
    });
    $("export-json-side").addEventListener("click", exportJson);
    $("export-csv-side").addEventListener("click", exportCsv);
    $("import-button").addEventListener("click", () => dom.importFile.click());
    dom.importFile.addEventListener("change", () => {
      if (dom.importFile.files && dom.importFile.files[0]) importReview(dom.importFile.files[0]);
    });
    $("reset-button").addEventListener("click", () => {
      if (!window.confirm("确认清空本浏览器中该实验的全部人工评分、备注和素材篮？")) return;
      review = blankReview();
      try { localStorage.removeItem(STORE_KEY); } catch (error) {}
      renderBlindReview();
      renderBlindReveal();
      renderAll();
      showToast("人工评审状态已清空");
    });
  }

  renderKpis();
  bindEvents();
  renderAll();
  setView("blind", {scroll:false});
})();
</script>
</body>
</html>
"""


def write_dashboard(output: Path, payload: dict[str, Any]) -> None:
    serialized = safe_script_json(payload)
    html_text = HTML_TEMPLATE.replace("__EVAL9_DATA__", serialized)
    if "__EVAL9_DATA__" in html_text:
        raise RuntimeError("Dashboard data placeholder was not fully replaced")

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(html_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def render_demo_readme(payload: dict[str, Any], entrypoint: str) -> str:
    lines = [
        "# Eval 9 Code Page Retrieval",
        "",
        f"- Candidate documents: {payload['meta']['candidate_documents']:,}",
        f"- Topics: {len(payload['topics'])}",
        "- Methods: Keyword, Token SAE, Temporal SAE, Mean-Chunk SAE, Cross-Chunk SAE",
        f"- Interactive review dashboard: `{entrypoint}`",
        "",
        f"`{entrypoint}` is a self-contained single file: all Top-20 retrieval",
        "results and the corresponding full page texts are embedded in it. It",
        "does not depend on relative page URLs, so its buttons work both over",
        "HTTP and when the HTML is opened directly with `file://`.",
        "",
        "The dashboard supports:",
        "",
        "- Keyword baseline versus four independent SAE retrieval branches;",
        "- a simplified blind-review entrance: each anonymous method group randomly",
        "  samples 3, 5, or 10 pages from its Top-20 and the reviewer selects one group;",
        "- a separate final-results entrance that reveals blind choices and retains",
        "  the complete Top-5 / Top-10 / Top-20 comparison interface;",
        "- semantic-expansion and multi-method-consensus filters;",
        "- full-text reading plus per-page and set-level human review;",
        "- a per-topic multi-page material basket with full-text JSON export;",
        "- review-state JSON/CSV export and JSON import.",
        "",
        "## Query coverage and lexical overlap of SAE Top-20",
        "",
        "| Topic | Keyword hits | Token | Temporal | Mean | Cross |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for topic in payload["topics"]:
        methods = topic["methods"]
        lines.append(
            f"| {topic['title']} | {methods['keyword']['total_hits']} | "
            f"{methods['token']['literal_hits']}/20 | "
            f"{methods['temporal']['literal_hits']}/20 | "
            f"{methods['mean']['literal_hits']}/20 | "
            f"{methods['cross']['literal_hits']}/20 |"
        )
    lines.extend(
        [
            "",
            "The keyword branch and the four SAE branches independently search the",
            "complete candidate pool. Keyword results do not select or filter SAE",
            "seeds or candidates.",
            "",
            "Keyword overlap is diagnostic only. A result can be topically relevant",
            "without containing the literal keyword, and a literal match can still be",
            "off-topic. Complete the anonymous random-sample blind review first, then",
            "use the final-results entrance to reveal methods and inspect all Top-K",
            "examples.",
            "",
        ]
    )
    return "\n".join(lines)


def remove_legacy_html(base: Path, output: Path) -> list[str]:
    """Remove obsolete split-page interfaces while preserving the new dashboard."""

    demo = base / "demo"
    output = output.resolve()
    removed: list[str] = []
    legacy_files = [base / "index.html", demo / "index.html"]
    legacy_files.extend(sorted(demo.glob("index.legacy*.html")))
    for path in legacy_files:
        if path.resolve() == output:
            continue
        if path.is_file():
            path.unlink()
            removed.append(str(path))

    legacy_pages = demo / "pages"
    if legacy_pages.exists() and legacy_pages.resolve() not in output.parents:
        shutil.rmtree(legacy_pages)
        removed.append(str(legacy_pages))
    return removed


def refresh_demo_manifest(base: Path, entrypoint: Path) -> dict[str, Any]:
    """Refresh the audit manifest after replacing split pages with one HTML file."""

    demo = base / "demo"
    manifest_path = demo / "retrieval_demo_manifest.json"
    manifest = load_json(manifest_path)
    relative_entrypoint = entrypoint.resolve().relative_to(demo.resolve()).as_posix()
    file_records: dict[str, dict[str, Any]] = {}
    for path in sorted(demo.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(demo)
        if any(part.startswith(".") for part in relative.parts):
            continue
        key = relative.as_posix()
        file_records[key] = {
            "bytes": path.stat().st_size,
            "path": key,
            "sha256": sha256_file(path),
        }

    manifest.pop("artifact_digest", None)
    # Retain the original format and identity so the retrieval builder can
    # safely reuse unchanged results instead of recreating legacy HTML pages.
    manifest["format"] = "chunk-saes-code-page-retrieval-demo-v1"
    manifest["complete"] = True
    manifest["files"] = file_records
    manifest["entrypoint"] = relative_entrypoint
    manifest["interface"] = {
        "format": "eval9-code-page-human-review-dashboard-v1",
        "self_contained": True,
        "http_required": False,
        "legacy_split_html_removed": True,
        "features": [
            "keyword-vs-sae",
            "random-sample-blind-review-3-5-10",
            "blind-choice-reveal",
            "final-top-k-results",
            "multi-page-basket",
            "full-text-reader",
            "json-csv-export",
        ],
    }
    manifest["artifact_digest"] = json_digest(manifest)
    write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    output = (
        args.output
        or (base / "demo" / DASHBOARD_FILENAME)
    ).resolve()
    if args.top_k not in {5, 10, 20}:
        raise ValueError("--top-k must be one of 5, 10, or 20")

    payload = build_payload(base, args.top_k)
    write_dashboard(output, payload)
    removed: list[str] = []
    manifest_digest = None
    demo = (base / "demo").resolve()
    if output.parent == demo:
        if not args.keep_legacy_html:
            removed = remove_legacy_html(base, output)
        write_text_atomic(
            demo / "README.md",
            render_demo_readme(payload, output.name),
        )
        manifest = refresh_demo_manifest(base, output)
        manifest_digest = manifest["artifact_digest"]
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "topics": len(payload["topics"]),
                "methods_per_topic": len(payload["method_order"]),
                "embedded_results": payload["meta"]["embedded_results"],
                "embedded_unique_documents": payload["meta"][
                    "embedded_unique_documents"
                ],
                "removed_legacy_paths": removed,
                "manifest_digest": manifest_digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
