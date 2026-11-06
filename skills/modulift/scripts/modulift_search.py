"""
Run several modulift searches in one process and print the results as JSON.

Loading the dataset is the slow part (it is ~485k packages), so batch every query
you need into a single invocation instead of calling this script repeatedly.

Examples:
    python modulift_search.py --describe "parse and validate cron expressions" \
        --keywords cron scheduling --similar apscheduler --limit 8 --verify

    # Check names against the live PyPI index only (no dataset download):
    python modulift_search.py --verify-only croniter python-crontab
"""

import argparse
import contextlib
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PYPI_URL = "https://pypi.org/pypi/{name}/json"
USER_AGENT = "modulift-skill/1.0"


def pypi_info(name: str) -> dict:
    """Live facts about a package from the PyPI JSON API."""
    request = urllib.request.Request(
        PYPI_URL.format(name=name), headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"on_pypi": False}
        return {"on_pypi": None, "error": f"HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"on_pypi": None, "error": str(e)}

    info = data.get("info", {})
    files = data.get("urls") or []
    upload_times = [f.get("upload_time_iso_8601", "") for f in files]
    classifiers = info.get("classifiers") or []
    status = [
        c.split(" :: ", 2)[-1]
        for c in classifiers
        if c.startswith("Development Status")
    ]
    return {
        "on_pypi": True,
        "latest_version": info.get("version"),
        "latest_release": max(upload_times)[:10] if upload_times else None,
        "release_count": len(data.get("releases") or {}),
        "requires_python": info.get("requires_python") or None,
        "license": info.get("license_expression")
        or (info.get("license") or "")[:60]
        or None,
        "development_status": status[0] if status else None,
        "yanked": bool(info.get("yanked")),
        "summary": info.get("summary"),
        "project_urls": info.get("project_urls") or {},
    }


def verify(names):
    unique = list(dict.fromkeys(names))
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(unique, pool.map(pypi_info, unique)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--describe",
        action="append",
        default=[],
        metavar="TEXT",
        help="Describe the need in plain words (repeatable).",
    )
    parser.add_argument(
        "--describe-method", choices=["cosine", "fts"], default="cosine"
    )
    parser.add_argument(
        "--keywords",
        action="append",
        nargs="+",
        default=[],
        metavar="KW",
        help="One keyword set (repeatable).",
    )
    parser.add_argument("--relation", choices=["and", "or"], default="and")
    parser.add_argument(
        "--keyword-method", choices=["exact", "jaccard"], default="exact"
    )
    parser.add_argument(
        "--similar",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="Find packages similar to this one (repeatable).",
    )
    parser.add_argument(
        "--similar-method", choices=["cosine", "tf-idf", "jaccard"], default="cosine"
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="Look up a package in the dataset (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--verify", action="store_true", help="Add live PyPI facts for every result."
    )
    parser.add_argument(
        "--verify-only",
        nargs="+",
        metavar="PACKAGE",
        help="Only check these names on PyPI; skips the dataset.",
    )
    args = parser.parse_args(argv)

    if args.verify_only:
        print(json.dumps({"pypi": verify(args.verify_only)}, indent=2))
        return 0

    if not (args.describe or args.keywords or args.similar or args.name):
        parser.error("give at least one of --describe, --keywords, --similar, --name")

    try:
        import modulift as mf
    except ImportError:
        print(
            "modulift is not installed: run `pip install modulift` (or use `uv run --with modulift`).",
            file=sys.stderr,
        )
        return 2

    queries = []

    def run(kind, query, call):
        try:
            # modulift <= 0.1.0 printed notices to stdout; keep stdout clean so the
            # output stays valid JSON with any installed version.
            with contextlib.redirect_stdout(sys.stderr):
                results = call()
            queries.append({"kind": kind, "query": query, "results": results})
        except (ValueError, RuntimeError) as e:
            queries.append(
                {"kind": kind, "query": query, "error": str(e), "results": []}
            )

    for text in args.describe:
        run(
            f"describe:{args.describe_method}",
            text,
            lambda t=text: mf.search_by_description(
                t, limit=args.limit, method=args.describe_method
            ),
        )
    for kws in args.keywords:
        run(
            f"keywords:{args.keyword_method}:{args.relation}",
            kws,
            lambda k=kws: mf.search_by_keywords(
                *k, relation=args.relation, limit=args.limit, method=args.keyword_method
            ),
        )
    for package in args.similar:
        run(
            f"similar:{args.similar_method}",
            package,
            lambda p=package: mf.find_similar_packages(
                p, limit=args.limit, method=args.similar_method
            ),
        )
    for package in args.name:
        run("name", package, lambda p=package: [mf.search_by_package_name(p)])

    output = {"queries": queries}
    if args.verify:
        names = [r["package"] for q in queries for r in q["results"]]
        output["pypi"] = verify(names)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
