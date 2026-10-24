"""
Generate GitHub wiki pages from docs/ so the wiki and the documentation site share one source.

    python scripts/build_wiki.py            # writes ./wiki/
    git clone https://github.com/prerak-joshi/vasicek_3f.wiki.git
    cp wiki/*.md vasicek_3f.wiki/ && cd vasicek_3f.wiki && git add . && git commit -m "Update wiki" && git push

The GitHub wiki must be enabled and have at least one page (create "Home" once in the browser).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "wiki"
SITE = "https://prerak-joshi.github.io/vasicek_3f"

# docs file -> wiki page name (order = sidebar order)
PAGES = {
    "index.md": "Home",
    "getting-started.md": "Getting-Started",
    "methodology.md": "Methodology",
    "term-premium.md": "Interpreting-the-Term-Premium",
    "validation.md": "Validation",
    "dashboard.md": "Dashboard-Guide",
    "architecture.md": "Architecture",
    "faq.md": "FAQ",
}


def convert(text: str) -> str:
    # MkDocs-only syntax -> plain GitHub markdown
    text = re.sub(r"\)\{[^}]*\}", ")", text)  # attribute lists on links/buttons
    # Mermaid fences and $...$ / $$...$$ math render natively on GitHub wikis.

    def link(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        page, _, anchor = target.partition("#")
        if page in PAGES:
            return f"[{label}]({PAGES[page]}{'#' + anchor if anchor else ''})"
        if page == "api.md":
            return f"[{label}]({SITE}/api/)"
        if page == "changelog.md":
            return f"[{label}](https://github.com/prerak-joshi/vasicek_3f/blob/main/CHANGELOG.md)"
        return m.group(0)

    return re.sub(r"\[([^\]]+)\]\(([\w\-]+\.md(?:#[\w\-]+)?)\)", link, text)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for src, name in PAGES.items():
        body = convert((DOCS / src).read_text(encoding="utf-8"))
        note = f"\n\n---\n*Generated from `docs/{src}`. Full site with API reference: {SITE}*\n"
        (OUT / f"{name}.md").write_text(body + note, encoding="utf-8")
    sidebar = ["**vasicek3f**", ""] + [f"- [{n.replace('-', ' ')}]({n})" for n in PAGES.values()]
    sidebar += [
        "",
        f"- [API reference]({SITE}/api/)",
        "- [Live dashboard](https://vasicek3f-5ygmrt2fr8rvg97zfofzlq.streamlit.app)",
    ]
    (OUT / "_Sidebar.md").write_text("\n".join(sidebar) + "\n", encoding="utf-8")
    (OUT / "_Footer.md").write_text(
        "Educational project · MIT licence · not investment advice\n", encoding="utf-8"
    )
    print(f"wrote {len(PAGES) + 2} pages to {OUT}")


if __name__ == "__main__":
    main()
