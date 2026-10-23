"""Build the web page index.html from contrastive_generation.md (needs pypandoc_binary).

    python site/build.py

The page reuses the layout of edapinenut.github.io: its header, cover and name card, its section headings with
icons, and its footer. Every `##` section of the post becomes one such section.
"""
import re
import tempfile
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/EDAPINENUT/contrast_gen"
SITE = "https://edapinenut.github.io/contrast_gen"
DATE = "October 2026"
DESCRIPTION = ("MeanFlow, Distribution Matching Distillation and drifting models as one pull-push update: "
               "a few-step generator moves its samples toward a positive and away from a negative.")
ICONS = [("velocit", "fa-location-arrow"), ("simulation", "fa-road"), ("as pull", "fa-exchange"),
         ("meanflow and dmd", "fa-cubes"), ("design", "fa-sliders"), ("toy", "fa-flask"),
         ("drifting", "fa-random"), ("conclusion", "fa-flag-checkered"), ("appendix", "fa-paperclip"),
         ("code", "fa-code")]


def github_link(path):
    kind = "tree" if path.endswith("/") else "blob"
    return f"{REPO}/{kind}/main/{path.rstrip('/')}"


def icon(title):
    t = title.lower()
    return next((ic for key, ic in ICONS if key in t), "fa-bookmark")


def section(sid, title, content, ic):
    return (f'<div id="{sid}" class="section">\n  <h1 class="section-heading">\n'
            f'    <i class="fa {ic}" aria-hidden="true"></i>\n    {title}\n  </h1>\n'
            f'  <div class="section-content">\n{content.strip()}\n  </div>\n</div>\n')


def highlighting_css():
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write("$highlighting-css$")
    return pypandoc.convert_text("```python\nx = 1\n```", "html5", format="markdown",
                                 extra_args=["--standalone", f"--template={f.name}", "--syntax-highlighting=pygments",
                                             "--metadata", "title=x"])


def main():
    src = (ROOT / "contrastive_generation.md").read_text()
    # Drafting notes (planned figures, fact-check reminders) are HTML comments: keep them out of the page.
    src = re.sub(r"<!--.*?-->\n?", "", src, flags=re.S)
    title_line, body = src.split("\n", 1)
    title = title_line.lstrip("#").strip()
    # GitHub Pages serves no directory listings and no rendered notebooks, so code links go to the repository.
    body = re.sub(r"\]\(((?:demo|materials)/[^)]*)\)",
                  lambda m: f"]({m.group(1) if m.group(1).endswith(('.gif', '.png')) else github_link(m.group(1))})",
                  body)
    html = pypandoc.convert_text(body, "html5", format="markdown",
                                 extra_args=["--mathjax", "--syntax-highlighting=pygments"])
    html = re.sub(r"<hr\s*/?>\n?", "", html)
    intro, *chunks = re.split(r"(?=<h2 )", html)
    parts, toc = [], []
    for chunk in chunks:
        m = re.match(r'<h2 id="([^"]+)"[^>]*>(.*?)</h2>(.*)', chunk, flags=re.S)
        sid, sec_title, content = m.groups()
        toc.append(f'      <li><a href="#{sid}">{sec_title}</a></li>')
        parts.append(section(sid, sec_title, content, icon(sec_title)))
    contents = '<ul class="contents-list">\n' + "\n".join(toc) + "\n</ul>"
    page_body = (section("overview", "Overview", intro, "fa-id-card")
                 + section("contents", "Contents", contents, "fa-list-ul") + "".join(parts))
    page = (ROOT / "site" / "template.html").read_text()
    for key, val in dict(title=title, description=DESCRIPTION, date=DATE, repo=REPO, site=SITE,
                         highlighting_css=highlighting_css(), body=page_body).items():
        page = page.replace("{{" + key + "}}", val)
    (ROOT / "index.html").write_text(page)
    print(ROOT / "index.html", f"{len(page) // 1024} kB, {len(parts)} sections")


if __name__ == "__main__":
    main()
