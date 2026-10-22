"""Build the web page index.html from contrastive_generation.md (needs pypandoc_binary).

    python site/build.py
"""
import re
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/EDAPINENUT/contrast_gen.edapinenut.github.io"
DATE = "October 2026"
DESCRIPTION = ("MeanFlow, Distribution Matching Distillation and drifting models as one pull-push update: "
               "a few-step generator moves its samples toward a positive and away from a negative.")


def github_link(path):
    kind = "tree" if path.endswith("/") else "blob"
    return f"{REPO}/{kind}/main/{path.rstrip('/')}"


def main():
    src = (ROOT / "contrastive_generation.md").read_text()
    # Drafting notes (planned figures, fact-check reminders) are HTML comments: keep them out of the page.
    src = re.sub(r"<!--.*?-->\n?", "", src, flags=re.S)
    title_line, body = src.split("\n", 1)
    title = title_line.lstrip("#").strip()
    # GitHub Pages serves no directory listings and no rendered notebooks, so code links go to the repository.
    body = re.sub(r"\]\(((?:demo|materials)/[^)]*)\)",
                  lambda m: f"]({github_link(m.group(1)) if not m.group(1).endswith(('.gif', '.png')) else m.group(1)})",
                  body)
    html = pypandoc.convert_text(
        body, "html5", format="markdown",
        extra_args=["--standalone", f"--template={ROOT / 'site' / 'template.html'}", "--mathjax",
                    "--toc", "--toc-depth=2", "--syntax-highlighting=pygments",
                    "--metadata", f"title={title}", "--metadata", f"date={DATE}",
                    "--variable", f"repo={REPO}", "--variable", f"description={DESCRIPTION}"])
    (ROOT / "index.html").write_text(html)
    print(ROOT / "index.html", f"{len(html) // 1024} kB")


if __name__ == "__main__":
    main()
