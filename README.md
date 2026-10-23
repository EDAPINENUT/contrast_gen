# Contrastive Generation

Blog post: **[Contrastive Generation: Training a Few-Step Generator by Pulling and Pushing Its Samples](https://edapinenut.github.io/contrast_gen/)**

- `contrastive_generation.md`: the post.
- `materials/`: its figures and animations (GIF, plus MP4 and SVG versions).
- `demo/`: the code. `contrastive_generation.ipynb` reruns every experiment of the post on CPU and rewrites `materials/`; `assess.py` holds the functions it calls.
- `index.html`, `site/`: the web page, built from the post with `python site/build.py` (needs `pypandoc_binary`).

```bash
pip install -r demo/requirements.txt
cd demo && jupyter nbconvert --to notebook --execute --inplace contrastive_generation.ipynb
```
