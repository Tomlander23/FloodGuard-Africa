# Analysis and live demo: how to run and deploy

## Re-run the analysis (from the repo root)
```
pip install -r requirements.txt scikit-learn matplotlib
python analysis/run_analysis.py     # metrics, 4 charts, demo/model.json
python demo/build_demo.py           # builds demo/index.html and docs/index.html
```
Open `docs/index.html` in any browser to test the demo locally. No server needed.

## Publish the live demo (free, GitHub Pages)
1. Commit and push the `docs/`, `demo/` and `analysis/` folders to `main`.
2. On GitHub: repo -> Settings -> Pages.
3. Source: "Deploy from a branch" -> Branch: `main` -> Folder: `/docs` -> Save.
4. Wait 1-2 minutes. The link will be `https://idris0001.github.io/FloodGuard-Africa/`.
5. Open the link on your phone and test it, then add it to the README and the pitch deck.

## What the numbers mean
- Accuracy is measured with leave-one-out cross-validation (each scenario scored by a model that never saw it).
- The page's model is trained on all 50 scenarios, so its "match" ticks are not the honest test; the 96% is.
- The test data is curated and cleanly separated by river stage. Present results as "pipeline validated", not "proven forecasting skill".
