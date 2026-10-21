// FloodGuard Africa - flood-risk decision tree (exported from analysis/run_analysis.py as demo/model.json).
// Walks the tree: at each node compare one input to a threshold, go left if <= threshold, else right.
function predictRisk(model, stage, rain) {
  var inputs = { river_stage_m: stage, rainfall_mm: rain };
  var i = 0;
  while (!model.nodes[i].leaf) {
    var n = model.nodes[i];
    i = inputs[n.feature] <= n.threshold ? n.left : n.right;
  }
  return model.levels[model.nodes[i].class];
}
if (typeof module !== "undefined") module.exports = { predictRisk: predictRisk };
