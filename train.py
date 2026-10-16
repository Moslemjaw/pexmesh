"""Train the small edge model and the bigger cloud model."""
import joblib, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from common import generate

X, y = generate(40000, seed=1)
Xte, yte = generate(10000, seed=2)
np.save("models/Xte.npy", Xte); np.save("models/yte.npy", yte)
edge = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
cloud = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=1, random_state=0).fit(X, y)
for name, m in [("edge", edge), ("cloud", cloud)]:
    joblib.dump(m, f"models/{name}.joblib")
    p = m.predict_proba(Xte)[:, 1]
    print(f"{name}: accuracy={accuracy_score(yte, p > .5):.3f}  AUC={roc_auc_score(yte, p):.3f}")
print("failure rate in test set:", yte.mean().round(3))
