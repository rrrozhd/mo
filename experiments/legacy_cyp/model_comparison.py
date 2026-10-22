import numpy as np, pandas as pd, time
from tdc.single_pred import ADME
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb, xgboost as xgb
RDLogger.DisableLog('rdApp.*')
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def feat(smis):
    X, ok = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        X.append(np.array(gen.GetFingerprint(m), dtype=np.uint8)); ok.append(i)
    return np.array(X), ok

MODELS = {
 'dummy':      lambda: DummyClassifier(strategy='prior'),
 'logreg':     lambda: LogisticRegression(max_iter=3000, C=1.0, solver='liblinear'),
 'rf':         lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0),
 'lightgbm':   lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, n_jobs=-1, random_state=0, verbose=-1),
 'xgboost':    lambda: xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7, n_jobs=-1, random_state=0, tree_method='hist', eval_metric='logloss'),
}
NAMES = ['CYP1A2_Veith','CYP2C9_Veith','CYP2C19_Veith','CYP2D6_Veith','CYP3A4_Veith']
out = open('ladder_results.csv','w'); out.write('enzyme,model,auc,ap,fit_s\n'); out.flush()
for n in NAMES:
    sp = ADME(name=n).get_split(method='scaffold', seed=42)
    Xtr, itr = feat(sp['train'].Drug.values); ytr = sp['train'].Y.values[itr]
    Xte, ite = feat(sp['test'].Drug.values);  yte = sp['test'].Y.values[ite]
    for mname, mk in MODELS.items():
        t0 = time.time(); m = mk(); m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:,1]
        try: a = roc_auc_score(yte, p)
        except Exception: a = float('nan')
        row = f"{n.split('_')[0]},{mname},{a:.4f},{average_precision_score(yte,p):.4f},{time.time()-t0:.1f}"
        out.write(row+'\n'); out.flush(); print(row, flush=True)
out.close()
df = pd.read_csv('ladder_results.csv')
print(df.pivot(index='enzyme', columns='model', values='auc').to_string())
