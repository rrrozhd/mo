import numpy as np, pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
RDLogger.DisableLog('rdApp.*')

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def feat(smis):
    X, ok = [], []
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        fp = np.array(gen.GetFingerprint(m), dtype=np.int8)
        X.append(fp); ok.append(i)
    return np.array(X), ok

NAMES = ['CYP1A2_Veith','CYP2C9_Veith','CYP2C19_Veith','CYP2D6_Veith','CYP3A4_Veith']
rows = []
for n in NAMES:
    d = ADME(name=n)
    for split_kind in ['scaffold','random']:
        sp = d.get_split(method=split_kind, seed=42)
        tr, te = sp['train'], sp['test']
        Xtr, itr = feat(tr.Drug.values); ytr = tr.Y.values[itr]
        Xte, ite = feat(te.Drug.values); yte = te.Y.values[ite]
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, min_samples_leaf=2)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:,1]
        rows.append(dict(enzyme=n.split('_')[0], split=split_kind, n_train=len(ytr), n_test=len(yte),
                         pos_rate=round(float(yte.mean()),3), auc=round(roc_auc_score(yte,p),3),
                         ap=round(average_precision_score(yte,p),3)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv('feasibility_cyp.csv', index=False)
print(df.pivot(index='enzyme', columns='split', values='auc'))
