import sys, re, collections, pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import MolWt
RDLogger.DisableLog('rdApp.*')
G = {'kg': 1e3, 'g': 1.0, 'mg': 1e-3}
M = {'mol': 1.0, 'mole': 1.0, 'moles': 1.0, 'mmol': 1e-3, 'mmole': 1e-3, 'mmoles': 1e-3}
PAIR = re.compile(r'\(\s*([\d.]+)\s*(kg|g|mg)\.?,?\s*([\d.]+)\s*(mol|moles?|mmol|mmoles?)\b', re.I)
t = pq.read_table(sys.argv[1]).to_pylist()
ratios = []; flagged = []
for row in t:
    r = reaction_pb2.Reaction.FromString(row['reaction']); s = r.notes.procedure_details
    for inp in r.inputs.values():
        for c in inp.components:
            name = next((i.value for i in c.identifiers if i.type == 6), None)  # NAME
            smi = next((i.value for i in c.identifiers if i.type == 2), None)   # SMILES
            if not name or not smi or len(name) < 4: continue
            k = s.find(name)
            if k < 0: continue
            m = PAIR.match(s[k + len(name):k + len(name) + 40].lstrip())
            mol = Chem.MolFromSmiles(smi)
            if not m or mol is None: continue
            try: g = float(m[1]) * G[m[2].lower()]; n = float(m[3]) * M[m[4].lower()]
            except ValueError: continue
            if n == 0: continue
            ratio = g / MolWt(mol) / n; ratios.append(ratio)
            if not 0.9 < ratio < 1.1: flagged.append((round(ratio, 2), name[:50], m[0], smi[:50]))
c = len(ratios)
print('checked components:', c, ' within 10%:', sum(0.9 < x < 1.1 for x in ratios), ' flagged:', len(flagged))
print('flagged ratio buckets:', collections.Counter(round(f[0], 0) for f in flagged).most_common(8))
for f in flagged[:12]: print(f)
