import sys, re, collections, pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2
t = pq.read_table(sys.argv[1]).to_pylist()
kinds = collections.Counter(); both_txt = 0; shown = 0
for row in t:
    r = reaction_pb2.Reaction.FromString(row['reaction'])
    for inp in r.inputs.values():
        for c in inp.components: kinds[c.amount.WhichOneof('kind')] += 1
    s = r.notes.procedure_details
    if re.search(r'\d\s*g\b[^)]{0,30}\d+\.?\d*\s*m?mol', s): both_txt += 1
    if shown < 2 and re.search(r'mmol', s): print(s[:600], '\n---'); shown += 1
print(kinds, 'procedures with "N g ... N mol/mmol" pattern:', both_txt, 'of', len(t))
