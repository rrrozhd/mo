import { useState } from "react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

const ENZYMES = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]

/** RDKit returns a whole SVG document; strip its prolog so it inlines cleanly. */
function Depiction({ svg, className = "" }: { svg: string; className?: string }) {
  return (
    <div
      className={`[&_svg]:h-auto [&_svg]:w-full overflow-hidden rounded-lg bg-white ${className}`}
      dangerouslySetInnerHTML={{ __html: svg.replace(/<\?xml[^>]*\?>/, "") }}
    />
  )
}

export function ScoreBars({ left, right, enzyme }: { left: any; right?: any; enzyme?: string }) {
  return (
    <div className="space-y-2">
      {ENZYMES.map(name => {
        const before = left?.enzymes?.[name]?.score ?? 0
        const after = right?.enzymes?.[name]?.score
        return (
          <div key={name} className="grid grid-cols-[6rem_minmax(0,1fr)_4.5rem] items-center gap-2 text-sm">
            <span className={name === enzyme ? "font-semibold" : "text-muted-foreground"}>
              {name}
            </span>
            <div className="space-y-1">
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-destructive/70" style={{ width: `${before * 100}%` }} />
              </div>
              {right && (
                <div className="h-2 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-chart-2" style={{ width: `${(after ?? 0) * 100}%` }} />
                </div>
              )}
            </div>
            <span className="text-right font-mono tabular-nums text-muted-foreground">
              {right ? `${before.toFixed(2)}→${(after ?? 0).toFixed(2)}` : before.toFixed(2)}
            </span>
            {(left?.enzymes?.[name]?.outside_training_distribution || right?.enzymes?.[name]?.outside_training_distribution) && (
              <p className="col-span-3 text-sm leading-relaxed text-destructive">{name}: outside training distribution</p>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function Neighbourhood({ data }: { data: any }) {
  const points: any[] = data.points ?? []
  if (!points.length) return null
  const xs = points.map(p => p.x)
  const ys = points.map(p => p.y)
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
  const place = (value: number, min: number, max: number) =>
    max === min ? 50 : 6 + (88 * (value - min)) / (max - min)
  return (
    <div className="space-y-2">
      <div className="relative aspect-[2/1] w-full rounded-lg border bg-card">
        {points.map((point, index) => (
          <span
            key={index}
            title={`${point.is_query ? "query" : point.measured_label ? "positive dataset label" : "negative dataset label"} · similarity ${point.similarity}`}
            className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full ${
              point.is_query
                ? "size-3.5 bg-foreground ring-2 ring-background"
                : point.measured_label === 1
                  ? "size-2 bg-destructive/70"
                  : "size-2 bg-chart-2"
            }`}
            style={{ left: `${place(point.x, minX, maxX)}%`, top: `${place(point.y, minY, maxY)}%` }}
          />
        ))}
      </div>
      <Legend
        items={[["bg-foreground", "query"], ["bg-destructive/70", "positive dataset label"],
                ["bg-chart-2", "negative dataset label"]]}
      />
      <p className="text-sm leading-relaxed text-muted-foreground">{data.note}</p>
    </div>
  )
}

function Legend({ items }: { items: [string, string][] }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
      {items.map(([colour, label]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className={`size-2.5 rounded-full ${colour}`} /> {label}
        </span>
      ))}
    </div>
  )
}

export function StructurePair({ data }: { data: any }) {
  return (
    <div className="space-y-2">
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <Badge variant="destructive" className="mb-1">Query</Badge>
          <Depiction svg={data.panels.a.svg} />
        </div>
        <div>
          <Badge className="mb-1">Candidate</Badge>
          <Depiction svg={data.panels.b.svg} />
        </div>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
        Grey is the shared scaffold ({data.shared_atom_count} atoms). Orange is what differs.
      </p>
    </div>
  )
}

export function Attribution({ data }: { data: any }) {
  return (
    <div className="space-y-2">
      <Depiction svg={data.svg} className="mx-auto max-w-sm" />
      {data.atoms_unscorable > 0 && (
        <Alert variant="destructive">
          <AlertDescription>
            {data.atoms_unscorable} of {data.atoms_total} atoms could not be scored and are grey.
            Deleting a ring or aromatic atom leaves nothing that sanitizes, so this method says
            nothing about them — grey is "no answer", not "no effect".
          </AlertDescription>
        </Alert>
      )}
      <Legend items={[["bg-[#ff4040]", `raises ${data.enzyme}`], ["bg-[#408cff]", "lowers it"],
                      ["bg-[#cccccc]", "unscorable"]]} />
      <p className="text-sm leading-relaxed text-muted-foreground">
        {data.weight_units}. This perturbs the model's input — it explains the model, not the chemistry.
      </p>
    </div>
  )
}

export function AnalogList({ data, groups = [data] }: { data: any; groups?: any[] }) {
  const cards = new Map<string, { analog: any; labels: Map<string, any>; similarities: Set<number> }>()
  for (const group of groups) {
    for (const analog of group.analogs ?? []) {
      let card = cards.get(analog.smiles)
      if (!card) {
        card = { analog, labels: new Map(), similarities: new Set() }
        cards.set(analog.smiles, card)
      }
      const label = { enzyme: group.enzyme, value: analog.measured_label, meaning: analog.measured_meaning }
      card.labels.set(JSON.stringify(label), label)
      card.similarities.add(analog.similarity)
    }
  }
  const notes = [...new Set(groups.map(group => group.note).filter(Boolean))]
  return (
    <div className="@container space-y-3">
      <div className="grid grid-cols-1 gap-3 @min-[30rem]:grid-cols-2 @min-[42rem]:grid-cols-3">
        {[...cards.values()].map(({ analog, labels, similarities }, index) => (
          <figure key={analog.smiles} className="min-w-0 overflow-hidden rounded-lg border">
            <Depiction svg={analog.svg} className="rounded-none border-b" />
            <figcaption className="space-y-1.5 p-2.5">
              <p className="text-sm leading-snug font-medium wrap-anywhere" title={analog.name || undefined}>
                {analog.name
                  ? analog.name.length > 90 ? `${analog.name.slice(0, 90)}…` : analog.name
                  : <span className="text-muted-foreground">Training analog {index + 1}</span>}
              </p>
              <div className="flex flex-wrap items-center gap-1.5">
                {[...labels.entries()].map(([key, label]) => (
                  <Badge key={key} variant={label.value ? "destructive" : "default"} className="h-auto max-w-full whitespace-normal">
                    {label.enzyme || "Unspecified enzyme"}: {label.meaning}
                  </Badge>
                ))}
                <Badge variant="outline" className="h-auto max-w-full whitespace-normal">Tanimoto {[...similarities].join(", ")}</Badge>
              </div>
              <Smiles value={analog.smiles} />
            </figcaption>
          </figure>
        ))}
      </div>
      {groups.filter(group => !group.analogs?.length).map((group, index) => (
        <p key={index} className="text-sm text-muted-foreground">
          {group.enzyme || "Unspecified enzyme"}: no measured analog above the similarity floor.
        </p>
      ))}
      {notes.map(note => <p key={note} className="text-sm leading-relaxed text-muted-foreground">{note}</p>)}
    </div>
  )
}

export function ExternalAnalogList({ data }: { data: any }) {
  const analogs: any[] = data.analogs ?? []
  const sourceUrl = data.source === "https://www.ebi.ac.uk/chembl/api/data" ? data.source : null
  return (
    <div className="@container space-y-3 text-sm leading-relaxed">
      <p className="text-muted-foreground">
        {sourceUrl
          ? <a href={sourceUrl} target="_blank" rel="noreferrer" className="underline underline-offset-2">ChEMBL similarity search</a>
          : "ChEMBL similarity search"}
        {data.retrieved_at && ` · retrieved ${data.retrieved_at}`}
      </p>
      {data.status === "error" && (
        <p role="alert" className="text-destructive">Search failed: {data.error || "Retrieval did not complete."}</p>
      )}
      {data.status === "unavailable" && (
        <p>{data.error || "External analogue search is unavailable in this session."}</p>
      )}
      {(data.status === "ok" || data.status === "not_found") && (
        <p className="text-muted-foreground">
          Showing {analogs.length} compounds · {data.candidate_count ?? 0} ChEMBL matches
          {data.threshold_percent != null && ` · search threshold ${data.threshold_percent}%`}
        </p>
      )}
      {!analogs.length && data.status === "not_found" && (
        <p>No external analogues returned at this threshold. This does not establish that none exist.</p>
      )}
      <ul className="grid grid-cols-1 gap-3 @min-[30rem]:grid-cols-2 @min-[42rem]:grid-cols-3">
        {analogs.map(analog => (
          <li key={analog.compound_id} className="min-w-0 overflow-hidden rounded-lg border">
            {analog.svg && (
              <div className="border-b bg-white">
                <Depiction svg={analog.svg} className="mx-auto w-full max-w-xs rounded-none" />
              </div>
            )}
            <div className="space-y-2 p-3">
              <p className="font-medium wrap-anywhere">
                {/^CHEMBL\d+$/.test(analog.compound_id)
                  ? <a href={`https://www.ebi.ac.uk/chembl/compound_report_card/${analog.compound_id}/`} target="_blank" rel="noreferrer" className="underline underline-offset-2">{analog.name || analog.compound_id}</a>
                  : analog.name || analog.compound_id}
              </p>
              <div className="flex flex-wrap gap-1.5">
                <Badge variant="outline" className="h-auto max-w-full whitespace-normal">ChEMBL similarity {analog.chembl_similarity_percent.toFixed(1)}%</Badge>
                <Badge variant="outline" className="h-auto max-w-full whitespace-normal">Morgan Tanimoto {analog.morgan_similarity.toFixed(3)}</Badge>
              </div>
              {!!analog.candidate_ids?.length && (
                <p className="wrap-anywhere text-muted-foreground">Already in this batch: {analog.candidate_ids.join(", ")}</p>
              )}
              <Smiles value={analog.smiles} />
            </div>
          </li>
        ))}
      </ul>
      {data.truncated && <p>Results are limited; additional analogues may exist.</p>}
      <p className="text-muted-foreground">
        Similarity does not establish CYP activity. These compounds may overlap the training
        data; supplier availability has not been checked.
      </p>
    </div>
  )
}

export function SourceDocuments({ data }: { data: any }) {
  const papers: any[] = data.papers ?? []
  return (
    <div className="space-y-3 text-sm leading-relaxed">
      <p className="text-muted-foreground">
        <a href="https://www.ebi.ac.uk/chembl/api/data" target="_blank" rel="noreferrer" className="underline underline-offset-2">ChEMBL source documents</a>
        {data.retrieved_at && ` · retrieved ${data.retrieved_at}`}
        {data.origin === "recorded" && " · recorded playback"}
      </p>
      {data.status === "error" && (
        <p role="alert" className="text-destructive">Document lookup failed: {data.error || "Retrieval did not complete."}</p>
      )}
      {data.status === "unavailable" && (
        <p>{data.error || "Document lookup is unavailable in this session."}</p>
      )}
      {data.status === "not_found" && !papers.length && (
        <p>No source documents were returned for these assay records.</p>
      )}
      <ul className="space-y-3">
        {papers.map(paper => (
          <li key={paper.document_id} className="space-y-2 rounded-lg border p-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{paper.document_type || "Unspecified document type"}</Badge>
              {/^CHEMBL\d+$/.test(paper.document_id)
                ? <a href={`https://www.ebi.ac.uk/chembl/explore/document/${paper.document_id}`} target="_blank" rel="noreferrer" className="underline underline-offset-2">{paper.document_id}</a>
                : <span>{paper.document_id}</span>}
            </div>
            <p className="font-medium">{paper.title || "Untitled source document"}</p>
            {paper.authors && <p className="text-muted-foreground">{paper.authors}</p>}
            {(paper.journal || paper.year) && (
              <p className="text-muted-foreground">{[paper.journal, paper.year].filter(Boolean).join(" · ")}</p>
            )}
            <div className="flex flex-wrap gap-x-3 gap-y-1">
              {typeof paper.doi === "string" && /^10\.\d{4,9}\/\S+$/.test(paper.doi) && (
                <a href={`https://doi.org/${encodeURIComponent(paper.doi)}`} target="_blank" rel="noreferrer" className="wrap-anywhere underline underline-offset-2">DOI: {paper.doi}</a>
              )}
              {paper.pubmed_id != null && /^\d+$/.test(String(paper.pubmed_id)) && (
                <a href={`https://pubmed.ncbi.nlm.nih.gov/${paper.pubmed_id}/`} target="_blank" rel="noreferrer" className="underline underline-offset-2">PubMed: {paper.pubmed_id}</a>
              )}
            </div>
            {paper.abstract ? (
              <details>
                <summary className="cursor-pointer text-muted-foreground underline underline-offset-2">Read abstract</summary>
                <p className="mt-2 whitespace-pre-wrap wrap-anywhere">{paper.abstract}</p>
              </details>
            ) : <p className="text-muted-foreground">No abstract provided by ChEMBL.</p>}
          </li>
        ))}
      </ul>
      {!!data.missing_ids?.length && (
        <p className="text-muted-foreground">Documents not retrieved: {data.missing_ids.join(", ")}</p>
      )}
      {data.documents_truncated && <p>Showing a limited set of this snapshot's source documents.</p>}
    </div>
  )
}

/** SMILES is the machine's name for the molecule; keep it available, not in the way. */
export function Smiles({ value }: { value: string }) {
  const [shown, setShown] = useState(false)
  return (
    <div>
      <button
        type="button"
        aria-expanded={shown}
        onClick={() => setShown(open => !open)}
        className="text-sm text-muted-foreground underline underline-offset-2"
      >
        {shown ? "Hide SMILES" : "SMILES"}
      </button>
      {shown && <p className="mt-1 font-mono text-sm wrap-anywhere text-muted-foreground">{value}</p>}
    </div>
  )
}

export function EvidenceRecords({ data }: { data: any }) {
  const records: any[] = data.records ?? []
  return (
    <div className="space-y-3 text-sm leading-relaxed">
      <p className="text-muted-foreground">Human CYP3A4 evidence{data.retrieved_at ? ` · retrieved ${data.retrieved_at}` : ""}</p>
      {data.status === "error" && <p role="alert" className="text-destructive">Evidence lookup failed: {data.error || "Retrieval did not complete."}</p>}
      {data.status === "identity_mismatch" && <p>No exact structure match. Returned identities were rejected.</p>}
      {data.truncated && <p>Results are truncated; additional observations may exist.</p>}
      {!records.length && data.status === "not_found" && <p>Nothing found for this exact structure. Absent evidence is not evidence of inactivity.</p>}
    <ul className="space-y-2">
      {records.map((record, index) => (
        <li key={index} className="rounded-lg border p-3">
          <p className="font-medium">{record.assay_description || `Observation ${index + 1}`}</p>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {record.standard_type && (
              <Badge variant="outline">
                {record.standard_type} {record.standard_relation ?? ""} {record.standard_value ?? ""} {record.standard_units ?? ""}
              </Badge>
            )}
            {record.activity_comment && <Badge variant="secondary">{record.activity_comment}</Badge>}
            {record.veith_overlap && <Badge variant="outline">Known training-source overlap</Badge>}
          </div>
        </li>
      ))}
    </ul>
    </div>
  )
}
