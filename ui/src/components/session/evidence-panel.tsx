import { ExternalLink, Info } from "lucide-react"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { RawJson } from "@/components/json-block"
import { exact, safeUrl } from "@/lib/api"

const LINKS: [string, string][] = [
  ["Activity", "activity_url"],
  ["Assay", "assay_url"],
  ["Document", "document_url"],
]

type EvidenceSnapshot = {
  snapshot_id: string
  candidate_id: string | null
  canonical_query: string
  origin: string
  retrieved_at?: string | null
  playback_at?: string
  fixture_hash?: string
  result?: { status?: string; records?: any[] }
}

export function EvidencePanel({ state }: { state: any }) {
  const snapshots: EvidenceSnapshot[] = state.snapshots || []

  return (
    <div className="space-y-3">
      <Alert>
        <Info />
        <AlertDescription>
          {state.source_mode === "recorded"
            ? "Recorded source mode: dated captured observations; no live evidence requests or fallback."
            : "Live source mode: dated ChEMBL retrievals. Cached observations retain their original time; explicit refresh makes a new lookup."}
          {" "}Observations remain separate from predictions. Missing evidence does not mean inactive. A shared assay does
          not establish independent replication or a causal explanation. A false overlap flag does not prove independence.
        </AlertDescription>
      </Alert>

      {snapshots.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted-foreground">
          No evidence attached yet. Use a candidate's Look up evidence action; Refresh evidence makes a new request.
        </p>
      ) : (
        <Accordion className="rounded-lg border px-3">
          {snapshots.map(snapshot => {
            const current = snapshot.candidate_id !== null &&
              state.current_evidence[snapshot.candidate_id] === snapshot.snapshot_id
            const label = snapshot.candidate_id === null ? "Structure lookup" : current ? "Current" : "Historical"
            const records: any[] = snapshot.result?.records || []
            return (
              <AccordionItem key={snapshot.snapshot_id} value={snapshot.snapshot_id}>
                <AccordionTrigger>
                  <span className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs">{exact(snapshot.candidate_id ?? snapshot.canonical_query)}</span>
                    <Badge variant={current ? "default" : "outline"}>{label}</Badge>
                    <Badge variant="secondary">{snapshot.result?.status}</Badge>
                    <span className="text-xs font-normal text-muted-foreground">
                      {snapshot.origin} · {snapshot.retrieved_at || "No original retrieval"}
                    </span>
                  </span>
                </AccordionTrigger>
                <AccordionContent className="pb-3">
                  {snapshot.origin === "recorded" && (
                    <p className="text-xs text-muted-foreground">
                      Recorded playback: {snapshot.playback_at}. Fixture: {snapshot.fixture_hash}.
                    </p>
                  )}
                  {records.length > 0 && (
                    <ul className="mt-2 space-y-2">
                      {records.map((record, index) => (
                        <li key={index} className="rounded-lg border p-2.5 text-xs">
                          <p className="font-medium">{record.assay_description || `Observation ${index + 1}`}</p>
                          <div className="mt-1 flex flex-wrap gap-1.5">
                            {record.standard_type && (
                              <Badge variant="outline">
                                {record.standard_type} {record.standard_relation ?? ""} {record.standard_value ?? ""} {record.standard_units ?? ""}
                              </Badge>
                            )}
                            {record.activity_comment && <Badge variant="secondary">{record.activity_comment}</Badge>}
                          </div>
                          <div className="mt-1.5 flex flex-wrap gap-3">
                            {LINKS.map(([label, key]) => {
                              const href = safeUrl(record[key])
                              return href ? (
                                <a
                                  key={key}
                                  href={href}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1 text-primary hover:underline"
                                >
                                  {label} <ExternalLink className="size-3" />
                                </a>
                              ) : null
                            })}
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                  <RawJson value={snapshot} label="Raw snapshot" />
                </AccordionContent>
              </AccordionItem>
            )
          })}
        </Accordion>
      )}
    </div>
  )
}
