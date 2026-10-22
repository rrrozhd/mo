import { CheckCircle2, Download, Table2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { exact } from "@/lib/api"

/** The batch a chemist is working on, always visible above the chat. */
export function BatchSummary({
  state, busy, onReview, onExport, onOpenDetails,
}: {
  state: any
  busy: boolean
  onReview: () => void
  onExport: () => void
  onOpenDetails: () => void
}) {
  const selected: any[] = state.selection?.selected ?? []
  const required: string[] = state.constraints.include_ids
  const excluded: string[] = state.constraints.exclude_ids
  return (
    <div className="rounded-xl border bg-card p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={state.selection?.complete ? "default" : "destructive"}>
          {selected.length} of {state.constraints.budget} selected
        </Badge>
        <Badge variant="outline">max Tanimoto {state.constraints.max_similarity}</Badge>
        {required.length > 0 && <Badge variant="secondary">{required.length} required</Badge>}
        {excluded.length > 0 && <Badge variant="secondary">{excluded.length} excluded</Badge>}
        <div className="ml-auto flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={onOpenDetails}>
            <Table2 /> Candidates
          </Button>
          <Button variant="outline" size="sm" disabled={busy} onClick={onReview}>
            <CheckCircle2 /> Mark reviewed
          </Button>
          <Button variant="outline" size="sm" disabled={busy} onClick={onExport}>
            <Download /> Export
          </Button>
        </div>
      </div>
      <p className="mt-2 font-mono text-[11px] break-words text-muted-foreground">
        {selected.length ? selected.map(row => exact(row.id)).join("  ") : "Nothing selected."}
      </p>
      {!state.selection?.complete && state.selection?.shortfall_reason && (
        <p className="mt-1 text-xs text-destructive">{state.selection.shortfall_reason}</p>
      )}
    </div>
  )
}
