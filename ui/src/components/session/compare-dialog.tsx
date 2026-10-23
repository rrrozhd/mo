import { useMemo, useState } from "react"
import { GitCompare } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { RawJson } from "@/components/json-block"
import { Smiles } from "@/components/chat/molecule-attachments"
import { exact } from "@/lib/api"

const RELATIONSHIP_LABEL: Record<string, string> = {
  stereo_only: "Same constitution, stereochemistry differs",
  identical: "Identical exact structures",
  different: "Different structures",
}

export function CompareDialog({
  state,
  busy,
  onCompare,
}: {
  state: any
  busy: boolean
  onCompare: (a: string, b: string) => void
}) {
  const rows: { id: string }[] = state.dataset?.scored || []
  const [a, setA] = useState(rows[0]?.id ?? "")
  const [b, setB] = useState(rows[1]?.id ?? rows[0]?.id ?? "")

  const result = useMemo(() => {
    const events = (state.operations || []).flatMap((op: any) => op.events || [])
    return events.filter((event: any) => event.kind === "compare_candidates").at(-1)?.result
  }, [state.operations])

  return (
    <Dialog>
      <DialogTrigger
        render={
          <Button variant="outline" size="sm">
            <GitCompare /> Compare structures
          </Button>
        }
      />
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Compare exact structures</DialogTitle>
          <DialogDescription>
            Compares the supplied structures directly. It does not pool assay observations or infer activity.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-wrap items-end gap-3">
          <div className="grid gap-1.5">
            <Label>First candidate</Label>
            <Select value={a} onValueChange={value => setA(value as string)}>
              <SelectTrigger className="w-56">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {rows.map(row => (
                  <SelectItem key={row.id} value={row.id}>
                    {row.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="grid gap-1.5">
            <Label>Second candidate</Label>
            <Select value={b} onValueChange={value => setB(value as string)}>
              <SelectTrigger className="w-56">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {rows.map(row => (
                  <SelectItem key={row.id} value={row.id}>
                    {row.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button disabled={busy || !a || !b || a === b} onClick={() => onCompare(a, b)}>
            Compare
          </Button>
        </div>

        {result && (
          <>
            <Separator />
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">
                  {RELATIONSHIP_LABEL[result.structural_relationship] ?? result.structural_relationship}
                </Badge>
                <Badge variant="outline">Tanimoto {Number(result.fingerprint_tanimoto).toFixed(3)}</Badge>
                <Badge variant={result.exact_identity_equal ? "default" : "ghost"}>
                  {result.exact_identity_equal ? "Exactly identical" : "Not exactly identical"}
                </Badge>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {[result.a, result.b].map((side: any) => (
                  <div key={side.id} className="rounded-lg border p-3">
                    <p className="text-sm font-medium">{side.name || exact(side.id)}</p>
                    <dl className="mt-2 space-y-1 text-xs text-muted-foreground">
                      <div className="flex justify-between gap-2">
                        <dt>Score</dt>
                        <dd className="font-mono tabular-nums">{Number(side.score).toFixed(6)}</dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt>Validation percentile</dt>
                        <dd className="font-mono tabular-nums">{Number(side.validation_percentile).toFixed(4)}</dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt>Nearest training similarity</dt>
                        <dd className="font-mono tabular-nums">{Number(side.nearest_training_similarity).toFixed(4)}</dd>
                      </div>
                      <div className="flex justify-between gap-2">
                        <dt>Confidence</dt>
                        <dd>{side.confidence}</dd>
                      </div>
                    </dl>
                    <div className="mt-2"><Smiles value={side.canonical_smiles} /></div>
                  </div>
                ))}
              </div>
              <RawJson value={result} label="Raw comparison" />
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
