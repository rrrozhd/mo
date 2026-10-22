import { useRef, useState } from "react"
import { TriangleAlert, Upload } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

export function DatasetLoader({
  busy,
  hasWork,
  onSample,
  onUpload,
}: {
  busy: boolean
  hasWork: boolean
  onSample: () => void
  onUpload: (file: File, budget: number) => void
}) {
  const [budget, setBudget] = useState("4")
  const file = useRef<HTMLInputElement>(null)

  return (
    <div className="space-y-4">
      {hasWork && (
        <Alert variant="destructive">
          <TriangleAlert />
          <AlertDescription>
            Loading a sample or CSV replaces the current session and loses unexported work.
          </AlertDescription>
        </Alert>
      )}

      <Button variant="secondary" className="w-full" disabled={busy} onClick={onSample}>
        Load 20-compound sample · budget 4
      </Button>

      <div className="flex items-center gap-3">
        <Separator className="flex-1" />
        <span className="text-xs text-muted-foreground">or</span>
        <Separator className="flex-1" />
      </div>

      <form
        className="space-y-3"
        onSubmit={event => {
          event.preventDefault()
          const chosen = file.current?.files?.[0]
          if (chosen) onUpload(chosen, Number(budget))
        }}
      >
        <div className="grid gap-1.5">
          <Label htmlFor="csv-file">Candidate CSV</Label>
          <Input id="csv-file" ref={file} type="file" accept=".csv,text/csv" required />
          <p className="text-xs text-muted-foreground">
            Requires <code className="font-mono">id,smiles</code>; extra columns are discarded. Limit: 200 rows, 2 MiB
            UTF-8, 4,000 characters per ID or SMILES.
          </p>
        </div>
        <div className="grid gap-1.5">
          <Label htmlFor="upload-budget">Initial compound budget</Label>
          <Input
            id="upload-budget"
            type="number"
            min={1}
            max={200}
            step={1}
            required
            value={budget}
            onChange={event => setBudget(event.target.value)}
          />
        </div>
        <Button type="submit" className="w-full" disabled={busy}>
          <Upload /> Load CSV and replace session
        </Button>
      </form>
    </div>
  )
}
