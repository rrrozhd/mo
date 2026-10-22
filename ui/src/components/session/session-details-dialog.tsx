import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CandidatesTable } from "@/components/session/candidates-table"
import { CompareDialog } from "@/components/session/compare-dialog"
import { EvidencePanel } from "@/components/session/evidence-panel"
import { HistoryPanel } from "@/components/session/history-panel"

/** Everything the chat can also do, kept reachable by hand for inspection. */
export function SessionDetailsDialog({
  open, onOpenChange, state, busy, budget, similarity, setBudget, setSimilarity,
  dirty, onApplyConstraints, onConstraint, onEvidence, onCompare,
}: any) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[88vh] overflow-y-auto sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle>Session details</DialogTitle>
          <DialogDescription>
            Inspect candidates, source evidence, and recorded changes.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="candidates" className="min-w-0">
          <TabsList>
            <TabsTrigger value="candidates">Candidates</TabsTrigger>
            <TabsTrigger value="evidence">Evidence</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>

          <TabsContent value="candidates" className="min-w-0 space-y-3">
            <div className="flex flex-wrap items-end gap-3">
              <div className="grid gap-1.5">
                <Label htmlFor="budget">Compound budget</Label>
                <Input id="budget" type="number" min={1} step={1} className="w-28"
                       value={budget} onChange={(e: any) => setBudget(e.target.value)} />
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="similarity">Max pairwise Tanimoto</Label>
                <Input id="similarity" type="number" min={0} max={1} step="any" className="w-28"
                       value={similarity} onChange={(e: any) => setSimilarity(e.target.value)} />
              </div>
              <Button disabled={busy || !dirty} onClick={onApplyConstraints}>Apply constraints</Button>
              <div className="ml-auto">
                <CompareDialog state={state} busy={busy} onCompare={onCompare} />
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Required compounds count toward the budget. The similarity limit is a diversity policy,
              not a confidence threshold.
            </p>
            <Separator />
            <CandidatesTable state={state} busy={busy} onConstraint={onConstraint} onEvidence={onEvidence} />
          </TabsContent>

          <TabsContent value="evidence" className="min-w-0">
            <EvidencePanel state={state} />
          </TabsContent>

          <TabsContent value="history" className="min-w-0">
            <HistoryPanel state={state} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
