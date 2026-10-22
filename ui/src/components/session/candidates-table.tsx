import { useState } from "react"
import { MoreHorizontal, Pin, PinOff, Ban, Undo2, FlaskConical, RefreshCw, Hexagon } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { exact } from "@/lib/api"

type Row = { id: string; score: number }

type Props = {
  state: any
  busy: boolean
  onConstraint: (field: "include_ids" | "exclude_ids", id: string, enabled: boolean) => void
  onEvidence: (id: string, refresh: boolean) => void
}

export function CandidatesTable({ state, busy, onConstraint, onEvidence }: Props) {
  const [structure, setStructure] = useState<string | null>(null)
  const rows: Row[] = state.dataset?.scored || []
  const selected = new Set<string>((state.selection?.selected || []).map((row: Row) => row.id))
  const skipped = new Map<string, string>((state.selection?.skipped || []).map((row: any) => [row.id, row.reason]))
  const required = new Set<string>(state.constraints.include_ids)
  const excluded = new Set<string>(state.constraints.exclude_ids)

  return (
    <>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader className="sticky top-0 z-10 bg-card">
            <TableRow>
              <TableHead className="w-[38%]">Candidate</TableHead>
              <TableHead className="w-[22%]">Score</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="w-12 text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map(row => {
              const isRequired = required.has(row.id)
              const isExcluded = excluded.has(row.id)
              const isSelected = selected.has(row.id)
              const reason = skipped.get(row.id)
              return (
                <TableRow
                  key={row.id}
                  data-state={isSelected ? "selected" : undefined}
                  className={isExcluded ? "opacity-55" : undefined}
                >
                  <TableCell className="font-mono text-xs break-words whitespace-pre-wrap">
                    {exact(row.id)}
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 shrink-0 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary/70"
                          style={{ width: `${Math.min(100, Math.max(0, row.score * 100))}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs tabular-nums text-muted-foreground">
                        {row.score.toFixed(4)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="whitespace-normal">
                    {isSelected ? (
                      <div className="flex flex-wrap items-center gap-1.5">
                        <Badge>In batch</Badge>
                        {isRequired && (
                          <Badge variant="outline" className="gap-1">
                            <Pin /> Required
                          </Badge>
                        )}
                      </div>
                    ) : isExcluded ? (
                      <Badge variant="destructive">Excluded</Badge>
                    ) : reason ? (
                      <Tooltip>
                        <TooltipTrigger
                          render={
                            <span className="line-clamp-2 max-w-60 cursor-help text-xs text-muted-foreground">
                              {reason}
                            </span>
                          }
                        />
                        <TooltipContent className="max-w-xs">{reason}</TooltipContent>
                      </Tooltip>
                    ) : (
                      <span className="text-xs text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger
                        render={
                          <Button variant="ghost" size="icon" disabled={busy} aria-label={`Actions for ${row.id}`}>
                            <MoreHorizontal />
                          </Button>
                        }
                      />
                      <DropdownMenuContent align="end" className="w-52">
                        <DropdownMenuItem onClick={() => onConstraint("include_ids", row.id, !isRequired)}>
                          {isRequired ? <PinOff /> : <Pin />}
                          {isRequired ? "Drop requirement" : "Require in batch"}
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => onConstraint("exclude_ids", row.id, !isExcluded)}>
                          {isExcluded ? <Undo2 /> : <Ban />}
                          {isExcluded ? "Remove exclusion" : "Exclude"}
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onClick={() => onEvidence(row.id, false)}>
                          <FlaskConical /> Look up evidence
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => onEvidence(row.id, true)}>
                          <RefreshCw /> Refresh evidence
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onClick={() => setStructure(row.id)}>
                          <Hexagon /> View structure
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>

      <Dialog open={!!structure} onOpenChange={open => !open && setStructure(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="font-mono text-sm">{structure && exact(structure)}</DialogTitle>
            <DialogDescription>Exact supplied structure, rendered locally.</DialogDescription>
          </DialogHeader>
          {structure && (
            <img
              alt={`Structure for ${exact(structure)}`}
              className="mx-auto h-64 w-full rounded-lg bg-white object-contain"
              src={`/api/depiction?${new URLSearchParams({ session_id: state.session_id, candidate_id: structure })}`}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
