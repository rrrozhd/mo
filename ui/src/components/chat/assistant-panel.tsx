import type { Dispatch, SetStateAction } from "react"
import { Plus, Send } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { MarkdownAnswer } from "@/components/chat/markdown-answer"
import { Json, RawJson } from "@/components/json-block"
import {
  AnalogList, Attribution, EvidenceRecords, ExternalAnalogList, Neighbourhood, ScoreBars,
  Smiles, SourceDocuments, StructurePair,
} from "@/components/chat/molecule-attachments"

const BLOCK_TITLE: Record<string, string> = {
  scores: "Five-enzyme profile",
  analogs: "Measured analogs",
  external_analogs: "External analogues",
  structures: "What changed",
  map: "Neighbourhood",
  attribution: "Atom attribution",
  evidence: "Published evidence",
  papers: "Source documents",
}

function groupAnalogAttachments(blocks: any[]) {
  const result: any[] = []
  const queries = new Map<string, any>()
  for (const block of blocks) {
    const query = block.data?.query_smiles
    if (block.kind !== "analogs" || typeof query !== "string" || !query) {
      result.push(block)
      continue
    }
    const existing = queries.get(query)
    if (existing) existing.analogGroups.push(block.data)
    else {
      const grouped = { ...block, analogGroups: [block.data] }
      queries.set(query, grouped)
      result.push(grouped)
    }
  }
  return result
}

function Attachment({ block }: { block: any }) {
  const queryName = block.data?.query_name ?? block.data?.name
  const body =
    block.kind === "scores" ? <ScoreBars left={block.data} /> :
    block.kind === "analogs" ? <AnalogList data={block.data} groups={block.analogGroups} /> :
    block.kind === "external_analogs" ? <ExternalAnalogList data={block.data} /> :
    block.kind === "structures" ? <StructurePair data={block.data} /> :
    block.kind === "map" ? <Neighbourhood data={block.data} /> :
    block.kind === "attribution" ? <Attribution data={block.data} /> :
    block.kind === "evidence" ? <EvidenceRecords data={block.data} /> :
    block.kind === "papers" ? <SourceDocuments data={block.data} /> : null
  if (!body) return null
  return (
    <div className="mt-5 min-w-0 rounded-xl border bg-background p-4">
      <p className="mb-3 text-sm font-medium">{BLOCK_TITLE[block.kind]}</p>
      {typeof queryName === "string" && queryName && <p className="mb-3 text-sm font-medium">{queryName}</p>}
      {block.data?.query_smiles && (
        <div className="mb-3"><Smiles value={block.data.query_smiles} /></div>
      )}
      {body}
    </div>
  )
}

const DECISION_LABEL: Record<string, string> = {
  explicit_change: "Explicit change",
  question: "Question",
  suggestion: "Suggestion",
}

export function AssistantPanel({
  state,
  busy,
  assistantAvailable,
  onSend,
  onApply,
  onAddFiles,
  message,
  onMessageChange: setMessage,
}: {
  state: any
  busy: boolean
  assistantAvailable: boolean
  onSend: (message: string) => Promise<boolean>
  onApply: (proposalId: string) => void
  onAddFiles: () => void
  message: string
  onMessageChange: Dispatch<SetStateAction<string>>
}) {
  const turns: any[] = state.turns || []
  const proposals: any[] = state.proposals || []

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-5">
      {!assistantAvailable && (
        <Alert>
          <AlertDescription>
            Mo is unavailable in this session. Direct selection, comparison and evidence controls remain available.
          </AlertDescription>
        </Alert>
      )}

      <div className="min-w-0 flex-1 space-y-10">
        {turns.map((turn, index) => (
          <div key={index} className="min-w-0 space-y-5">
            <div className="ml-auto w-fit max-w-[90%] rounded-2xl rounded-br-sm bg-muted px-4 py-3 text-base leading-relaxed whitespace-pre-wrap wrap-anywhere">
              {turn.message}
            </div>
            <div className="min-w-0">
              <div className="mb-3 flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                <span className="font-medium">Mo</span>
                {turn.interpretation && turn.interpretation.decision !== "question" && (
                  <Badge variant="secondary">
                    {DECISION_LABEL[turn.interpretation.decision] ?? turn.interpretation.decision}
                  </Badge>
                )}
                {turn.interpretation?.clarification && (
                  <span>{turn.interpretation.clarification}</span>
                )}
              </div>
              {turn.answer?.explanation && (
                <MarkdownAnswer>{turn.answer.explanation}</MarkdownAnswer>
              )}
              {turn.answer?.unresolved_questions?.length > 0 && (
                <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                  Open questions: {turn.answer.unresolved_questions.join(" ")}
                </p>
              )}
              {turn.error && <p className="text-sm text-destructive">{turn.error.message}</p>}
              {turn.normalized_patch && Object.keys(turn.normalized_patch).length > 0 && (
                <Json value={turn.normalized_patch} className="mt-2" />
              )}
              {groupAnalogAttachments(turn.answer?.attachments ?? []).map((block: any, position: number) => (
                <Attachment key={position} block={block} />
              ))}
              <RawJson value={turn} label="Retained turn and references" />
            </div>
          </div>
        ))}

        {proposals.map(proposal => {
          const stale =
            proposal.applied ||
            proposal.base_revision !== state.revision ||
            proposal.evidence_version !== state.evidence_version ||
            proposal.session_id !== state.session_id
          return (
            <div key={proposal.proposal_id} className="rounded-xl border border-dashed p-3">
              <Badge variant={proposal.applied ? "secondary" : stale ? "outline" : "default"}>
                {proposal.applied ? "Applied proposal" : stale ? "Stale proposal" : "Proposed change"}
              </Badge>
              <p className="mt-3 text-base leading-relaxed">{proposal.rationale}</p>
              <Json value={proposal.patch} className="mt-2" />
              <Button
                size="sm"
                className="mt-2"
                disabled={busy || stale}
                onClick={() => onApply(proposal.proposal_id)}
              >
                Apply this proposal
              </Button>
            </div>
          )
        })}
      </div>

      {!!state.operations?.length && <RawJson value={state.operations} label="Actual operation results" />}

      <div className="sticky bottom-0 z-20 shrink-0 bg-background pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
        <form
          className="flex items-end gap-2 rounded-xl border bg-card p-2"
          onSubmit={async event => {
            event.preventDefault()
            if (busy || !assistantAvailable || !message.trim()) return
            const sent = message
            if (await onSend(sent)) setMessage(current => current === sent ? "" : current)
          }}
        >
          <Textarea
            aria-label="Message"
            className="max-h-40 min-h-9 min-w-0 flex-1 resize-none border-0 bg-transparent px-2 py-2 text-base leading-5 shadow-none md:text-base dark:bg-transparent"
            rows={1}
            maxLength={8000}
            value={message}
            onChange={event => setMessage(event.target.value)}
            placeholder="Ask Mo about molecules…"
            onKeyDown={event => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }}
          />
          <div className="flex shrink-0 items-center gap-1">
              <Tooltip>
                <TooltipTrigger render={
                  <Button type="button" variant="ghost" size="icon" className="size-9" aria-label="Add files" disabled={busy} onClick={onAddFiles}>
                    <Plus className="size-4" />
                  </Button>
                } />
                <TooltipContent>Add files or load seed sample</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger render={
                  <Button type="submit" size="icon" className="size-9" aria-label="Send message" disabled={busy || !assistantAvailable || !message.trim()}>
                    <Send className="size-4" />
                  </Button>
                } />
                <TooltipContent>Send message · Ctrl / ⌘ + Enter</TooltipContent>
              </Tooltip>
          </div>
        </form>
      </div>
    </div>
  )
}
