import { useEffect, useState } from "react"
import {
  CheckCircle2,
  Download,
  FlaskConical,
  Loader2,
  Moon,
  PanelLeft,
  RefreshCw,
  Sun,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { AssistantPanel } from "@/components/chat/assistant-panel"
import { ChatSidebar } from "@/components/chat/chat-sidebar"
import { BatchSummary } from "@/components/session/batch-summary"
import { SessionDetailsDialog } from "@/components/session/session-details-dialog"
import { DatasetLoader } from "@/components/dataset-loader"
import { exact, request } from "@/lib/api"
import { useSession } from "@/hooks/use-session"
import { Toaster } from "@/components/ui/sonner"

function useTheme() {
  const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark")
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    localStorage.setItem("theme", dark ? "dark" : "light")
  }, [dark])
  return { dark, toggle: () => setDark(value => !value) }
}

export default function App() {
  const { state, assistantAvailable, submitting, busy, act, keys, mutate, startJob, loadState, commit } = useSession()
  const { dark, toggle } = useTheme()
  const constraints = state?.constraints
  const [budget, setBudget] = useState(String(constraints?.budget ?? ""))
  const [similarity, setSimilarity] = useState(String(constraints?.max_similarity ?? ""))
  const [loaderOpen, setLoaderOpen] = useState(false)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [drafts, setDrafts] = useState<Record<string, string>>({})

  const [savedConstraints, setSavedConstraints] = useState(constraints)
  if (constraints && (constraints.budget !== savedConstraints?.budget ||
      constraints.max_similarity !== savedConstraints?.max_similarity)) {
    setSavedConstraints(constraints)
    setBudget(String(constraints.budget))
    setSimilarity(String(constraints.max_similarity))
  }

  if (!state) {
    return (
      <div className="flex min-h-svh items-center justify-center gap-2 text-sm text-muted-foreground">
        <Toaster theme={dark ? "dark" : "light"} richColors position="top-right" />
        <Loader2 className="size-4 animate-spin" /> Connecting to the local session…
      </div>
    )
  }

  const hasDataset = !!state.dataset
  const lastOperation = (state.operations || []).at(-1)
  const operationStatus = state.busy
    ? `${state.operation.kind}: ${state.operation.status}${state.operation.status === "failed" ? " · waiting for the worker to finish" : ""}`
    : lastOperation
      ? `${lastOperation.kind}: ${lastOperation.status}${lastOperation.error ? ` · ${lastOperation.error.message}` : ""}`
      : "No operation running."

  const loadSample = () => act(async () => {
    await mutate("/api/sample", { keys: keys() })
    setDrafts({})
    setLoaderOpen(false)
  })

  const startNewChat = () => act(async () => {
    await mutate("/api/chat/new", { keys: keys(true) })
    setHistoryOpen(false)
    setDetailsOpen(false)
    setLoaderOpen(false)
  })

  const switchChat = (chatId: string) => act(async () => {
    await mutate("/api/chat/switch", { keys: keys(true), chat_id: chatId })
    setHistoryOpen(false)
  })

  const upload = (file: File, value: number) => act(async () => {
    if (file.size > 2 * 1024 * 1024) throw new Error("Choose a CSV no larger than 2 MiB.")
    const form = new FormData()
    form.append("file", file)
    form.append("metadata", JSON.stringify({ keys: keys(), budget: value }))
    const body = await request("/api/upload", form, { form: true })
    commit(body.state)
    setDrafts({})
    setLoaderOpen(false)
  })

  const setConstraint = (field: "include_ids" | "exclude_ids", id: string, enabled: boolean) =>
    act(() => {
      const values = state.constraints[field].filter((value: string) => value !== id)
      if (enabled) values.push(id)
      return mutate("/api/constraints", {
        keys: keys(),
        patch: { [field]: values },
        action: `${enabled ? "Add" : "Remove"} ${exact(id)} ${field === "include_ids" ? "requirement" : "exclusion"}`,
      })
    })

  const applyConstraints = () =>
    act(() =>
      mutate("/api/constraints", {
        keys: keys(),
        patch: { budget: Number(budget), max_similarity: Number(similarity) },
        action: "Apply compound budget and similarity controls",
      })
    )

  const exportSession = () => act(async () => {
    const blob = (await request("/api/export", { keys: keys(true), format: "json" }, { download: true })) as Blob
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `cyp3a4-session-r${state.revision}-e${state.evidence_version}.json`
    link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  })

  const dirty =
    hasDataset &&
    (budget !== String(state.constraints.budget) || similarity !== String(state.constraints.max_similarity))

  return (
    <div className="flex min-h-dvh flex-col bg-background md:pl-56">
      <Toaster theme={dark ? "dark" : "light"} richColors position="top-right" />
      <ChatSidebar open={historyOpen} onOpenChange={setHistoryOpen} state={state} busy={busy} onSwitch={switchChat} onNewChat={startNewChat} />
      <header className="sticky top-0 z-30 shrink-0 border-b bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-3 px-4 py-2.5 lg:px-6">
          <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open chats" onClick={() => setHistoryOpen(true)}>
            <PanelLeft />
          </Button>
          <FlaskConical className="size-5 shrink-0 text-primary" />
          <div className="mr-auto min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold">Mo</h1>
            <p className="truncate text-xs text-muted-foreground">
              Session {state.session_id.slice(0, 8)} · revision {state.revision} · evidence v{state.evidence_version} ·{" "}
              {state.source_mode} sources
            </p>
          </div>

          {state.busy && (
            <Badge variant="secondary" className="gap-1">
              <Loader2 className="animate-spin" /> {state.operation.kind}
            </Badge>
          )}
          {hasDataset &&
            (state.review ? (
              <Badge variant="outline" className="gap-1">
                <CheckCircle2 /> Reviewed r{state.review.revision}
              </Badge>
            ) : (
              <Badge variant="ghost">Draft</Badge>
            ))}

          {!hasDataset && (!!state.turns?.length || !!state.archived_chats?.length) && (
            <Tooltip>
              <TooltipTrigger render={
                <Button variant="ghost" size="icon" disabled={busy} onClick={exportSession} aria-label="Export chat history">
                  <Download />
                </Button>
              } />
              <TooltipContent>Export chat history</TooltipContent>
            </Tooltip>
          )}

          <Dialog open={loaderOpen} onOpenChange={setLoaderOpen}>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>Add a chemical batch</DialogTitle>
                <DialogDescription>
                  Choose a CSV with compound IDs and SMILES, or try the sample batch.
                </DialogDescription>
              </DialogHeader>
              <DatasetLoader busy={busy} hasWork={hasDataset || !!state.turns?.length || !!state.archived_chats?.length} onSample={loadSample} onUpload={upload} />
            </DialogContent>
          </Dialog>

          <Tooltip>
            <TooltipTrigger
              render={
                <Button variant="ghost" size="icon" disabled={submitting} onClick={() => act(loadState)} aria-label="Reload state">
                  <RefreshCw className={submitting ? "animate-spin" : undefined} />
                </Button>
              }
            />
            <TooltipContent>Reload state</TooltipContent>
          </Tooltip>
          <Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">
            {dark ? <Sun /> : <Moon />}
          </Button>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col px-4 pt-4 lg:px-6">
        <p className="mx-auto mb-4 w-full max-w-3xl text-xs leading-relaxed text-muted-foreground">
          Scores are uncalibrated rankings—not evidence of substrate status, clinical risk, or experimental success.
        </p>

        {(state.busy || lastOperation) && (
          <p role="status" className="mb-4 text-center text-xs text-muted-foreground">{operationStatus}</p>
        )}

        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4">
          {hasDataset && (
            <BatchSummary
              state={state}
              busy={busy}
              onReview={() => act(() => mutate("/api/review", { keys: keys(true) }))}
              onExport={exportSession}
              onOpenDetails={() => setDetailsOpen(true)}
            />
          )}
          <AssistantPanel
            key={state.chat_id}
            state={state}
            busy={busy}
            assistantAvailable={assistantAvailable}
            message={drafts[state.chat_id] ?? ""}
            onMessageChange={value => setDrafts(current => ({
              ...current,
              [state.chat_id]: typeof value === "function" ? value(current[state.chat_id] ?? "") : value,
            }))}
            onAddFiles={() => setLoaderOpen(true)}
            onSend={message => act(() => startJob("/api/chat", { keys: keys(), message }))}
            onApply={proposalId =>
              act(() => mutate("/api/apply", { keys: keys(true), proposal_id: proposalId }))
            }
          />
        </div>

        {hasDataset && (
          <SessionDetailsDialog
            open={detailsOpen}
            onOpenChange={setDetailsOpen}
            state={state}
            busy={busy}
            budget={budget}
            similarity={similarity}
            setBudget={setBudget}
            setSimilarity={setSimilarity}
            dirty={dirty}
            onApplyConstraints={applyConstraints}
            onConstraint={setConstraint}
            onEvidence={(id: string, refresh: boolean) =>
              act(() => startJob("/api/evidence", { keys: keys(), ids: [id], refresh }))}
            onCompare={(a: string, b: string) =>
              act(() => startJob("/api/compare", { keys: keys(), a_id: a, b_id: b }))}
          />
        )}
      </main>
    </div>
  )
}
