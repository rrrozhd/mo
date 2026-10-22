import { RawJson } from "@/components/json-block"

export function HistoryPanel({ state }: { state: any }) {
  const revisions: any[] = state.revisions || []
  if (!revisions.length) return <p className="py-8 text-center text-sm text-muted-foreground">No decisions recorded yet.</p>

  return (
    <ol className="relative space-y-4 border-l pl-5">
      {revisions.map(revision => (
        <li key={revision.revision} className="relative">
          <span className="absolute -left-[26px] top-1 flex size-4 items-center justify-center rounded-full border bg-background text-[9px] font-medium tabular-nums">
            {revision.revision}
          </span>
          <p className="text-sm">{revision.user_action}</p>
          <RawJson value={revision} label="Recorded revision" />
        </li>
      ))}
    </ol>
  )
}
