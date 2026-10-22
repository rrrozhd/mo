import { ChevronRight } from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

export function Json({ value, className = "" }: { value: unknown; className?: string }) {
  return (
    <pre
      className={`max-h-96 overflow-auto rounded-lg bg-muted/60 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words ${className}`}
    >
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

export function RawJson({ value, label = "Raw record" }: { value: unknown; label?: string }) {
  return (
    <Collapsible className="mt-2">
      <CollapsibleTrigger className="group flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
        <ChevronRight className="size-3 transition-transform group-data-[panel-open]:rotate-90" />
        {label}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2">
        <Json value={value} />
      </CollapsibleContent>
    </Collapsible>
  )
}
