import { SquarePen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"

type ChatSidebarProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  state: any
  busy: boolean
  onSwitch: (chatId: string) => void
  onNewChat: () => void
}

function ChatList({ state, busy, onSwitch, onNewChat }: ChatSidebarProps) {
  const chats = [state, ...(state.archived_chats ?? []).slice().reverse()]
  return (
    <>
      <Button variant="outline" className="w-full shrink-0 justify-start" disabled={busy} onClick={onNewChat}>
        <SquarePen /> New chat
      </Button>
      <nav aria-label="Chat history" className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {chats.map(chat => {
          const current = chat.chat_id === state.chat_id
          const turns = chat.turns ?? []
          const title = turns[0]?.message?.trim().replace(/\s+/g, " ").slice(0, 80) || "New chat"
          return (
            <Button
              key={chat.chat_id}
              variant={current ? "secondary" : "ghost"}
              className={`h-auto w-full justify-start px-3 py-2.5 text-left ${current ? "disabled:opacity-100" : ""}`}
              disabled={busy || current}
              aria-current={current ? "page" : undefined}
              onClick={() => onSwitch(chat.chat_id)}
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate">{title}</span>
                <span className="mt-1 block text-xs font-normal text-muted-foreground">
                  {turns.length} {turns.length === 1 ? "message" : "messages"}{current ? " · Current" : ""}
                </span>
              </span>
            </Button>
          )
        })}
      </nav>
      <p className="shrink-0 text-xs leading-relaxed text-muted-foreground">
        All chats share the current compounds and evidence.
      </p>
    </>
  )
}

export function ChatSidebar(props: ChatSidebarProps) {
  return (
    <>
      <aside aria-label="Chats" className="fixed inset-y-0 left-0 z-40 hidden h-dvh w-56 flex-col gap-4 border-r bg-muted/30 p-4 md:flex">
        <h2 className="py-1 text-sm font-semibold">Chats</h2>
        <ChatList {...props} />
      </aside>
      <Dialog open={props.open} onOpenChange={props.onOpenChange}>
        <DialogContent className="top-0 left-0 flex h-dvh w-72 max-w-[85vw] translate-x-0 translate-y-0 flex-col rounded-none sm:max-w-72 data-open:zoom-in-100 data-closed:zoom-out-100">
          <DialogHeader>
            <DialogTitle>Chats</DialogTitle>
            <DialogDescription className="sr-only">Create a chat or continue an earlier conversation.</DialogDescription>
          </DialogHeader>
          <ChatList {...props} />
        </DialogContent>
      </Dialog>
    </>
  )
}
