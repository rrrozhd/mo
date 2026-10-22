import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"

export function MarkdownAnswer({ children }: { children: string }) {
  return (
    <div className="chat-markdown">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="chat-table" role="region" aria-label="Answer table" tabIndex={0}>
              <table>{children}</table>
            </div>
          ),
          pre: ({ children }) => (
            <pre role="region" tabIndex={0} aria-label="Code block">{children}</pre>
          ),
        }}
      >
        {children}
      </Markdown>
    </div>
  )
}
