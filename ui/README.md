# Mo UI

React + Tailwind + shadcn/ui front end for `src/workbench/`.

`src/hooks/use-session.ts` owns API state, action admission and operation polling.
`src/components/chat/` renders the assistant conversation and scientific attachments;
`src/components/session/` contains batch summaries and session inspection controls.
`dataset-loader.tsx` and `json-block.tsx` are shared components, while `components/ui/`
contains the reusable interface primitives. `src/lib/api.ts` handles requests and safe
external links.

The conversation uses `chat/markdown-answer.tsx` with scoped `.chat-markdown`
styles in `src/index.css`. Keep Markdown semantics intact: ordered lists retain
their numbers, while tables and fenced code scroll within the answer. Raw HTML
is not enabled. Fresh sessions open directly into chat, without requiring a batch.
The chat fills the viewport and keeps the composer at the bottom; long conversations
use document scrolling, with diagnostics above the composer.
The composer plus button opens the shared CSV loader and optional seed sample; uploading
replaces the session rather than appending candidates. A successful replacement
also clears the old message draft. Failed uploads preserve the current session.

The left sidebar contains New chat and earlier conversations; on mobile, Open chats
opens it as a drawer. New chat keeps the batch, constraints and evidence. Selecting
a conversation restores its assistant context. Drafts stay
with their chat while the page is open. A stable chat ID identifies conversations;
switching rotates session authority so stale requests and old proposals cannot apply.
All chats remain in JSON export; the header offers export without a batch. Chats are
in memory and do not survive server restart or dataset replacement without exporting.

Builds are verified with Node 26.5.0 (see `.node-version`) and npm 11.17.0.
Use the committed lockfile when installing:

```bash
cd ui && npm ci
npm run build   # emits index.html, app.js, style.css into ../src/workbench_web
npm run dev     # http://127.0.0.1:5173, proxies /api to 127.0.0.1:8765
```

Run the Python workbench on its default port, 8765, before opening the development
UI. The proxy translates only the exact local development Origin; Flask still
checks its Host, Origin, and CSRF token. Port 5173 is fixed rather than silently
falling back to another port. `src/workbench_web/` contains generated build output;
edit `ui/src/` and rebuild instead of editing the emitted files.

The build deliberately emits fixed filenames so the server's `/assets/` allowlist
(`app.js`, `style.css`) needs no change. The server's CSP must keep
`style-src 'self' 'unsafe-inline'`: the component library positions popups with
element styles.
