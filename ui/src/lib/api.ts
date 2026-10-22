let csrf = ""

export function setCsrf(token: string) {
  csrf = token
}

type Options = { form?: boolean; download?: boolean }

export async function request(path: string, payload?: unknown, { form = false, download = false }: Options = {}) {
  const init: RequestInit = {}
  if (payload !== undefined) {
    init.method = "POST"
    init.headers = { "X-CSRF-Token": csrf }
    init.body = form ? (payload as FormData) : JSON.stringify(payload)
    if (!form) (init.headers as Record<string, string>)["Content-Type"] = "application/json"
  }
  const response = await fetch(path, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}) as any)
    throw new Error(body?.error?.message || `Request failed (${response.status}).`)
  }
  return download ? response.blob() : response.json()
}

/** IDs are shown in JSON notation so spaces and line breaks stay distinguishable. */
export const exact = (value: unknown) => JSON.stringify(value)

export function safeUrl(value: unknown): string | null {
  if (typeof value !== "string") return null
  for (const character of value) {
    if (character.charCodeAt(0) <= 0x20) return null
  }
  try {
    const url = new URL(value)
    return ["http:", "https:"].includes(url.protocol) ? url.href : null
  } catch {
    return null
  }
}
