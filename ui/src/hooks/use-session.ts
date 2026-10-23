import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { request, setCsrf } from "@/lib/api"

type State = any

export function useSession() {
  const [state, setState] = useState<State | null>(null)
  const [assistantAvailable, setAssistantAvailable] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const stateRef = useRef<State | null>(null)
  const submittingRef = useRef(false)
  const timer = useRef<number | undefined>(undefined)

  const commit = useCallback((next: State) => {
    stateRef.current = next
    setState(next)
  }, [])

  const readState = useCallback(async (sessionId: string) => {
    const body = await request(`/api/state?${new URLSearchParams({ session_id: sessionId })}`)
    commit(body.state)
    return body.state
  }, [commit])

  // Expiry marks a failed operation, but its worker may still be unwinding.
  const poll = useCallback((operationId: string, sessionId: string) => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(async () => {
      try {
        await request(`/api/operation?${new URLSearchParams({ session_id: sessionId, operation_id: operationId })}`)
        const next = await readState(sessionId)
        if (next.busy) poll(operationId, sessionId)
      } catch (error: any) {
        toast.error(`${error.message} Reload state to reconnect; no work will be repeated.`)
      }
    }, 500)
  }, [readState])

  const loadState = useCallback(async () => {
    const previous = stateRef.current?.session_id
    const body = await request("/api/bootstrap")
    setCsrf(body.csrf_token)
    setAssistantAvailable(body.assistant_available)
    commit(body.state)
    if (previous && previous !== body.state.session_id) {
      toast.warning("The server session changed. Earlier unexported work is no longer in this session. No work was repeated.")
    }
    window.clearTimeout(timer.current)
    if (body.state.busy && body.state.operation) poll(body.state.operation.operation_id, body.state.session_id)
  }, [commit, poll])

  const act = useCallback(async (action: () => Promise<void>) => {
    if (submittingRef.current) return false
    submittingRef.current = true
    setSubmitting(true)
    try {
      await action()
      return true
    } catch (error: any) {
      toast.error(error.message)
      return false
    } finally {
      submittingRef.current = false
      setSubmitting(false)
    }
  }, [])

  const keys = useCallback((withEvidence = false) => {
    const current = stateRef.current
    const result: Record<string, unknown> = {
      session_id: current.session_id,
      dataset_digest: current.dataset_digest,
      base_revision: current.revision,
    }
    if (withEvidence) result.evidence_version = current.evidence_version
    return result
  }, [])

  const mutate = useCallback(async (path: string, payload: unknown) => {
    const body = await request(path, payload)
    commit(body.state)
  }, [commit])

  const startJob = useCallback(async (path: string, payload: unknown) => {
    const body = await request(path, payload)
    const sessionId = stateRef.current.session_id
    const next = await readState(sessionId)
    if (next.busy) poll(body.operation_id, sessionId)
  }, [poll, readState])

  useEffect(() => {
    act(loadState)
    return () => window.clearTimeout(timer.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const busy = submitting || !!state?.busy

  return { state, assistantAvailable, submitting, busy, act, keys, mutate, startJob, loadState, commit }
}
