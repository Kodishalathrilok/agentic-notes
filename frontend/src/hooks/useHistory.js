import { useState, useEffect, useCallback } from 'react'
import { supabase } from '../lib/supabase'
import { HISTORY_KEY, readJSON, writeJSON } from '../lib/storage'

const GUEST = 'guest'

// ---- localStorage backend (namespaced per account) ----
function loadLocal(accountId) {
  const list = readJSON(HISTORY_KEY, accountId, [])
  return Array.isArray(list) ? list : []
}
function saveLocal(accountId, list) {
  writeJSON(HISTORY_KEY, accountId, list)
}

// ---- mapping between DB rows and the UI session shape ----
function rowToSession(r) {
  return {
    id: r.id,
    date: r.created_at,
    mode: r.mode,
    title: r.title || '',
    tags: r.tags || [],
    notes_preview: (r.notes || '').slice(0, 100),
    notes: r.notes || '',
    quiz: r.quiz || '',
    flashcards: r.flashcards || '',
    sources: r.sources || [],
    is_public: r.is_public,
  }
}
function sessionToRow(s) {
  return {
    title: s.title || '',
    mode: s.mode,
    tags: s.tags || [],
    notes: s.notes || '',
    quiz: s.quiz || '',
    flashcards: s.flashcards || '',
    sources: s.sources || [],
  }
}

/**
 * Unified history store. When `user` is signed in (and Supabase is configured),
 * reads/writes go to the cloud `sessions` table; otherwise localStorage.
 */
export default function useHistory(user) {
  const [history, setHistory] = useState([])
  // Key everything on the user ID, not on a `signed in?` boolean. With a
  // boolean, going straight from one account to another (a session swap with
  // no signed-out moment in between) left `reload` unchanged, so the effect
  // never re-ran and the first account's sessions stayed on screen.
  const userId = user?.id || null
  const accountId = userId || GUEST
  const cloud = !!(userId && supabase)

  // Blank the list the instant the account changes, so the previous user's
  // rows aren't rendered during the fetch that replaces them.
  useEffect(() => {
    setHistory([])
  }, [accountId])

  const reload = useCallback(async () => {
    if (cloud) {
      const { data, error } = await supabase
        .from('sessions')
        .select('*')
        // Explicit owner filter. RLS also enforces this, but its "read public
        // sessions" policy deliberately exposes every is_public row to every
        // signed-in caller — so an unfiltered select pulled other people's
        // shared notes into this user's History.
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(50)
      if (!error) setHistory((data || []).map(rowToSession))
    } else {
      setHistory(loadLocal(accountId))
    }
  }, [cloud, userId, accountId])

  useEffect(() => {
    reload()
  }, [reload])

  const addSession = useCallback(
    async (session) => {
      if (cloud) {
        const { data } = await supabase.from('sessions').insert(sessionToRow(session)).select().single()
        if (data) setHistory((h) => [rowToSession(data), ...h])
      } else {
        const next = [session, ...loadLocal(accountId)].slice(0, 20)
        saveLocal(accountId, next)
        setHistory(next)
      }
    },
    [cloud, accountId]
  )

  const updateSession = useCallback(
    async (id, fields) => {
      if (cloud) {
        await supabase.from('sessions').update(fields).eq('id', id)
      } else {
        const next = loadLocal(accountId).map((s) => (s.id === id ? { ...s, ...fields } : s))
        saveLocal(accountId, next)
      }
      setHistory((h) => h.map((s) => (s.id === id ? { ...s, ...fields } : s)))
    },
    [cloud, accountId]
  )

  const deleteSession = useCallback(
    async (id) => {
      if (cloud) {
        await supabase.from('sessions').delete().eq('id', id)
      } else {
        saveLocal(accountId, loadLocal(accountId).filter((s) => s.id !== id))
      }
      setHistory((h) => h.filter((s) => s.id !== id))
    },
    [cloud, accountId]
  )

  // Make a session public and return its share id (cloud only).
  const shareSession = useCallback(
    async (id) => {
      if (!cloud) return null
      await supabase.from('sessions').update({ is_public: true }).eq('id', id)
      setHistory((h) => h.map((s) => (s.id === id ? { ...s, is_public: true } : s)))
      return id
    },
    [cloud]
  )

  return { history, cloud, reload, addSession, updateSession, deleteSession, shareSession }
}
