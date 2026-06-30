import { useState, useEffect, useCallback } from 'react'
import { supabase } from '../lib/supabase'

const HISTORY_KEY = 'agentic-notes-history'

// ---- localStorage backend ----
function loadLocal() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  } catch {
    return []
  }
}
function saveLocal(list) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list))
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
  const cloud = !!(user && supabase)

  const reload = useCallback(async () => {
    if (cloud) {
      const { data, error } = await supabase
        .from('sessions')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50)
      if (!error) setHistory((data || []).map(rowToSession))
    } else {
      setHistory(loadLocal())
    }
  }, [cloud])

  useEffect(() => {
    reload()
  }, [reload])

  const addSession = useCallback(
    async (session) => {
      if (cloud) {
        const { data } = await supabase.from('sessions').insert(sessionToRow(session)).select().single()
        if (data) setHistory((h) => [rowToSession(data), ...h])
      } else {
        const next = [session, ...loadLocal()].slice(0, 20)
        saveLocal(next)
        setHistory(next)
      }
    },
    [cloud]
  )

  const updateSession = useCallback(
    async (id, fields) => {
      if (cloud) {
        await supabase.from('sessions').update(fields).eq('id', id)
      } else {
        const next = loadLocal().map((s) => (s.id === id ? { ...s, ...fields } : s))
        saveLocal(next)
      }
      setHistory((h) => h.map((s) => (s.id === id ? { ...s, ...fields } : s)))
    },
    [cloud]
  )

  const deleteSession = useCallback(
    async (id) => {
      if (cloud) {
        await supabase.from('sessions').delete().eq('id', id)
      } else {
        saveLocal(loadLocal().filter((s) => s.id !== id))
      }
      setHistory((h) => h.filter((s) => s.id !== id))
    },
    [cloud]
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
