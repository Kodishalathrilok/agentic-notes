import Icon from './Icons'

const MODES = [
  { value: 'exam', label: 'Exam' },
  { value: 'revision', label: 'Revision' },
  { value: 'deep study', label: 'Deep Study' },
  { value: 'summary', label: 'Summary' },
]
const TONES = [
  { value: 'academic', label: 'Academic' },
  { value: 'formal', label: 'Formal' },
  { value: 'casual', label: 'Casual' },
  { value: 'simple', label: 'Simple' },
]
const LENGTHS = [
  { value: 'short', label: 'Short' },
  { value: 'medium', label: 'Medium' },
  { value: 'long', label: 'Long' },
]
const FORMATS = [
  { value: 'bullet', label: 'Bullets' },
  { value: 'numbered', label: 'Numbered' },
  { value: 'paragraph', label: 'Paragraph' },
]

// Segmented pill group — replaces the old native <select>s
function Segmented({ label, value, onChange, options, disabled }) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-semibold text-espresso-500">{label}</span>
      <div className="flex flex-wrap gap-1 rounded-2xl border border-espresso-900/10 bg-white/50 p-1">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            disabled={disabled}
            onClick={() => onChange(o.value)}
            className={`flex-1 whitespace-nowrap rounded-xl px-2.5 py-1.5 text-xs font-semibold transition-all duration-200 ${
              value === o.value
                ? 'bg-gradient-to-br from-espresso-800 to-espresso-900 text-cream shadow-soft'
                : 'text-espresso-600 hover:bg-espresso-900/5 hover:text-espresso-900'
            } disabled:opacity-60`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function ControlPanel({
  mode,
  setMode,
  tone,
  setTone,
  length,
  setLength,
  format,
  setFormat,
  model,
  setModel,
  models,
  instructions,
  setInstructions,
  onGenerate,
  isStreaming,
  canGenerate,
}) {
  const modelOptions = (models || []).map((m) => ({ value: m.id, label: m.label }))

  return (
    <div className="glass-card group/panel relative overflow-hidden rounded-3xl p-6 transition-shadow duration-300 hover:shadow-lift">
      <div className="pointer-events-none absolute -left-16 -bottom-16 h-40 w-40 rounded-full bg-rose-300/15 blur-2xl" />

      <h2 className="mb-4 flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-espresso-500">
        <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-gradient-to-br from-espresso-700 to-espresso-900 text-white shadow-soft">
          <Icon.Sparkles className="h-3.5 w-3.5" />
        </span>
        Settings
      </h2>

      <div className="space-y-3.5">
        <Segmented label="Mode" value={mode} onChange={setMode} options={MODES} disabled={isStreaming} />
        <Segmented label="Tone" value={tone} onChange={setTone} options={TONES} disabled={isStreaming} />
        <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2">
          <Segmented label="Length" value={length} onChange={setLength} options={LENGTHS} disabled={isStreaming} />
          <Segmented label="Format" value={format} onChange={setFormat} options={FORMATS} disabled={isStreaming} />
        </div>
      </div>

      {modelOptions.length > 0 && (
        <label className="mt-3.5 flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-espresso-500">Model</span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={isStreaming}
            className="field cursor-pointer bg-white/60 backdrop-blur-sm"
          >
            {modelOptions.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="mt-3.5 flex flex-col gap-1.5">
        <span className="text-xs font-semibold text-espresso-500">Custom instructions (optional)</span>
        <textarea
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
          disabled={isStreaming}
          rows={2}
          placeholder="e.g. focus on formulas, add mnemonics, use UK spelling…"
          className="field resize-y bg-white/60 backdrop-blur-sm"
        />
      </label>

      <button
        onClick={onGenerate}
        disabled={isStreaming || !canGenerate}
        className="gen-btn mt-5 flex w-full items-center justify-center gap-2 rounded-2xl px-5 py-3 text-sm font-bold text-white transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lift active:translate-y-0 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isStreaming ? (
          <>
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.4 0 0 5.4 0 12h4z" />
            </svg>
            Generating…
          </>
        ) : (
          <>
            <Icon.Sparkles className="h-4 w-4" />
            Generate Notes
          </>
        )}
      </button>
    </div>
  )
}