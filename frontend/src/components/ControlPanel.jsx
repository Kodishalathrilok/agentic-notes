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
  { value: 'bullet', label: 'Bullet Points' },
  { value: 'numbered', label: 'Numbered List' },
  { value: 'paragraph', label: 'Paragraph' },
]

function Select({ label, value, onChange, options, disabled }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="field cursor-pointer"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
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
    <div className="card p-5">
      <h2 className="mb-4 text-xs font-bold uppercase tracking-[0.2em] text-stone-500">
        Settings
      </h2>

      <div className="grid grid-cols-2 gap-3">
        <Select label="Mode" value={mode} onChange={setMode} options={MODES} disabled={isStreaming} />
        <Select label="Tone" value={tone} onChange={setTone} options={TONES} disabled={isStreaming} />
        <Select label="Length" value={length} onChange={setLength} options={LENGTHS} disabled={isStreaming} />
        <Select label="Format" value={format} onChange={setFormat} options={FORMATS} disabled={isStreaming} />
      </div>

      {modelOptions.length > 0 && (
        <div className="mt-3">
          <Select
            label="Model"
            value={model}
            onChange={setModel}
            options={modelOptions}
            disabled={isStreaming}
          />
        </div>
      )}

      <label className="mt-3 flex flex-col gap-1">
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
          Custom instructions (optional)
        </span>
        <textarea
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
          disabled={isStreaming}
          rows={2}
          placeholder="e.g. focus on formulas, add mnemonics, use UK spelling…"
          className="field resize-y"
        />
      </label>

      <button
        onClick={onGenerate}
        disabled={isStreaming || !canGenerate}
        className="btn-primary mt-5 w-full py-2.5"
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
