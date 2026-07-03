/**
 * Lightweight, dependency-free text animation (motion-primitives-style API).
 * Splits text per word or per char and reveals each unit with a staggered
 * CSS animation. Presets: 'slide' | 'fade' | 'blur'.
 *
 * <TextEffect per="word" preset="slide" as="h1">Animate me</TextEffect>
 */
export default function TextEffect({
  children,
  per = 'word',
  as: Tag = 'span',
  preset = 'slide',
 className = '',
  unitClassName = '',
  stagger = 0.06,
  startDelay = 0,
}) {
  const text = String(children ?? '')
  const units = per === 'char' ? Array.from(text) : text.split(' ')

  const nodes = []
  units.forEach((unit, i) => {
    nodes.push(
      <span
        key={i}
 className={`te-${preset} inline-block ${unitClassName}`}
        style={{ animationDelay: `${startDelay + i * stagger}s` }}
      >
        {unit}
      </span>
    )
    if (per === 'word' && i < units.length - 1) nodes.push(' ')
  })

  return <Tag className={className}>{nodes}</Tag>
}
