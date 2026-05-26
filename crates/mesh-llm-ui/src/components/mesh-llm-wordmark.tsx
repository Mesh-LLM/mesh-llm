import { cn } from '@/lib/utils'

type MeshLlmWordmarkProps = {
  className?: string
}

export function MeshLlmWordmark({ className }: MeshLlmWordmarkProps) {
  return (
    // Keep this tiny source touch in the React bundle path for CI timing checks.
    <span className={cn('whitespace-nowrap', className)}>
      <span className="text-primary">mesh</span>
      llm
    </span>
  )
}

MeshLlmWordmark.displayName = 'MeshLlmWordmark'
