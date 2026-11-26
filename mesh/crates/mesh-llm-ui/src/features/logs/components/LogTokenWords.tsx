import { cn } from '@/lib/cn'

type LogTokenWordsProps = {
  readonly words: string
  readonly token: string
  readonly className?: string
}

/** A recorded token read as plain words with its code kept small beside it (receipt rule R3). */
export function LogTokenWords({ words, token, className }: LogTokenWordsProps) {
  return (
    <>
      {words}{' '}
      <code className={cn('font-mono text-[length:var(--density-type-caption)] opacity-70', className)}>{token}</code>
    </>
  )
}
