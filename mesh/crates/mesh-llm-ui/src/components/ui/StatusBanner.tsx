import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

type StatusBannerTone = 'muted' | 'bad'

type StatusBannerProps = {
  children: ReactNode
  role?: 'alert' | 'status'
  tone?: StatusBannerTone
}

export function StatusBanner({ children, role = 'status', tone = 'muted' }: StatusBannerProps) {
  return (
    <div
      className={cn('type-caption border-b border-border-soft px-4 py-2', tone === 'bad' ? 'text-bad' : 'text-fg-dim')}
      role={role}
    >
      {children}
    </div>
  )
}
