import { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { BrandIcon } from '@/components/brand-icon'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

const DOCS_URL = 'https://meshllm.cloud'

export function InviteFriendEmptyState({
  invitationReady,
  selectedModel,
  isPublicMesh
}: {
  invitationReady: boolean
  selectedModel: string
  isPublicMesh: boolean
}) {
  const [open, setOpen] = useState(false)

  if (isPublicMesh) {
    return (
      <div className="mx-auto w-full max-w-md space-y-4 px-2 text-center">
        <div className="flex justify-center">
          <BrandIcon className="h-12 w-12 text-primary/50 animate-wiggle" />
        </div>
        <p className="text-sm text-muted-foreground">
          Mesh LLM is a project to let people contribute spare compute, build private personal AI, using open models.
        </p>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-controls="invite-friend-details"
          aria-expanded={open}
          className="mx-auto flex items-center gap-1.5 text-xs text-muted-foreground/70 hover:text-foreground transition-colors"
        >
          <ChevronDown className={cn('h-3 w-3 transition-transform', open ? '' : '-rotate-90')} />
          <span>Learn more…</span>
        </button>
        {open ? (
          <div id="invite-friend-details" className="space-y-4 rounded-md border border-dashed p-3 text-left">
            <div className="text-xs text-muted-foreground">
              <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                Learn about this project →
              </a>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Contribute to the pool</div>
              <div className="text-xs text-muted-foreground">
                Have a spare machine? Add it to this mesh and share compute with others.
              </div>
              <code className="block rounded-md border bg-muted/40 px-2 py-1.5 text-xs">mesh-llm --auto</code>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Run your own private mesh</div>
              <div className="text-xs text-muted-foreground">
                Pool machines across your home, office, or friends — fully private, no cloud needed.{' '}
                <a
                  href={DOCS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Getting started →
                </a>
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Use with coding agents</div>
              <div className="text-xs text-muted-foreground">
                Works with Claude Code, Goose, pi, and any OpenAI-compatible client.{' '}
                <a
                  href={`${DOCS_URL}/#agents`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Agent setup →
                </a>
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <div className="text-xs font-medium">Agent gossip</div>
              <div className="text-xs text-muted-foreground">
                Let your agents coordinate across machines — share status, findings, and questions. Works with any LLM
                setup.{' '}
                <a
                  href={`${DOCS_URL}/#blackboard`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline hover:text-foreground"
                >
                  Blackboard docs →
                </a>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className="mx-auto w-full max-w-md space-y-3 px-2 text-center">
      <div className="flex justify-center">
        <BrandIcon className="h-12 w-12 text-primary/50 animate-wiggle" />
      </div>
      <p className="text-sm text-muted-foreground">
        Mesh LLM lets you build private personal AI, using open models.{' '}
        <a href={DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
          Learn more →
        </a>
      </p>
      <div className="space-y-2 rounded-md border border-dashed p-3 text-left">
        <div className="text-xs font-medium">
          {invitationReady ? 'Private mesh invitation ready' : 'Private mesh connection'}
        </div>
        <div className="text-xs text-muted-foreground">
          {selectedModel ? `Selected model: ${selectedModel}` : 'Model selection will be chosen automatically.'}
        </div>
        <div className="text-xs text-muted-foreground">
          Use the mesh connection controls to securely add another machine.
        </div>
      </div>
    </div>
  )
}
