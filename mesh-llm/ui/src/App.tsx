import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react';
import {
  Bot,
  Braces,
  Check,
  Circle,
  Copy,
  Cpu,
  Gauge,
  Hash,
  Laptop,
  Loader2,
  Moon,
  Network,
  RotateCcw,
  Send,
  Sparkles,
  Sun,
  UserPlus,
  User,
  Wifi,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './components/ui/accordion';
import { Alert, AlertDescription, AlertTitle } from './components/ui/alert';
import { Badge } from './components/ui/badge';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from './components/ui/navigation-menu';
import { Popover, PopoverContent, PopoverTrigger } from './components/ui/popover';
import { ScrollArea } from './components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Separator } from './components/ui/separator';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './components/ui/table';
import { Textarea } from './components/ui/textarea';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip';
import { BrandIcon } from './components/brand-icon';
import { MeshLlmWordmark } from './components/mesh-llm-wordmark';
import { cn } from './lib/utils';

type MeshModel = {
  name: string;
  status: 'warm' | 'cold' | string;
  node_count: number;
  size_gb: number;
};

type Peer = {
  id: string;
  role: string;
  models: string[];
  vram_gb: number;
  serving?: string | null;
};

type StatusPayload = {
  node_id: string;
  token: string;
  api_key_token?: string;
  node_status: string;
  is_host: boolean;
  is_client: boolean;
  llama_ready: boolean;
  model_name: string;
  api_port: number;
  my_vram_gb: number;
  model_size_gb: number;
  mesh_name?: string | null;
  peers: Peer[];
  mesh_models: MeshModel[];
  inflight_requests: number;
};

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  model?: string;
  stats?: string;
  error?: boolean;
};

type TopSection = 'dashboard' | 'chat';

type TopologyNode = {
  id: string;
  vram: number;
  self: boolean;
  host: boolean;
  client: boolean;
  serving: string;
};

type ThemeMode = 'auto' | 'light' | 'dark';

const THEME_STORAGE_KEY = 'mesh-llm-theme';

function readThemeMode(): ThemeMode {
  if (typeof window === 'undefined') return 'auto';
  const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
  return stored === 'light' || stored === 'dark' || stored === 'auto' ? stored : 'auto';
}

function applyThemeMode(mode: ThemeMode) {
  if (typeof window === 'undefined') return;
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const dark = mode === 'dark' || (mode === 'auto' && media.matches);
  document.documentElement.classList.toggle('dark', dark);
  document.documentElement.style.colorScheme = mode === 'auto' ? 'light dark' : dark ? 'dark' : 'light';
}

function nextThemeMode(mode: ThemeMode): ThemeMode {
  if (mode === 'auto') return 'light';
  if (mode === 'light') return 'dark';
  return 'auto';
}

export function App() {
  const [section, setSection] = useState<TopSection>('dashboard');
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode());
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [reasoningOpen, setReasoningOpen] = useState<Record<string, boolean>>({});
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const warmModels = useMemo(() => {
    const list = (status?.mesh_models ?? []).filter((m) => m.status === 'warm').map((m) => m.name);
    if (!list.length && status?.model_name) list.push(status.model_name);
    return list;
  }, [status]);

  const inviteCommand = useMemo(() => {
    const token = status?.token ?? '';
    const model = selectedModel || warmModels[0] || status?.model_name || '';
    return token && model ? `mesh-llm --join ${token} --model ${model}` : '';
  }, [selectedModel, status?.model_name, status?.token, warmModels]);
  const inviteToken = status?.token ?? '';
  const apiDirectUrl = useMemo(() => {
    const port = status?.api_port ?? 9337;
    return `http://127.0.0.1:${port}/v1`;
  }, [status?.api_port]);
  const apiKeyToken = status?.api_key_token ?? '';

  useEffect(() => {
    if (!warmModels.length) return;
    if (!selectedModel || !warmModels.includes(selectedModel)) setSelectedModel(warmModels[0]);
  }, [warmModels, selectedModel]);

  useEffect(() => {
    applyThemeMode(themeMode);
    window.localStorage.setItem(THEME_STORAGE_KEY, themeMode);
  }, [themeMode]);

  useEffect(() => {
    if (themeMode !== 'auto') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyThemeMode('auto');
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [themeMode]);

  useEffect(() => {
    let stop = false;

    const loadStatus = () => {
      fetch('/api/status')
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json() as Promise<StatusPayload>;
        })
        .then((data) => {
          if (stop) return;
          setStatus(data);
          setStatusError(null);
        })
        .catch((err: Error) => {
          if (!stop) {
            setStatusError(`Failed to fetch status (${err.message})`);
            console.warn('Failed to fetch /api/status:', err.message);
          }
        });
    };

    loadStatus();

    const statusEvents = new EventSource('/api/events');
    statusEvents.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as StatusPayload;
        setStatus(payload);
        setStatusError(null);
      } catch {
        // ignore malformed status event
      }
    };
    statusEvents.onerror = () => {
      setStatusError('Status stream disconnected. Retrying...');
      console.warn('Status stream disconnected. Retrying...');
    };

    return () => {
      stop = true;
      statusEvents.close();
    };
  }, []);

  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isSending]);

  const canChat = !!status && (status.llama_ready || (status.is_client && warmModels.length > 0));

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !status || isSending) return;

    const model = selectedModel || status.model_name;
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: trimmed, model };
    const assistantId = crypto.randomUUID();
    const assistantMessage: ChatMessage = { id: assistantId, role: 'assistant', content: '', model };
    const historyForRequest = [...messages, userMessage];

    setMessages([...historyForRequest, assistantMessage]);
    setInput('');
    setIsSending(true);

    const reqStart = performance.now();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: historyForRequest.map((m) => ({ role: m.role, content: m.content })),
          stream: true,
          stream_options: { include_usage: true },
        }),
      });

      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      let full = '';
      let reasoning = '';
      let completionTokens: number | null = null;
      let firstTokenAt: number | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (!data || data === '[DONE]') continue;
          try {
            const chunk = JSON.parse(data) as {
              usage?: { completion_tokens?: number };
              choices?: Array<{ delta?: { content?: string; reasoning_content?: string } }>;
            };
            const delta = chunk.choices?.[0]?.delta;
            if (Number.isFinite(chunk.usage?.completion_tokens)) completionTokens = chunk.usage!.completion_tokens!;
            const contentDelta = delta?.content ?? '';
            const reasoningDelta = delta?.reasoning_content ?? '';
            if (!contentDelta && !reasoningDelta) continue;
            if (firstTokenAt == null) firstTokenAt = performance.now();
            full += contentDelta;
            reasoning += reasoningDelta;
            setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: full, reasoning: reasoning || undefined } : m)));
          } catch {
            // ignore malformed chunk
          }
        }
      }

      const endAt = performance.now();
      const genStart = firstTokenAt ?? reqStart;
      const genSecs = Math.max(0.001, (endAt - genStart) / 1000);
      const ttftMs = Math.max(0, Math.round((firstTokenAt ?? endAt) - reqStart));
      const tokenCount = Number.isFinite(completionTokens) ? completionTokens! : Math.max(1, Math.round(Math.max(full.length, 1) / 4));
      const tps = tokenCount / genSecs;
      const stats = `${tokenCount} tok · ${tps.toFixed(1)} tok/s · TTFT ${ttftMs}ms`;

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: m.content || '(empty response)', reasoning: m.reasoning || undefined, stats }
            : m,
        ),
      );

    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: `Error: ${message}`, error: true } : m)));
    } finally {
      setIsSending(false);
    }
  }

  const topologyNodes = useMemo<TopologyNode[]>(() => {
    if (!status) return [];
    const nodes: TopologyNode[] = [];
    if (status.node_id) {
      nodes.push({
        id: status.node_id,
        vram: status.my_vram_gb || 0,
        self: true,
        host: status.is_host,
        client: status.is_client,
        serving: status.model_name || '',
      });
    }
    for (const p of status.peers ?? []) {
      nodes.push({
        id: p.id,
        vram: p.vram_gb,
        self: false,
        host: /^Host/.test(p.role),
        client: p.role === 'Client',
        serving: p.serving || '',
      });
    }
    return nodes;
  }, [status]);

  const sections: Array<{ key: TopSection; label: string }> = [
    { key: 'dashboard', label: 'Dashboard' },
    { key: 'chat', label: 'Chat' },
  ];

  function handleSubmit() {
    if (!canChat) return;
    void sendMessage(input);
  }

  return (
    <div className="h-screen overflow-hidden bg-background [height:100svh]">
      <div className="flex h-full min-h-0 flex-col">
        <AppHeader
          sections={sections}
          section={section}
          setSection={setSection}
          themeMode={themeMode}
          setThemeMode={setThemeMode}
          statusError={statusError}
          inviteCommand={inviteCommand}
          inviteToken={inviteToken}
          apiDirectUrl={apiDirectUrl}
          apiKeyToken={apiKeyToken}
          onApiKeyReset={(token) =>
            setStatus((prev) => (prev ? { ...prev, api_key_token: token } : prev))
          }
        />

        <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
          {section === 'chat' ? (
            <div className="mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col p-4">
              <ChatPage
                inviteToken={status?.token ?? ''}
                warmModels={warmModels}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                messages={messages}
                reasoningOpen={reasoningOpen}
                setReasoningOpen={setReasoningOpen}
                chatScrollRef={chatScrollRef}
                input={input}
                setInput={setInput}
                isSending={isSending}
                canChat={canChat}
                onSubmit={handleSubmit}
              />
            </div>
          ) : null}

          {section === 'dashboard' ? (
            <div className="min-h-0 flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-7xl p-4">
                <DashboardPage
                  status={status}
                  topologyNodes={topologyNodes}
                  selectedModel={selectedModel || status?.model_name || ''}
                  themeMode={themeMode}
                />
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

function AppHeader({
  sections,
  section,
  setSection,
  themeMode,
  setThemeMode,
  statusError,
  inviteCommand,
  inviteToken,
  apiDirectUrl,
  apiKeyToken,
  onApiKeyReset,
}: {
  sections: Array<{ key: TopSection; label: string }>;
  section: TopSection;
  setSection: React.Dispatch<React.SetStateAction<TopSection>>;
  themeMode: ThemeMode;
  setThemeMode: React.Dispatch<React.SetStateAction<ThemeMode>>;
  statusError: string | null;
  inviteCommand: string;
  inviteToken: string;
  apiDirectUrl: string;
  apiKeyToken: string;
  onApiKeyReset: (token: string) => void;
}) {
  const [inviteCopied, setInviteCopied] = useState(false);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [apiDirectCopied, setApiDirectCopied] = useState(false);
  const [apiTokenCopied, setApiTokenCopied] = useState(false);
  const [isResettingApiToken, setIsResettingApiToken] = useState(false);

  async function copyInviteCommand() {
    if (!inviteCommand) return;
    try {
      await navigator.clipboard.writeText(inviteCommand);
      setInviteCopied(true);
      window.setTimeout(() => setInviteCopied(false), 1500);
    } catch {
      setInviteCopied(false);
    }
  }

  async function copyInviteToken() {
    if (!inviteToken) return;
    try {
      await navigator.clipboard.writeText(inviteToken);
      setTokenCopied(true);
      window.setTimeout(() => setTokenCopied(false), 1500);
    } catch {
      setTokenCopied(false);
    }
  }

  async function copyApiDirectUrl() {
    if (!apiDirectUrl) return;
    try {
      await navigator.clipboard.writeText(apiDirectUrl);
      setApiDirectCopied(true);
      window.setTimeout(() => setApiDirectCopied(false), 1500);
    } catch {
      setApiDirectCopied(false);
    }
  }

  async function copyApiKeyToken() {
    if (!apiKeyToken) return;
    try {
      await navigator.clipboard.writeText(apiKeyToken);
      setApiTokenCopied(true);
      window.setTimeout(() => setApiTokenCopied(false), 1500);
    } catch {
      setApiTokenCopied(false);
    }
  }

  async function resetApiKeyToken() {
    if (isResettingApiToken) return;
    setIsResettingApiToken(true);
    try {
      const response = await fetch('/api/api-key/reset', { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = (await response.json()) as { api_key_token?: string };
      if (body.api_key_token) {
        onApiKeyReset(body.api_key_token);
        setApiTokenCopied(false);
      }
    } catch (err) {
      console.warn('Failed to reset API token:', err);
    } finally {
      setIsResettingApiToken(false);
    }
  }

  return (
    <header className="shrink-0 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex h-16 w-full max-w-7xl items-center gap-4 px-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center">
          <BrandIcon className="h-5 w-5 text-primary" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-base font-semibold">
            <MeshLlmWordmark />
          </div>
        </div>
        <NavigationMenu>
          <NavigationMenuList>
            {sections.map(({ key, label }) => (
              <NavigationMenuItem key={key}>
                <NavigationMenuLink asChild>
                  <button
                    type="button"
                    onClick={() => setSection(key)}
                    className={navigationMenuTriggerStyle()}
                    data-active={section === key ? '' : undefined}
                    aria-current={section === key ? 'page' : undefined}
                  >
                    {label}
                  </button>
                </NavigationMenuLink>
              </NavigationMenuItem>
            ))}
          </NavigationMenuList>
        </NavigationMenu>
        <TooltipProvider>
          <div className="ml-auto flex items-center gap-2">
            <Popover>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <Button type="button" variant="outline" size="icon" aria-label="API access">
                      <Braces className="h-4 w-4" />
                    </Button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>API</TooltipContent>
              </Tooltip>
              <PopoverContent className="w-[420px] space-y-3" align="end">
              <div className="space-y-1">
                <div className="text-sm font-medium">API Access</div>
                <div className="text-xs text-muted-foreground">Use this endpoint and API key token.</div>
              </div>
              <div className="space-y-2">
                <div className="text-xs font-medium text-muted-foreground">Direct Endpoint URL</div>
                {apiDirectUrl ? (
                  <code className="block overflow-x-auto whitespace-nowrap rounded-md border bg-muted/40 px-2 py-1.5 text-xs">
                    {apiDirectUrl}
                  </code>
                ) : (
                  <div className="text-xs text-muted-foreground">Direct endpoint unavailable until status is loaded.</div>
                )}
                <Button type="button" size="sm" variant="secondary" disabled={!apiDirectUrl} onClick={() => void copyApiDirectUrl()}>
                  {apiDirectCopied ? <Check className="mr-1.5 h-4 w-4" /> : <Copy className="mr-1.5 h-4 w-4" />}
                  {apiDirectCopied ? 'Copied' : 'Copy endpoint URL'}
                </Button>
              </div>
              <div className="space-y-2">
                <div className="text-xs font-medium text-muted-foreground">API Key Token</div>
                {apiKeyToken ? (
                  <code className="block overflow-x-auto whitespace-nowrap rounded-md border bg-muted/40 px-2 py-1.5 text-xs">
                    {apiKeyToken}
                  </code>
                ) : (
                  <div className="text-xs text-muted-foreground">Token unavailable until status is loaded.</div>
                )}
                <div className="flex flex-wrap items-center gap-2">
                  <Button type="button" size="sm" variant="outline" disabled={!apiKeyToken} onClick={() => void copyApiKeyToken()}>
                    {apiTokenCopied ? <Check className="mr-1.5 h-4 w-4" /> : <Copy className="mr-1.5 h-4 w-4" />}
                    {apiTokenCopied ? 'Copied' : 'Copy token'}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={isResettingApiToken}
                    onClick={() => void resetApiKeyToken()}
                  >
                    {isResettingApiToken ? (
                      <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />
                    ) : (
                      <RotateCcw className="mr-1.5 h-4 w-4" />
                    )}
                    {isResettingApiToken ? 'Resetting...' : 'Reset token'}
                  </Button>
                </div>
              </div>
              </PopoverContent>
            </Popover>
            <Popover>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger asChild>
                    <Button
                      type="button"
                      variant="outline"
                      size="icon"
                      aria-label="Invite"
                      disabled={!inviteCommand}
                    >
                      <UserPlus className="h-4 w-4" />
                    </Button>
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent>Invite</TooltipContent>
              </Tooltip>
              <PopoverContent className="w-[420px] space-y-3" align="end">
              <div className="space-y-1">
                <div className="text-sm font-medium">Invite to this mesh</div>
                <div className="text-xs text-muted-foreground">Share the join command or token.</div>
              </div>
              {inviteCommand ? (
                <code className="block overflow-x-auto whitespace-nowrap rounded-md border bg-muted/40 px-2 py-1.5 text-xs">
                  {inviteCommand}
                </code>
              ) : (
                <div className="text-xs text-muted-foreground">No invite command available yet.</div>
              )}
              <div className="flex flex-wrap items-center gap-2">
                <Button type="button" size="sm" variant="secondary" disabled={!inviteCommand} onClick={() => void copyInviteCommand()}>
                  {inviteCopied ? <Check className="mr-1.5 h-4 w-4" /> : <Copy className="mr-1.5 h-4 w-4" />}
                  {inviteCopied ? 'Command copied' : 'Copy command'}
                </Button>
                <Button type="button" size="sm" variant="outline" disabled={!inviteToken} onClick={() => void copyInviteToken()}>
                  {tokenCopied ? <Check className="mr-1.5 h-4 w-4" /> : <Copy className="mr-1.5 h-4 w-4" />}
                  {tokenCopied ? 'Token copied' : 'Copy token'}
                </Button>
              </div>
              </PopoverContent>
            </Popover>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  aria-label={`Theme ${themeMode}. Click to cycle Auto, Light, Dark`}
                  onClick={() => setThemeMode((prev) => nextThemeMode(prev))}
                >
                  {themeMode === 'auto' ? <Laptop className="h-4 w-4" /> : null}
                  {themeMode === 'light' ? <Sun className="h-4 w-4" /> : null}
                  {themeMode === 'dark' ? <Moon className="h-4 w-4" /> : null}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Theme: {themeMode}</TooltipContent>
            </Tooltip>
          </div>
        </TooltipProvider>
      </div>
      {statusError ? (
        <div className="mx-auto w-full max-w-7xl px-4 pb-3">
          <Alert variant="destructive">
            <Circle className="h-4 w-4" />
            <AlertTitle>Status connection issue</AlertTitle>
            <AlertDescription>{statusError}</AlertDescription>
          </Alert>
        </div>
      ) : null}
    </header>
  );
}

function ChatPage(props: {
  inviteToken: string;
  warmModels: string[];
  selectedModel: string;
  setSelectedModel: (v: string) => void;
  messages: ChatMessage[];
  reasoningOpen: Record<string, boolean>;
  setReasoningOpen: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  chatScrollRef: React.RefObject<HTMLDivElement>;
  input: string;
  setInput: (v: string) => void;
  isSending: boolean;
  canChat: boolean;
  onSubmit: () => void;
}) {
  const {
    inviteToken,
    warmModels,
    selectedModel,
    setSelectedModel,
    messages,
    reasoningOpen,
    setReasoningOpen,
    chatScrollRef,
    input,
    setInput,
    isSending,
    canChat,
    onSubmit,
  } = props;

  return (
    <Card className="flex h-full min-h-0 flex-1 flex-col overflow-hidden">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-3">
          <CardTitle className="text-base">Chat</CardTitle>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Model</span>
            <Select value={selectedModel || warmModels[0] || ''} onValueChange={setSelectedModel} disabled={!warmModels.length}>
              <SelectTrigger className="h-8 w-[220px]">
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                {warmModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    {shortName(model)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <Separator />
      <CardContent className="min-h-0 flex-1 p-0">
        <div ref={chatScrollRef} className="h-full space-y-4 overflow-y-auto px-4 py-4 md:px-6">
          {messages.length === 0 ? (
            <InviteFriendEmptyState inviteToken={inviteToken} selectedModel={selectedModel || warmModels[0] || ''} />
          ) : null}

          {messages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message}
              reasoningOpen={!!reasoningOpen[message.id]}
              onReasoningToggle={(open) => setReasoningOpen((prev) => ({ ...prev, [message.id]: open }))}
            />
          ))}

          {isSending ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Streaming response...
            </div>
          ) : null}
        </div>
      </CardContent>
      <Separator />
      <div className="space-y-3 p-4">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            }
          }}
          rows={4}
          placeholder={props.canChat ? 'Send a prompt to the mesh...' : 'Waiting for a warm model...'}
          disabled={!props.canChat || isSending}
          className="min-h-[112px] resize-none"
        />
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs text-muted-foreground">Enter to send. Shift+Enter for newline.</div>
          <Button onClick={onSubmit} disabled={!props.canChat || !input.trim() || isSending}>
            {isSending ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Send className="mr-2 h-4 w-4" />}
            Send
          </Button>
        </div>
      </div>
    </Card>
  );
}

function InviteFriendEmptyState({ inviteToken, selectedModel }: { inviteToken: string; selectedModel: string }) {
  const [copied, setCopied] = useState(false);
  const command = inviteToken && selectedModel ? `mesh-llm --join ${inviteToken} --model ${selectedModel}` : '';

  async function copy() {
    if (!command) return;
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Card className="mx-auto max-w-3xl border-dashed">
      <CardContent className="space-y-4 p-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg border">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Invite a friend to Mesh LLM</h2>
            <p className="text-sm text-muted-foreground">
              Open models exist. Excess compute exists. What&apos;s missing is coordination.
            </p>
          </div>
        </div>

        <div className="space-y-3 text-sm leading-6 text-muted-foreground">
          <p>
            Mesh LLM is a shared network for open AI inference. Instead of relying only on centralized infrastructure,
            people can contribute idle compute and gain access to collective capacity.
          </p>
          <p>
            Invite someone to join this mesh and load the current model. More participation increases available capacity
            and makes the network stronger.
          </p>
        </div>

        <div className="rounded-md border p-3">
          <div className="mb-2 text-xs font-medium text-muted-foreground">Join Command</div>
          {command ? (
            <div className="flex items-center gap-2">
              <code className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap rounded border px-2 py-1.5 text-xs">
                {command}
              </code>
              <Button type="button" variant="secondary" size="sm" onClick={() => void copy()}>
                {copied ? <Check className="mr-1 h-3.5 w-3.5" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </Button>
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">
              Start or join a mesh and select a model to generate an invite command.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function DashboardPage({
  status,
  topologyNodes,
  selectedModel,
  themeMode,
}: {
  status: StatusPayload | null;
  topologyNodes: TopologyNode[];
  selectedModel: string;
  themeMode: ThemeMode;
}) {
  const [modelFilter, setModelFilter] = useState<'all' | 'warm' | 'cold'>('all');
  const filteredModels = useMemo(() => {
    const models = status?.mesh_models ?? [];
    return [...models]
      .filter((m) => (modelFilter === 'all' ? true : m.status === modelFilter))
      .sort((a, b) => (b.node_count - a.node_count) || a.name.localeCompare(b.name));
  }, [status?.mesh_models, modelFilter]);

  return (
    <div className="space-y-4">
      <TooltipProvider>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <StatCard
          title="Node ID"
          value={status?.node_id ?? 'n/a'}
          icon={<Hash className="h-4 w-4" />}
          tooltip="Current node identifier in this mesh."
        />
        <StatCard
          title="Serving Model"
          value={status?.model_name ? shortName(status.model_name) : 'n/a'}
          icon={<Sparkles className="h-4 w-4" />}
          tooltip="Model currently served by this node."
        />
        <StatCard
          title="Mesh VRAM"
          value={`${meshGpuVram(status).toFixed(1)} GB`}
          icon={<Cpu className="h-4 w-4" />}
          tooltip="Total GPU VRAM across non-client nodes in the mesh."
        />
        <StatCard
          title="Nodes"
          value={`${topologyNodes.length}`}
          icon={<Network className="h-4 w-4" />}
          tooltip="Total nodes currently visible in topology."
        />
        <StatCard
          title="Inflight"
          value={`${status?.inflight_requests ?? 0}`}
          icon={<Gauge className="h-4 w-4" />}
          tooltip="Current in-flight request count."
        />
        </div>
      </TooltipProvider>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Network Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <MeshTopologyDiagram
            status={status}
            nodes={topologyNodes}
            selectedModel={selectedModel}
            themeMode={themeMode}
          />
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-7">
        <Card className="lg:col-span-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Connected Peers</CardTitle>
          </CardHeader>
          <CardContent className="min-h-0 pt-0">
            {(status?.peers.length ?? 0) > 0 ? (
              <ScrollArea className="h-[18rem] pr-3 md:h-[20rem]">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>ID</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead className="text-right">VRAM</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(status?.peers ?? []).map((peer) => (
                      <TableRow key={peer.id}>
                        <TableCell className="font-mono text-xs">{peer.id}</TableCell>
                        <TableCell>{peer.role}</TableCell>
                        <TableCell className="text-right">{peer.vram_gb.toFixed(1)} GB</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            ) : (
              <DashboardPanelEmpty
                icon={<Network className="h-4 w-4" />}
                title="No peers connected"
                description="Invite another node to join this mesh to see connected peers."
              />
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm">Model Catalog</CardTitle>
              <div className="ml-auto flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Filter</span>
                <Select value={modelFilter} onValueChange={(v) => setModelFilter(v as 'all' | 'warm' | 'cold')}>
                  <SelectTrigger className="h-8 w-[110px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="warm">Warm</SelectItem>
                    <SelectItem value="cold">Cold</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent className="min-h-0 pt-0">
            {filteredModels.length > 0 ? (
              <ScrollArea className="h-[18rem] pr-3 md:h-[20rem]">
                <div className="space-y-2 pr-2">
                  {filteredModels.map((model) => (
                    <div key={model.name} className="rounded-md border p-3">
                      <div className="flex items-center gap-2">
                        <div className="flex h-7 w-7 items-center justify-center rounded-md border bg-muted/40 text-muted-foreground">
                          <Sparkles className="h-3.5 w-3.5" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium">{shortName(model.name)}</div>
                          <div className="truncate text-xs text-muted-foreground">{model.name}</div>
                        </div>
                        <Badge
                          className={cn(
                            'gap-1',
                            model.status === 'warm' ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300' : '',
                            model.status === 'cold' ? 'border-sky-500/40 bg-sky-500/10 text-sky-700 dark:text-sky-300' : '',
                          )}
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          {model.status === 'warm' ? 'Warm' : model.status === 'cold' ? 'Cold' : model.status}
                        </Badge>
                      </div>
                      <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                        <span>{model.node_count} node{model.node_count === 1 ? '' : 's'}</span>
                        <span>{model.size_gb.toFixed(1)} GB</span>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : (
              <DashboardPanelEmpty
                icon={<Sparkles className="h-4 w-4" />}
                title={(status?.mesh_models.length ?? 0) > 0 ? `No ${modelFilter} models` : 'No model catalog data'}
                description={(status?.mesh_models.length ?? 0) > 0 ? 'Try changing the model filter.' : 'Model metadata will appear once the mesh reports available models.'}
              />
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

type PositionedTopologyNode = TopologyNode & {
  x: number;
  y: number;
  bucket: 'center' | 'serving' | 'worker' | 'client';
};

type TopologyNodeInfo = {
  role: string;
  servingLabel: string;
  fullServing: string;
  vramSharePct: number;
  modelUsagePct: number;
  modelGb: number;
};

type TopologyFlowNodeData = {
  node: PositionedTopologyNode;
  info: TopologyNodeInfo;
  selected: boolean;
};

function TopologyFlowNode({ data }: NodeProps<TopologyFlowNodeData>) {
  const isCenter = data.node.bucket === 'center';
  const dotClass = isCenter ? 'bg-primary border-primary' : 'bg-muted border-border';
  const vramWidth = Math.max(0, Math.min(100, data.info.vramSharePct));
  const modelWidth = Math.max(0, Math.min(100, data.info.modelUsagePct));

  return (
    <div className="w-[208px]">
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 1, height: 1, border: 0, pointerEvents: 'none' }} />
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 1, height: 1, border: 0, pointerEvents: 'none' }} />

      <div className={cn('mx-auto h-7 w-7 rounded-full border-2', dotClass)} />
      <div className="mt-1 break-all text-center text-[10px] leading-3 text-foreground">
        {data.node.self ? `${data.node.id} (you)` : data.node.id}
      </div>

      <div
        className={cn(
          'mt-1 rounded-md border bg-card px-2 py-1.5',
          data.selected ? 'border-ring ring-1 ring-ring/50' : 'border-border/90',
        )}
      >
        <div className="truncate text-[10px] leading-3 text-foreground/90">
          {data.info.role} · {data.info.servingLabel}
        </div>

        <div className="mt-1 flex items-center gap-1">
          <span className="text-[9px] text-primary">●</span>
          <div className="h-1 flex-1 rounded bg-muted">
            <div className="h-1 rounded bg-primary" style={{ width: `${vramWidth}%` }} />
          </div>
          <span className="text-[10px]">{data.info.vramSharePct}%</span>
        </div>

        <div className="mt-0.5 flex items-center gap-1">
          <span className="text-[9px] text-primary">◆</span>
          <div className="h-1 flex-1 rounded bg-muted">
            <div className="h-1 rounded bg-primary" style={{ width: `${modelWidth}%` }} />
          </div>
          <span className="text-[10px]">{data.info.modelUsagePct}%</span>
        </div>
      </div>
    </div>
  );
}

const topologyNodeTypes = { topologyNode: TopologyFlowNode };

function MeshTopologyDiagram({
  status,
  nodes,
  selectedModel,
  themeMode,
}: {
  status: StatusPayload | null;
  nodes: TopologyNode[];
  selectedModel: string;
  themeMode: ThemeMode;
}) {
  if (!status || !nodes.length) return <EmptyPanel text="No topology data yet." />;

  const center = nodes.find((n) => n.host) || nodes.find((n) => n.self) || nodes[0];
  const others = nodes.filter((n) => n.id !== center.id).sort((a, b) => (b.vram - a.vram) || a.id.localeCompare(b.id));
  const focusModel = selectedModel || status.model_name || '';
  const serving = others.filter((n) => !n.client && !!n.serving && (!focusModel || n.serving === focusModel));
  const servingIds = new Set(serving.map((n) => n.id));
  const clients = others.filter((n) => n.client);
  const workers = others.filter((n) => !n.client && !servingIds.has(n.id));

  const total = nodes.length;
  const nodeRadius = total >= 500 ? 3.6 : total >= 280 ? 4.8 : total >= 160 ? 6.2 : total >= 90 ? 7.4 : total >= 50 ? 8.8 : 10.4;
  const positioned = layoutTopologyNodes(center, serving, workers, clients, nodeRadius);
  const maxCoord = positioned.reduce((m, p) => Math.max(m, Math.hypot(p.x, p.y)), 0);
  const frame = Math.max(220, maxCoord + 230);
  const clientEdgeStride = total > 320 ? 6 : total > 220 ? 4 : total > 120 ? 2 : 1;
  const gpuNodeCount = nodes.filter((n) => !n.client).length;
  const meshVramGb = nodes.filter((n) => !n.client).reduce((sum, n) => sum + Math.max(0, n.vram), 0);
  const servingCount = nodes.filter((n) => !n.client && n.serving && n.serving !== '(idle)').length;

  const [selectedNodeId, setSelectedNodeId] = useState(center.id);

  useEffect(() => {
    setSelectedNodeId((prev) => (nodes.some((n) => n.id === prev) ? prev : center.id));
  }, [nodes, center.id]);

  const modelSizeByName = useMemo(() => new Map((status.mesh_models ?? []).map((m) => [m.name, m.size_gb])), [status.mesh_models]);
  const nodeInfoById = useMemo(() => {
    const out = new Map<string, TopologyNodeInfo>();
    for (const node of nodes) {
      const servingModel = !node.client && node.serving && node.serving !== '(idle)' ? node.serving : '';
      const role = node.client ? 'Client' : node.host ? 'Host' : servingModel ? 'Worker' : 'Idle';
      const modelGb = servingModel
        ? (modelSizeByName.get(servingModel) ?? (node.self ? status.model_size_gb || 0 : 0))
        : 0;
      const vramSharePct = !node.client && meshVramGb > 0 ? Math.round((Math.max(0, node.vram) / meshVramGb) * 100) : 0;
      const modelUsagePct = !node.client && node.vram > 0 && modelGb > 0
        ? Math.min(100, Math.round((modelGb / node.vram) * 100))
        : 0;
      out.set(node.id, {
        role,
        servingLabel: node.client ? 'CLIENT' : servingModel ? shortName(servingModel) : 'IDLE',
        fullServing: servingModel,
        vramSharePct,
        modelUsagePct,
        modelGb,
      });
    }
    return out;
  }, [nodes, modelSizeByName, status.model_size_gb, meshVramGb]);
  const flowColorMode = themeMode === 'auto'
    ? (typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light')
    : themeMode;

  const flowNodes = useMemo<Node<TopologyFlowNodeData>[]>(() => {
    return positioned.map((p) => ({
      id: p.id,
      type: 'topologyNode',
      position: { x: p.x + frame, y: p.y + frame },
      origin: [0.5, 0],
      data: {
        node: p,
        info: nodeInfoById.get(p.id) ?? {
          role: 'Node',
          servingLabel: 'IDLE',
          fullServing: '',
          vramSharePct: 0,
          modelUsagePct: 0,
          modelGb: 0,
        },
        selected: p.id === selectedNodeId,
      },
      draggable: false,
      selectable: false,
      connectable: false,
      zIndex: p.id === center.id ? 10 : 1,
    }));
  }, [positioned, frame, nodeInfoById, selectedNodeId, center.id]);

  const flowEdges = useMemo<Edge[]>(() => {
    const outer = positioned.filter((p) => p.id !== center.id);
    return outer
      .filter((p, idx) => !(p.bucket === 'client' && idx % clientEdgeStride !== 0))
      .map((p) => {
        const stroke =
          p.bucket === 'serving'
            ? 'rgba(34,197,94,0.35)'
            : p.bucket === 'worker'
              ? 'rgba(56,189,248,0.3)'
              : 'rgba(148,163,184,0.22)';
        return {
          id: `edge-${center.id}-${p.id}`,
          source: center.id,
          target: p.id,
          type: 'straight',
          animated: status.llama_ready,
          style: {
            stroke,
            strokeWidth: 1.2,
            strokeDasharray: p.bucket === 'client' ? '4 5' : undefined,
          },
        };
      });
  }, [positioned, center.id, clientEdgeStride, status.llama_ready]);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="h-[360px] overflow-hidden rounded-lg border md:h-[420px] lg:h-[460px] xl:h-[520px]">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={topologyNodeTypes}
          colorMode={flowColorMode}
          fitView
          fitViewOptions={{ padding: 0.22, maxZoom: 1.05 }}
          minZoom={0.2}
          maxZoom={1.6}
          zoomOnScroll={false}
          zoomOnPinch={false}
          panOnScroll={false}
          panOnDrag
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          onNodeClick={(_, node) => setSelectedNodeId(node.id)}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="hsl(var(--border))" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
    </div>
  );
}

function layoutTopologyNodes(
  center: TopologyNode,
  serving: TopologyNode[],
  workers: TopologyNode[],
  clients: TopologyNode[],
  nodeRadius: number,
): PositionedTopologyNode[] {
  const ringSpacing = nodeRadius * 8.4 + 62;
  const minChord = nodeRadius * 6.8 + 118;
  const positioned: PositionedTopologyNode[] = [{ ...center, x: 0, y: 0, bucket: 'center' }];
  const all = [
    ...serving.map((n) => ({ ...n, bucket: 'serving' as const })),
    ...workers.map((n) => ({ ...n, bucket: 'worker' as const })),
    ...clients.map((n) => ({ ...n, bucket: 'client' as const })),
  ];

  if (all.length > 0 && all.length <= 12) {
    const radius = 150 + ringSpacing + (all.length > 6 ? 20 : 0);
    for (let i = 0; i < all.length; i += 1) {
      const angle = -Math.PI / 2 + ((2 * Math.PI * i) / all.length);
      const node = all[i];
      positioned.push({
        ...node,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
      });
    }
    return positioned;
  }

  let ring = 1;
  const groups: Array<{ key: 'serving' | 'worker' | 'client'; nodes: TopologyNode[]; phase: number }> = [
    { key: 'serving', nodes: [...serving], phase: 0 },
    { key: 'worker', nodes: [...workers], phase: Math.PI / 9 },
    { key: 'client', nodes: [...clients], phase: Math.PI / 4 },
  ];

  for (const group of groups) {
    let phase = group.phase;
    let offset = 0;
    while (offset < group.nodes.length) {
      const radius = 110 + ring * ringSpacing;
      const capacity = Math.max(8, Math.floor((2 * Math.PI * radius) / minChord));
      const take = Math.min(capacity, group.nodes.length - offset);
      for (let i = 0; i < take; i += 1) {
        const angle = -Math.PI / 2 + phase + ((2 * Math.PI * i) / take);
        const node = group.nodes[offset + i];
        positioned.push({
          ...node,
          x: Math.cos(angle) * radius,
          y: Math.sin(angle) * radius,
          bucket: group.key,
        });
      }
      offset += take;
      phase += 0.21;
      ring += 1;
    }
  }

  return positioned;
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <div
      className={cn(
        'break-words text-sm leading-6',
        '[&_a]:underline [&_a]:underline-offset-2',
        '[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:italic',
        '[&_code]:rounded [&_code]:bg-background/70 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.9em]',
        '[&_h1]:mb-2 [&_h1]:mt-3 [&_h1]:text-base [&_h1]:font-semibold [&_h1:first-child]:mt-0',
        '[&_h2]:mb-2 [&_h2]:mt-3 [&_h2]:text-sm [&_h2]:font-semibold [&_h2:first-child]:mt-0',
        '[&_hr]:my-3 [&_hr]:border-border',
        '[&_li]:my-0.5',
        '[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5',
        '[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0',
        '[&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:whitespace-pre [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-border/70 [&_pre]:bg-background/80 [&_pre]:p-3',
        '[&_pre_code]:bg-transparent [&_pre_code]:p-0',
        '[&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs',
        '[&_td]:border [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1',
        '[&_th]:border [&_th]:border-border/60 [&_th]:bg-muted/40 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left',
        '[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5',
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

function ChatBubble({
  message,
  reasoningOpen,
  onReasoningToggle,
}: {
  message: ChatMessage;
  reasoningOpen: boolean;
  onReasoningToggle: (open: boolean) => void;
}) {
  const isUser = message.role === 'user';
  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className="w-full max-w-[92%] md:max-w-[82%]">
        <div className="mb-1 flex items-center gap-2 px-1 text-xs text-muted-foreground">
          {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
          <span>{isUser ? 'You' : 'Assistant'}</span>
          {message.model ? <span>· {shortName(message.model)}</span> : null}
        </div>
        <div
          className={cn(
            'rounded-lg border px-4 py-3 text-sm leading-6 break-words',
            isUser
              ? 'bg-muted whitespace-pre-wrap'
              : message.error
                ? 'border-destructive/50 text-destructive'
                : 'bg-background',
          )}
        >
          {message.content ? <MarkdownMessage content={message.content} /> : !isUser ? '...' : ''}
        </div>
        {message.reasoning ? (
          <Card className="mt-2">
            <CardContent className="p-3">
              <Accordion type="single" collapsible value={reasoningOpen ? 'reasoning' : ''} onValueChange={(v) => onReasoningToggle(v === 'reasoning')}>
                <AccordionItem value="reasoning" className="border-b-0">
                  <AccordionTrigger className="py-0 text-xs">Reasoning</AccordionTrigger>
                  <AccordionContent>
                    <div className="mt-2 whitespace-pre-wrap text-xs leading-5 text-muted-foreground">{message.reasoning}</div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </CardContent>
          </Card>
        ) : null}
        {message.stats ? <div className="mt-1 px-1 text-xs text-muted-foreground">{message.stats}</div> : null}
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  tooltip,
}: {
  title: string;
  value: string;
  icon: ReactNode;
  tooltip?: string;
}) {
  const card = (
    <Card>
      <CardContent className="p-3">
        <div className="mb-2 flex items-center gap-2 text-muted-foreground">{icon}<span className="text-xs">{title}</span></div>
        <div className="text-sm font-semibold text-foreground">{value}</div>
      </CardContent>
    </Card>
  );
  if (!tooltip) return card;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div>{card}</div>
      </TooltipTrigger>
      <TooltipContent side="bottom" align="center" sideOffset={8}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function EmptyPanel({ text }: { text: string }) {
  return (
    <Card>
      <CardContent className="p-4 text-sm text-muted-foreground">{text}</CardContent>
    </Card>
  );
}

function DashboardPanelEmpty({
  icon,
  title,
  description,
}: {
  icon: ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex h-[18rem] flex-col items-center justify-center rounded-md border border-dashed bg-muted/20 px-4 text-center md:h-[20rem]">
      <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-full border bg-background text-muted-foreground">
        {icon}
      </div>
      <div className="text-sm font-medium">{title}</div>
      <div className="mt-1 max-w-md text-xs text-muted-foreground">{description}</div>
    </div>
  );
}

function meshGpuVram(status: StatusPayload | null) {
  if (!status) return 0;
  return (status.is_client ? 0 : status.my_vram_gb || 0) + (status.peers || []).filter((p) => p.role !== 'Client').reduce((s, p) => s + p.vram_gb, 0);
}

function shortName(name: string) {
  return (name || '').replace(/-Q\w+$/, '').replace(/-Instruct/, '');
}
