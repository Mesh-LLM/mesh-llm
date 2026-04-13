type ChatAttachmentKind = "image" | "audio" | "file";
type ChatAttachmentStatus = "pending" | "uploading" | "failed";

type ChatAttachment = {
  id: string;
  kind: ChatAttachmentKind;
  dataUrl: string;
  mimeType: string;
  fileName?: string;
  status?: ChatAttachmentStatus;
  error?: string;
  extractedText?: string;
  extractionSummary?: string;
  renderedPageImages?: string[];
  imageDescription?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  model?: string;
  stats?: string;
  error?: boolean;
  image?: string;
  audio?: {
    dataUrl: string;
    mimeType: string;
    fileName?: string;
  };
  attachments?: ChatAttachment[];
};

type ChatConversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
};

type ChatState = {
  conversations: ChatConversation[];
  activeConversationId: string;
};

const DEFAULT_CHAT_TITLE = "New chat";

function randomId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function createConversation(
  seed?: Partial<Pick<ChatConversation, "title" | "messages">>,
): ChatConversation {
  const now = Date.now();
  return {
    id: randomId(),
    title: seed?.title || DEFAULT_CHAT_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: seed?.messages || [],
  };
}

export function createInitialChatState(): ChatState {
  return { conversations: [], activeConversationId: "" };
}

export async function loadPersistedChatState(
  readPersistedState: () => Promise<ChatState | null>,
): Promise<ChatState> {
  const fromDb = await readPersistedState();
  return fromDb ?? createInitialChatState();
}

export function findLastUserMessageIndex(messages: ChatMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "user") return i;
  }
  return -1;
}
