(function () {
  function getCodeText(code) {
    if (code.dataset.copyCodeText) {
      return code.dataset.copyCodeText;
    }

    return code.innerText.replace(/\n$/, "");
  }

  const CLI_LANGUAGE_PATTERN = /\blanguage-(?:bash|powershell|ps1|sh|shell|zsh)\b/;
  const TEXT_LANGUAGE_PATTERN = /\blanguage-text\b/;
  const URL_PATTERN = /^(?:https?:\/\/|localhost(?::\d+)?|127(?:\.\d{1,3}){3}(?::\d+)?)/;
  const URL_ONLY_PATTERN = /^(?:https?:\/\/[^\s'"`<>]+|localhost(?::\d+)?(?:\/[^\s'"`<>]*)?|127(?:\.\d{1,3}){3}(?::\d+)?(?:\/[^\s'"`<>]*)?)$/;
  const TOKEN_PATTERN = /(\s+|#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|https?:\/\/[^\s'"`<>]+|localhost(?::\d+)?(?:\/[^\s'"`<>]*)?|127(?:\.\d{1,3}){3}(?::\d+)?(?:\/[^\s'"`<>]*)?|<[^>\s]+>|\$[A-Za-z_][\w]*|~\/[^\s'"`<>]+|--[A-Za-z0-9][\w-]*|-[A-Za-z][\w-]*|\|\||&&|[|;&()]|[^\s|;&()]+)/g;
  const COMMANDS = new Set([
    "bash",
    "curl",
    "iex",
    "irm",
    "jq",
    "mesh-llm",
    "node",
    "npm",
    "screen",
    "set",
    "tmux",
    "true",
  ]);

  function isCliCodeBlock(code) {
    return CLI_LANGUAGE_PATTERN.test(code.className);
  }

  function isUrlOnlyTextBlock(code, text) {
    if (!TEXT_LANGUAGE_PATTERN.test(code.className)) {
      return false;
    }

    const lines = text.split("\n").map((line) => line.trim()).filter(Boolean);
    return lines.length > 0 && lines.every((line) => URL_ONLY_PATTERN.test(line));
  }

  function appendToken(fragment, token, className) {
    if (!className) {
      fragment.appendChild(document.createTextNode(token));
      return;
    }

    const span = document.createElement("span");
    span.className = `code-token ${className}`;
    span.textContent = token;
    fragment.appendChild(span);
  }

  function classifyToken(token, expectsCommand, expectsValue) {
    if (/^\s+$/.test(token)) {
      return { className: "", expectsCommand, expectsValue };
    }

    if (token.startsWith("#")) {
      return { className: "code-comment", expectsCommand: false, expectsValue: false };
    }

    if (token === "|" || token === "&&" || token === ";" || token === "(" || token === ")") {
      return { className: "code-operator", expectsCommand: token !== ")", expectsValue: false };
    }

    if (token === "||") {
      return { className: "code-operator", expectsCommand: true, expectsValue: false };
    }

    if (URL_PATTERN.test(token)) {
      return { className: "code-value", expectsCommand: false, expectsValue: false };
    }

    if (/^['"]/.test(token)) {
      return { className: "code-value", expectsCommand: false, expectsValue: false };
    }

    if (/^(?:<[^>\s]+>|\$[A-Za-z_][\w]*|~\/)/.test(token)) {
      return { className: "code-value", expectsCommand: false, expectsValue: false };
    }

    if (/^[A-Z_][A-Z0-9_]*=/.test(token) && expectsCommand) {
      return { className: "code-value", expectsCommand: true, expectsValue: false };
    }

    if (/^--?[A-Za-z]/.test(token)) {
      return { className: "code-flag", expectsCommand: false, expectsValue: true };
    }

    const commandName = token.replace(/,$/, "");
    if (expectsCommand || COMMANDS.has(commandName)) {
      return { className: "code-command", expectsCommand: false, expectsValue: false };
    }

    if (expectsValue) {
      return { className: "code-value", expectsCommand: false, expectsValue: false };
    }

    return { className: "", expectsCommand: false, expectsValue: false };
  }

  function highlightCliLine(fragment, line) {
    let expectsCommand = true;
    let expectsValue = false;
    TOKEN_PATTERN.lastIndex = 0;
    const tokens = line.match(TOKEN_PATTERN) || [line];

    tokens.forEach((token) => {
      const result = classifyToken(token, expectsCommand, expectsValue);
      appendToken(fragment, token, result.className);
      expectsCommand = result.expectsCommand;
      expectsValue = result.expectsValue;
    });
  }

  function highlightCodeBlock(code) {
    if (code.dataset.highlightReady === "true") {
      return;
    }

    const text = code.textContent.replace(/\n$/, "");
    const isCli = isCliCodeBlock(code);
    const isUrlOnly = isUrlOnlyTextBlock(code, text);

    if (!isCli && !isUrlOnly) {
      return;
    }

    code.dataset.copyCodeText = text;
    code.dataset.highlightReady = "true";
    code.dataset.codeHighlight = isCli ? "cli" : "url";

    const fragment = document.createDocumentFragment();
    text.split("\n").forEach((line, index, lines) => {
      if (isCli) {
        highlightCliLine(fragment, line);
      } else {
        appendToken(fragment, line, "code-value");
      }

      if (index < lines.length - 1) {
        fragment.appendChild(document.createTextNode("\n"));
      }
    });

    code.replaceChildren(fragment);
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  function refreshCopyIcon() {
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }

  function setCodeCopyState(button, state) {
    const isCopied = state === "copied";
    const isFailed = state === "failed";
    const label = isCopied ? "Copied code" : isFailed ? "Copy failed" : "Copy code";

    button.classList.toggle("copied", isCopied);
    button.classList.toggle("failed", isFailed);
    button.setAttribute("aria-label", label);
    button.title = label;
    button.innerHTML = '<i data-lucide="clipboard-copy" aria-hidden="true"></i><span class="sr-only">Copy code</span>';
    refreshCopyIcon();
  }

  function addCopyButton(pre, code) {
    if (pre.dataset.copyReady === "true") {
      return;
    }

    const text = getCodeText(code);
    if (!text.trim()) {
      return;
    }

    pre.dataset.copyReady = "true";

    const isSingleLine = !text.includes("\n");

    const frame = document.createElement("div");
    frame.className = isSingleLine ? "code-copy-frame code-copy-frame-single-line" : "code-copy-frame";
    if (isSingleLine) {
      pre.classList.add("items-center");
    }
    pre.parentNode.insertBefore(frame, pre);
    frame.appendChild(pre);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "code-copy-button";
    button.setAttribute("aria-label", "Copy code");
    button.title = "Copy code";
    setCodeCopyState(button, "copy");

    let resetTimer = null;
    button.addEventListener("click", async () => {
      window.clearTimeout(resetTimer);
      try {
        await copyText(getCodeText(code));
        setCodeCopyState(button, "copied");
      } catch (_error) {
        setCodeCopyState(button, "failed");
      }
      resetTimer = window.setTimeout(() => {
        setCodeCopyState(button, "copy");
      }, 1600);
    });

    frame.appendChild(button);
  }

  function addInlineCopyButtons() {
    document.querySelectorAll("[data-copy-text]").forEach((button) => {
      if (button.dataset.copyReady === "true") {
        return;
      }

      button.dataset.copyReady = "true";
      const defaultLabel = button.getAttribute("aria-label") || "Copy";
      let resetTimer = null;

      button.addEventListener("click", async () => {
        window.clearTimeout(resetTimer);
        try {
          await copyText(button.dataset.copyText || "");
          button.classList.add("copied");
          button.setAttribute("aria-label", "Copied");
        } catch (_error) {
          button.classList.add("failed");
          button.setAttribute("aria-label", "Copy failed");
        }
        resetTimer = window.setTimeout(() => {
          button.classList.remove("copied", "failed");
          button.setAttribute("aria-label", defaultLabel);
        }, 1600);
      });
    });
  }

  function addCtaInstallSwitcher() {
    document.querySelectorAll(".cta-terminal").forEach((terminal) => {
      const display = terminal.querySelector("[data-install-command-display]");
      const copyButton = terminal.querySelector(".cta-copy[data-copy-text]");
      const tabs = terminal.querySelectorAll("[data-install-method]");

      if (!display || !copyButton || tabs.length === 0) {
        return;
      }

      tabs.forEach((tab) => {
        if (tab.dataset.installReady === "true") {
          return;
        }

        tab.dataset.installReady = "true";
        tab.addEventListener("click", () => {
          const method = tab.dataset.installMethod;
          const template = terminal.querySelector(`[data-install-template="${method}"]`);

          if (!template) {
            return;
          }

          tabs.forEach((item) => {
            const isActive = item === tab;
            item.classList.toggle("active", isActive);
            item.setAttribute("aria-selected", isActive ? "true" : "false");
          });

          display.innerHTML = template.innerHTML;
          copyButton.dataset.copyText = tab.dataset.installCopy || "";
          copyButton.classList.remove("copied", "failed");
          copyButton.setAttribute("aria-label", `Copy ${method} install command`);
        });
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("pre > code").forEach((code) => {
      highlightCodeBlock(code);
      addCopyButton(code.parentElement, code);
    });
    addCtaInstallSwitcher();
    addInlineCopyButtons();
  });
})();
