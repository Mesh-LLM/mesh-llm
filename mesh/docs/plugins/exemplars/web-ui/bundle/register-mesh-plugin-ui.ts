import type {
  MeshPluginUiBundleModule,
  MeshPluginUiConfigMountContext,
  MeshPluginUiHost,
  MeshPluginUiMountContext,
} from "./host-contract";

const moduleRegistration = {
  async registerMeshPluginUi(host) {
    host.state.update({ loadedAt: "exemplar" });

    return {
      pages: {
        overview: mountOverviewPage,
      },
      configSections: {
        "page-actions": mountPageActionsSection,
      },
    };
  },
} satisfies MeshPluginUiBundleModule;

export const registerMeshPluginUi = moduleRegistration.registerMeshPluginUi;

function retentionDays(host: MeshPluginUiHost): number {
  const configured = Number(host.config.visible.settings.retention_days ?? 14);
  return Number.isFinite(configured)
    ? Math.max(1, Math.min(365, Math.round(configured)))
    : 14;
}

function styleButton(
  button: HTMLButtonElement,
  host: MeshPluginUiHost,
  primary = false,
) {
  const { tokens } = host.appearance;
  Object.assign(button.style, {
    alignItems: "center",
    background: primary ? tokens.accent : tokens.panelStrong,
    border: `1px solid ${primary ? tokens.accent : tokens.border}`,
    borderRadius: tokens.radius,
    color: primary ? tokens.accentInk : tokens.foreground,
    cursor: "pointer",
    display: "inline-flex",
    font: "inherit",
    fontWeight: "600",
    minHeight: "36px",
    padding: "0 12px",
  });
}

function mountOverviewPage({ element, host, page }: MeshPluginUiMountContext) {
  const days = retentionDays(host);
  let noteCount = 0;
  const { tokens } = host.appearance;

  Object.assign(element.style, {
    display: "grid",
    gap: "18px",
    maxWidth: "760px",
    padding: "4px",
  });

  const heading = document.createElement("h2");
  heading.textContent = page.label;
  heading.style.color = tokens.foreground;
  heading.style.fontSize = "1.25rem";
  heading.style.fontWeight = "650";
  heading.style.margin = "0";

  const status = document.createElement("p");
  status.textContent = `${host.plugin.name} is ready. Its exemplar.notes.v1 capability remains available when this page is hidden.`;
  status.style.color = tokens.foreground;
  status.style.margin = "0";
  status.style.maxWidth = "68ch";

  const retentionPanel = document.createElement("section");
  retentionPanel.setAttribute("aria-label", "Retention window");
  Object.assign(retentionPanel.style, {
    background: tokens.panelStrong,
    border: `1px solid ${tokens.border}`,
    borderRadius: tokens.radiusLarge,
    display: "grid",
    gap: "10px",
    padding: "16px",
  });

  const retentionHeading = document.createElement("h3");
  retentionHeading.textContent = "Retention window";
  retentionHeading.style.color = tokens.foreground;
  retentionHeading.style.fontSize = "0.875rem";
  retentionHeading.style.fontWeight = "600";
  retentionHeading.style.margin = "0";

  const retentionValue = document.createElement("p");
  retentionValue.textContent = `${days} days`;
  retentionValue.style.color = tokens.accent;
  retentionValue.style.fontSize = "1.5rem";
  retentionValue.style.fontWeight = "700";
  retentionValue.style.margin = "0";
  retentionValue.dataset.retentionDays = String(days);

  const meter = document.createElement("div");
  meter.setAttribute("aria-label", "Configured retention days");
  meter.setAttribute("aria-valuemax", "365");
  meter.setAttribute("aria-valuemin", "1");
  meter.setAttribute("aria-valuenow", String(days));
  meter.setAttribute("role", "meter");
  Object.assign(meter.style, {
    background: tokens.borderSoft,
    borderRadius: tokens.radius,
    height: "8px",
    overflow: "hidden",
  });
  const meterFill = document.createElement("div");
  meterFill.style.background = tokens.accent;
  meterFill.style.height = "100%";
  meterFill.style.width = `${Math.max(2, (days / 365) * 100)}%`;
  meter.append(meterFill);
  retentionPanel.append(retentionHeading, retentionValue, meter);

  const actions = document.createElement("div");
  actions.style.display = "flex";
  actions.style.flexWrap = "wrap";
  actions.style.gap = "10px";
  const addNote = document.createElement("button");
  addNote.type = "button";
  addNote.textContent = "Add sample note";
  styleButton(addNote, host, true);
  const openSettings = document.createElement("button");
  openSettings.type = "button";
  openSettings.textContent = "Open plugin settings";
  styleButton(openSettings, host);
  actions.append(addNote, openSettings);

  const interactionStatus = document.createElement("p");
  interactionStatus.setAttribute("aria-live", "polite");
  interactionStatus.setAttribute("role", "status");
  interactionStatus.textContent = "No sample notes yet.";
  interactionStatus.style.color = tokens.foreground;
  interactionStatus.style.margin = "0";

  const addSampleNote = () => {
    noteCount += 1;
    interactionStatus.textContent = `Sample note ${noteCount} will be retained for ${days} days.`;
  };
  const navigateToSettings = () =>
    host.navigation.navigateTo("/configuration/plugins");
  addNote.addEventListener("click", addSampleNote);
  openSettings.addEventListener("click", navigateToSettings);

  element.replaceChildren(
    heading,
    status,
    retentionPanel,
    actions,
    interactionStatus,
  );

  return {
    unmount() {
      addNote.removeEventListener("click", addSampleNote);
      openSettings.removeEventListener("click", navigateToSettings);
      element.removeAttribute("style");
      element.replaceChildren();
    },
  };
}

function mountPageActionsSection({
  element,
  host,
}: MeshPluginUiConfigMountContext) {
  const days = retentionDays(host);
  const { tokens } = host.appearance;
  Object.assign(element.style, {
    alignItems: "center",
    display: "flex",
    flexWrap: "wrap",
    gap: "12px",
    justifyContent: "space-between",
  });

  const copy = document.createElement("div");
  const current = document.createElement("p");
  current.textContent = `Current retention: ${days} days`;
  current.style.color = tokens.foreground;
  current.style.fontWeight = "600";
  current.style.margin = "0";
  const guidance = document.createElement("p");
  guidance.textContent =
    "Edit Retention days below with the host-generated schema control.";
  guidance.style.color = tokens.foreground;
  guidance.style.margin = "4px 0 0";
  copy.append(current, guidance);

  const openPage = document.createElement("button");
  openPage.type = "button";
  openPage.textContent = "Open exemplar page";
  styleButton(openPage, host, true);
  const navigateToPage = () => host.navigation.openPluginPage("overview");
  openPage.addEventListener("click", navigateToPage);
  element.replaceChildren(copy, openPage);

  return {
    unmount() {
      openPage.removeEventListener("click", navigateToPage);
      element.removeAttribute("style");
      element.replaceChildren();
    },
  };
}
