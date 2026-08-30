(() => {
  "use strict";

  const app = document.querySelector("[data-cli-explorer]");
  if (!app) return;

  const errorBanner = app.querySelector("[data-load-error]");
  const visibleCount = app.querySelector("[data-visible-count]");
  const copyStatus = app.querySelector("[data-copy-status]");
  const inventoryElement = document.getElementById("cli-inventory-data");

  const showError = () => {
    if (errorBanner) errorBanner.hidden = false;
    if (visibleCount) visibleCount.textContent = "Inventory unavailable";
  };

  if (!inventoryElement || !window.d3) {
    showError();
    return;
  }

  let payload;
  try {
    payload = JSON.parse(inventoryElement.textContent || "");
  } catch {
    showError();
    return;
  }

  if (!payload || payload.schemaVersion !== 1 || !payload.root || typeof payload.root !== "object") {
    showError();
    return;
  }

  const d3 = window.d3;
  const NODE_WIDTH = 232;
  const NODE_HEIGHT = 32;
  const COLUMN_STEP = 288;
  const NODE_SHELL_RIGHT = NODE_WIDTH - 8;
  const BUS_SOURCE_OFFSET = 8;
  const BUS_TRUNK_OFFSET = 20;
  const NODE_COUNT_X = NODE_WIDTH - 34;
  const NODE_DISCLOSURE_X = NODE_WIDTH - 17;
  const MOBILE_NODE_OFFSET_X = -98;
  const ORTHOGONAL_ELBOW_RADIUS = 8;
  const LINK_GRADIENT_DISTANCE = 32;
  const LINK_GRADIENT_REACHED = 28;
  const LINK_GRADIENT_COLORS = {
    command: { color: "#94a3b8", opacity: "0.72" },
    option: { color: "#f59e0b", opacity: "0.58" },
    positional: { color: "#a78bfa", opacity: "0.58" }
  };
  const BUS_GRADIENT_BLUE = "#60a5fa";
  let uid = 0;
  const index = [];

  function asList(value) {
    if (value === undefined || value === null || value === "") return [];
    if (Array.isArray(value)) return value.map(item => String(item)).filter(Boolean);
    return [String(value)];
  }

  function normalizeNode(raw, parent = null) {
    if (!raw || typeof raw !== "object") return null;
    const kind = raw.kind === "option" || raw.kind === "positional"
      ? raw.kind
      : parent
        ? "command"
        : "root";
    const name = String(raw.name || raw.long || raw.short || (kind === "root" ? "mesh-llm" : "unnamed"));
    const node = {
      ...raw,
      id: `cli-node-${uid++}`,
      inventoryId: raw.id === undefined ? "" : String(raw.id),
      kind,
      name,
      description: raw.description === undefined ? "" : String(raw.description),
      hidden: raw.hidden === true,
      aliases: asList(raw.aliases),
      short: raw.short === undefined ? "" : String(raw.short),
      long: raw.long === undefined ? "" : String(raw.long),
      valueNames: asList(raw.valueNames),
      defaultValues: asList(raw.defaultValues),
      possibleValues: asList(raw.possibleValues),
      conflicts: asList(raw.conflicts),
      children: [],
      originalChildren: [],
      parentNode: parent,
      path: raw.path ? String(raw.path) : parent ? `${parent.path} ${name}` : name,
      expanded: kind === "root"
    };
    index.push(node);
    node.originalChildren = (Array.isArray(raw.children) ? raw.children : [])
      .map(child => normalizeNode(child, node))
      .filter(Boolean);
    node.children = node.originalChildren;
    return node;
  }

  const cliRoot = normalizeNode(payload.root);
  if (!cliRoot) {
    showError();
    return;
  }

  const state = {
    showHidden: false,
    showRootOptions: false,
    focusedRootOption: null,
    focusPath: null,
    focusBranch: null,
    orientationPreference: null,
    selected: cliRoot,
    query: "",
    activeResult: -1,
    initialized: false,
    resizeFrame: 0,
    layoutBounds: null
  };

  const svgNode = app.querySelector("[data-tree]");
  const canvasWrap = app.querySelector(".cli-explorer-canvas-wrap");
  const layerNode = app.querySelector("[data-zoom-layer]");
  const linksNode = app.querySelector("[data-links-layer]");
  const nodesNode = app.querySelector("[data-nodes-layer]");
  const defsNode = app.querySelector("[data-link-defs]");
  const svg = d3.select(svgNode);
  const layer = d3.select(layerNode);
  const linksLayer = d3.select(linksNode);
  const nodesLayer = d3.select(nodesNode);
  const defs = d3.select(defsNode);
  const zoom = d3.zoom()
    .scaleExtent([0.28, 2])
    .on("zoom", event => layer.attr("transform", event.transform));
  const activeAnimations = new WeakMap();

  function animeApi() {
    return window.anime && typeof window.anime.animate === "function" ? window.anime : null;
  }

  function stopAnimation(target) {
    const animation = activeAnimations.get(target);
    if (!animation) return;
    if (typeof animation.cancel === "function") animation.cancel();
    else if (typeof animation.pause === "function") animation.pause();
    activeAnimations.delete(target);
  }

  function animateState(target, values, properties, duration, onUpdate, onComplete) {
    stopAnimation(target);
    const anime = animeApi();
    if (!anime || !duration) {
      Object.assign(values, properties);
      onUpdate?.();
      onComplete?.();
      return null;
    }
    let animation;
    animation = anime.animate(values, {
      ...properties,
      duration,
      ease: "outCubic",
      onUpdate: () => onUpdate?.(),
      onComplete: () => {
        if (activeAnimations.get(target) !== animation) return;
        activeAnimations.delete(target);
        onComplete?.();
      }
    });
    activeAnimations.set(target, animation);
    return animation;
  }

  svg.call(zoom).on("dblclick.zoom", null).on("wheel.zoom", null);
  // The wrapper is the complete interaction boundary: the SVG, legend,
  // status chips, and its blank background all belong to the graph surface.
  // D3's wheel handler is removed above so this capture listener owns the
  // gesture and can keep the document stationary even at either scale limit.
  canvasWrap?.addEventListener("wheel", event => {
    event.preventDefault();
    event.stopPropagation();
    const bounds = svgNode.getBoundingClientRect();
    const point = [event.clientX - bounds.left, event.clientY - bounds.top];
    const delta = -event.deltaY
      * (event.deltaMode === 1 ? 0.05 : event.deltaMode ? 1 : 0.002)
      * (event.ctrlKey ? 10 : 1);
    const factor = 2 ** delta;
    if (Number.isFinite(factor) && factor > 0) svg.call(zoom.scaleBy, factor, point);
  }, { passive: false, capture: true });
  svg.on("mousedown", () => svg.classed("is-dragging", true));
  window.addEventListener("mouseup", () => svg.classed("is-dragging", false));

  function isNarrow() {
    return window.matchMedia("(max-width: 720px)").matches;
  }

  function effectiveOrientation() {
    return state.orientationPreference || (isNarrow() ? "vertical" : "horizontal");
  }

  function isVerticalOrientation() {
    return effectiveOrientation() === "vertical";
  }

  function syncOrientationControls(orientation) {
    app.dataset.orientation = orientation;
    app.querySelectorAll('[data-action="orientation"]').forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.orientation === orientation));
    });
  }

  function visibleChildren(node) {
    if (!node.expanded) return [];
    return node.originalChildren.filter(child => {
      if (state.focusPath?.has(node) && node !== state.focusBranch && !state.focusPath.has(child)) return false;
      if (child.hidden && !state.showHidden) return false;
      if (node === cliRoot && child.kind === "option" && !state.showRootOptions && child !== state.focusedRootOption) return false;
      return true;
    });
  }

  function hierarchy() {
    return d3.hierarchy(cliRoot, visibleChildren);
  }

  function optionToken(node) {
    if (node.name && /^-/.test(node.name)) return node.name;
    const short = node.short ? `-${String(node.short).replace(/^-+/, "")}` : "";
    const long = node.long ? `--${String(node.long).replace(/^-+/, "")}` : "";
    return [short, long].filter(Boolean).join(", ") || node.name;
  }

  function nodeLabel(node) {
    const base = node.kind === "option" ? optionToken(node) : node.name;
    const values = node.valueNames.length ? ` ${node.valueNames.join(" | ")}` : "";
    return `${base}${values}`;
  }

  function descendantsCount(node) {
    let count = 0;
    const visit = current => {
      current.originalChildren.forEach(child => {
        count += 1;
        visit(child);
      });
    };
    visit(node);
    return count;
  }

  function truncatedLabel(node) {
    const label = nodeLabel(node);
    return label.length > 32 ? `${label.slice(0, 30)}…` : label;
  }

  function disclosurePath(node, verticalOrientation) {
    if (!node.originalChildren.length) return null;
    const x = NODE_DISCLOSURE_X + (verticalOrientation ? MOBILE_NODE_OFFSET_X : 0);
    const y = 0;
    return node.expanded
      ? `M${x - 4},${y - 2}L${x},${y + 2}L${x + 4},${y - 2}`
      : `M${x - 3},${y - 4}L${x + 2},${y}L${x - 3},${y + 4}`;
  }

  function layoutBanded(root) {
    const rowHeight = 44;
    const childGap = 10;
    const rootGap = 12;

    function measure(node) {
      if (!node.children?.length) {
        node.bandHeight = rowHeight;
        return rowHeight;
      }
      const gap = node.depth === 0 ? rootGap : childGap;
      const childrenHeight = node.children.reduce((sum, child) => sum + measure(child), 0)
        + gap * Math.max(0, node.children.length - 1);
      node.bandHeight = Math.max(rowHeight, childrenHeight);
      return node.bandHeight;
    }

    function place(node, top) {
      node.x = top + node.bandHeight / 2;
      node.y = node.depth * COLUMN_STEP;
      if (!node.children?.length) return;
      const gap = node.depth === 0 ? rootGap : childGap;
      const childrenHeight = node.children.reduce((sum, child) => sum + child.bandHeight, 0)
        + gap * Math.max(0, node.children.length - 1);
      let cursor = top + (node.bandHeight - childrenHeight) / 2;
      node.children.forEach(child => {
        place(child, cursor);
        cursor += child.bandHeight + gap;
      });
    }

    measure(root);
    place(root, 0);
  }

  function layoutVertical(root) {
    // Keep a deliberate horizontal rail between siblings while depth moves
    // down the canvas. The fit action can pan this wide rail instead of
    // shrinking node labels below the established readability floor.
    d3.tree().nodeSize([NODE_WIDTH + 28, 108])(root);
    root.each(node => { node.y = node.depth * 108; });
  }

  function hasExpandedNonRoot() {
    return index.some(node => node.kind !== "root" && node.originalChildren.length > 0 && node.expanded);
  }

  function elbowRadius(span, travel) {
    return Math.min(
      ORTHOGONAL_ELBOW_RADIUS,
      Math.max(0, span) / 2,
      Math.max(0, travel) / 2
    );
  }

  function linkGradientId(link) {
    return `cli-link-gradient-${link.target.data.id}`;
  }

  function linkGradientSpec(link, verticalOrientation, busX) {
    const sourceOpacity = verticalOrientation || link.source.depth > 0 ? "0.94" : "0.88";
    const target = LINK_GRADIENT_COLORS[link.target.data.kind] || LINK_GRADIENT_COLORS.command;
    if (verticalOrientation) {
      const trunkY = link.source.y + NODE_HEIGHT / 2 + BUS_TRUNK_OFFSET;
      return {
        id: linkGradientId(link),
        x1: link.target.x,
        y1: trunkY,
        x2: link.target.x,
        y2: trunkY + LINK_GRADIENT_DISTANCE,
        sourceOpacity,
        target
      };
    }
    const sourceBusX = link.source.depth === 0
      ? busX
      : link.source.y + NODE_SHELL_RIGHT + BUS_TRUNK_OFFSET;
    return {
      id: linkGradientId(link),
      x1: sourceBusX,
      y1: link.target.x,
      x2: sourceBusX + LINK_GRADIENT_DISTANCE,
      y2: link.target.x,
      sourceOpacity,
      target
    };
  }

  function updateLinkGradients(linksData, verticalOrientation, busX) {
    const gradients = defs.selectAll("linearGradient.cli-explorer-link-gradient")
      .data(linksData, link => linkGradientId(link));
    gradients.exit().remove();
    const entered = gradients.enter()
      .append("linearGradient")
      .attr("class", "cli-explorer-link-gradient")
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("spreadMethod", "pad");
    entered.append("stop").attr("class", "cli-explorer-gradient-source");
    entered.append("stop").attr("class", "cli-explorer-gradient-target");
    entered.append("stop").attr("class", "cli-explorer-gradient-target-end");
    entered.merge(gradients).each(function(link) {
      const spec = linkGradientSpec(link, verticalOrientation, busX);
      const gradient = d3.select(this);
      gradient
        .attr("id", spec.id)
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", spec.x1)
        .attr("y1", spec.y1)
        .attr("x2", spec.x2)
        .attr("y2", spec.y2)
        .attr("data-target-kind", link.target.data.kind)
        .attr("data-source-color", BUS_GRADIENT_BLUE);
      gradient.select(".cli-explorer-gradient-source")
        .attr("offset", "0%")
        .attr("stop-color", BUS_GRADIENT_BLUE)
        .attr("stop-opacity", spec.sourceOpacity);
      gradient.select(".cli-explorer-gradient-target")
        .attr("offset", `${(LINK_GRADIENT_REACHED / LINK_GRADIENT_DISTANCE) * 100}%`)
        .attr("stop-color", spec.target.color)
        .attr("stop-opacity", spec.target.opacity);
      gradient.select(".cli-explorer-gradient-target-end")
        .attr("offset", "100%")
        .attr("stop-color", spec.target.color)
        .attr("stop-opacity", spec.target.opacity);
    });
  }

  function linkPath(link, verticalOrientation, busX) {
    const siblings = link.source.children || [];
    const firstSibling = siblings[0];
    const lastSibling = siblings[siblings.length - 1];
    const isFirst = link.target === firstSibling;
    const isLast = link.target === lastSibling;

    if (verticalOrientation) {
      const trunkY = link.source.y + NODE_HEIGHT / 2 + BUS_TRUNK_OFFSET;
      const childTop = link.target.y - NODE_HEIGHT / 2;
      if (!isFirst && !isLast) return `M${link.target.x},${trunkY}V${childTop}`;
      const span = (lastSibling?.x ?? link.target.x) - (firstSibling?.x ?? link.target.x);
      const radius = elbowRadius(span, childTop - trunkY);
      if (!radius) return `M${link.target.x},${trunkY}V${childTop}`;
      const startX = isFirst ? link.target.x + radius : link.target.x - radius;
      return `M${startX},${trunkY}Q${link.target.x},${trunkY} ${link.target.x},${trunkY + radius}V${childTop}`;
    }

    const tx = link.target.y - 10;
    const ty = link.target.x;
    const sourceBusX = link.source.depth === 0
      ? busX
      : link.source.y + NODE_SHELL_RIGHT + BUS_TRUNK_OFFSET;
    if (!isFirst && !isLast) return `M${sourceBusX},${ty}H${tx}`;
    const span = (lastSibling?.x ?? link.target.x) - (firstSibling?.x ?? link.target.x);
    const radius = elbowRadius(span, tx - sourceBusX);
    if (!radius) return `M${sourceBusX},${ty}H${tx}`;
    const startY = isFirst ? ty + radius : ty - radius;
    const endX = sourceBusX + (tx >= sourceBusX ? radius : -radius);
    return `M${sourceBusX},${startY}Q${sourceBusX},${ty} ${endX},${ty}H${tx}`;
  }

  function measureLayoutBounds(root, verticalOrientation) {
    const nodes = root.descendants();
    if (!nodes.length) return null;
    const extents = nodes.map(node => {
      const x = verticalOrientation ? node.x - 110 : node.y - 12;
      const right = verticalOrientation ? node.x + 132 : node.y + NODE_WIDTH + 2;
      const y = verticalOrientation ? node.y - 22 : node.x - 22;
      const bottom = verticalOrientation ? node.y + 22 : node.x + 22;
      return { x, right, y, bottom };
    });
    const x = d3.min(extents, item => item.x);
    const right = d3.max(extents, item => item.right);
    const y = d3.min(extents, item => item.y);
    const bottom = d3.max(extents, item => item.bottom);
    return { x, y, width: right - x, height: bottom - y };
  }

  function nodeClass(node) {
    return [
      "cli-explorer-node",
      node.kind,
      node.hidden ? "hidden" : "",
      node === state.selected ? "selected" : "",
      state.query && matches(node, state.query) ? "search-hit" : ""
    ].filter(Boolean).join(" ");
  }

  function visibleDataNodes() {
    const result = [];
    const visit = node => {
      result.push(node);
      visibleChildren(node).forEach(visit);
    };
    visit(cliRoot);
    return result;
  }

  function focusTreeNode(node) {
    if (!node) return false;
    const target = [...nodesNode.querySelectorAll('[role="treeitem"]')]
      .find(item => item.getAttribute("data-node-id") === node.id);
    if (!target) return false;
    nodesNode.querySelectorAll('[role="treeitem"]').forEach(item => item.setAttribute("tabindex", "-1"));
    target.setAttribute("tabindex", "0");
    target.focus({ preventScroll: true });
    return true;
  }

  function handleNodeKeydown(event, hierarchyNode) {
    const node = hierarchyNode.data;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectNode(node, true);
      requestAnimationFrame(() => focusTreeNode(node));
      return;
    }

    const visible = visibleDataNodes();
    const currentIndex = visible.indexOf(node);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const offset = event.key === "ArrowDown" ? 1 : -1;
      focusTreeNode(visible[Math.max(0, Math.min(visible.length - 1, currentIndex + offset))]);
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      focusTreeNode(event.key === "Home" ? visible[0] : visible[visible.length - 1]);
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      if (!node.originalChildren.length) return;
      if (!node.expanded) {
        selectNode(node, true);
        requestAnimationFrame(() => focusTreeNode(node));
      } else {
        focusTreeNode(visibleChildren(node)[0]);
      }
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      if (node.originalChildren.length && node.expanded) {
        selectNode(node, true);
        requestAnimationFrame(() => focusTreeNode(node));
      } else {
        focusTreeNode(node.parentNode);
      }
    }
  }

  function update(source = cliRoot, immediate = false) {
    const narrow = isNarrow();
    const verticalOrientation = isVerticalOrientation();
    const mobileBranch = narrow && Boolean(state.focusBranch);
    const mobilePath = narrow && Boolean(state.focusPath) && !mobileBranch;
    const mobileOverview = narrow && !mobileBranch && !mobilePath;
    const overview = !state.focusPath && !state.focusBranch && !hasExpandedNonRoot();
    syncOrientationControls(verticalOrientation ? "vertical" : "horizontal");

    // The unfocused overview deliberately exposes only root-level siblings.
    // Clear stale expanded descendants before building the hierarchy so that
    // returning from a focused branch never leaves unpositioned nodes behind.
    if (mobileOverview) {
      index.forEach(node => {
        if (node.kind !== "root") node.expanded = false;
      });
    }

    const treeRoot = hierarchy();
    const visibleNodes = treeRoot.descendants();
    if (!visibleNodes.some(node => node.data === state.selected)) {
      state.selected = visibleNodes[0]?.data || cliRoot;
      renderInspector(state.selected);
    }
    if (verticalOrientation) {
      layoutVertical(treeRoot);
    } else {
      layoutBanded(treeRoot);
      if (overview && treeRoot.children?.length) {
        // Keep the root and the first visible command in the first viewport;
        // the remaining sibling band continues downward for vertical panning
        // at every width when horizontal orientation is explicitly selected.
        treeRoot.x = treeRoot.children[0].x;
      }
    }
    state.layoutBounds = measureLayoutBounds(treeRoot, verticalOrientation);

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const duration = immediate || reduced ? 0 : 190;
    const nodeTransform = node => verticalOrientation
      ? `translate(${node.x},${node.y})`
      : `translate(${node.y},${node.x})`;
    const rootChildren = treeRoot.children || [];
    const busX = treeRoot.y + NODE_SHELL_RIGHT + BUS_TRUNK_OFFSET;
    const rootBusData = !verticalOrientation && rootChildren.length ? [{
      x1: treeRoot.y + NODE_SHELL_RIGHT + BUS_SOURCE_OFFSET,
      x2: busX,
      centerY: treeRoot.x,
      minY: d3.min(rootChildren, child => child.x),
      maxY: d3.max(rootChildren, child => child.x),
      radius: elbowRadius(
        d3.max(rootChildren, child => child.x) - d3.min(rootChildren, child => child.x),
        d3.max(rootChildren, child => child.x) - d3.min(rootChildren, child => child.x)
      )
    }] : [];
    const rootBus = linksLayer.selectAll("path.cli-explorer-root-bus").data(rootBusData);
    rootBus.exit().remove();
    rootBus.enter().append("path").attr("class", "cli-explorer-root-bus").merge(rootBus)
      .attr("d", item => `M${item.x1},${item.centerY}H${item.x2}M${item.x2},${item.minY + item.radius}V${item.maxY - item.radius}`);

    const verticalBusData = verticalOrientation
      ? treeRoot.descendants()
        .filter(node => node.children?.length)
        .map(node => ({
          id: node.data.id,
          sourceX: node.x,
          sourceY: node.y + NODE_HEIGHT / 2,
          trunkY: node.y + NODE_HEIGHT / 2 + BUS_TRUNK_OFFSET,
          minX: d3.min(node.children, child => child.x),
          maxX: d3.max(node.children, child => child.x),
          radius: elbowRadius(
            d3.max(node.children, child => child.x) - d3.min(node.children, child => child.x),
            d3.max(node.children, child => child.x) - d3.min(node.children, child => child.x)
          )
        }))
      : [];
    const verticalBuses = linksLayer.selectAll("path.cli-explorer-vertical-bus").data(verticalBusData, item => item.id);
    verticalBuses.exit().remove();
    verticalBuses.enter().append("path").attr("class", "cli-explorer-vertical-bus").merge(verticalBuses)
      .attr("d", item => `M${item.sourceX},${item.sourceY}V${item.trunkY}M${item.minX + item.radius},${item.trunkY}H${item.maxX - item.radius}`);

    const branchBusData = !verticalOrientation
      ? treeRoot.descendants()
        .filter(node => node.depth > 0 && node.children?.length)
        .map(node => ({
          id: node.data.id,
          sourceX: node.y + NODE_SHELL_RIGHT + BUS_SOURCE_OFFSET,
          busX: node.y + NODE_SHELL_RIGHT + BUS_TRUNK_OFFSET,
          sourceY: node.x,
          minY: d3.min(node.children, child => child.x),
          maxY: d3.max(node.children, child => child.x),
          radius: elbowRadius(
            d3.max(node.children, child => child.x) - d3.min(node.children, child => child.x),
            d3.max(node.children, child => child.x) - d3.min(node.children, child => child.x)
          )
        }))
      : [];
    const branchBuses = linksLayer.selectAll("path.cli-explorer-branch-bus").data(branchBusData, item => item.id);
    branchBuses.exit().remove();
    branchBuses.enter().append("path").attr("class", "cli-explorer-branch-bus").merge(branchBuses)
      .attr("d", item => `M${item.sourceX},${item.sourceY}H${item.busX}M${item.busX},${item.minY + item.radius}V${item.maxY - item.radius}`);

    const treeLinks = treeRoot.links();
    updateLinkGradients(treeLinks, verticalOrientation, busX);
    const links = linksLayer.selectAll("path.cli-explorer-link").data(treeLinks, link => link.target.data.id);
    const linkExit = links.exit();
    const linkEnter = links.enter().append("path")
      .attr("class", link => `cli-explorer-link to-${link.target.data.kind}`)
      .attr("data-gradient-id", link => linkGradientId(link))
      .attr("stroke", link => `url(#${linkGradientId(link)})`)
      .style("stroke", link => `url(#${linkGradientId(link)})`)
      .attr("opacity", duration ? 0 : 1)
      .attr("d", link => linkPath(link, verticalOrientation, busX));
    const linkMerged = linkEnter.merge(links)
      .attr("class", link => `cli-explorer-link to-${link.target.data.kind}`)
      .attr("data-gradient-id", link => linkGradientId(link))
      .attr("stroke", link => `url(#${linkGradientId(link)})`)
      .style("stroke", link => `url(#${linkGradientId(link)})`);
    const animateLink = (element, link, removeAfter) => {
      const fromPath = element.getAttribute("d") || linkPath(link, verticalOrientation, busX);
      const targetPath = linkPath(link, verticalOrientation, busX);
      const pathAt = d3.interpolateString(fromPath, targetPath);
      const values = {
        progress: 0,
        opacity: Number.parseFloat(element.getAttribute("opacity") || (removeAfter ? "1" : "0"))
      };
      animateState(
        element,
        values,
        { progress: 1, opacity: removeAfter ? 0 : 1 },
        duration,
        () => {
          element.setAttribute("d", pathAt(values.progress));
          element.setAttribute("opacity", String(values.opacity));
        },
        removeAfter ? () => element.remove() : null
      );
    };
    if (duration && animeApi()) {
      linkExit.each(function(link) { animateLink(this, link, true); });
      linkMerged.each(function(link) { animateLink(this, link, false); });
    } else {
      linkExit.remove();
      linkMerged.attr("opacity", 1).attr("d", link => linkPath(link, verticalOrientation, busX));
    }

    const nodes = nodesLayer.selectAll("g.cli-explorer-node").data(treeRoot.descendants(), node => node.data.id);
    const entered = nodes.enter().append("g")
      .attr("class", node => nodeClass(node.data))
      .attr("role", "treeitem")
      .attr("tabindex", 0)
      .attr("transform", verticalOrientation
        ? `translate(${source._x ?? 0},${source._y ?? 0})`
        : `translate(${source._y ?? 0},${source._x ?? 0})`)
      .attr("opacity", duration ? 0 : 1)
      .on("click", (event, node) => {
        event.stopPropagation();
        selectNode(node.data, true);
      })
      .on("keydown", (event, node) => {
        handleNodeKeydown(event, node);
      });

    entered.append("rect")
      .attr("class", "cli-explorer-node-hit")
      .attr("x", -12)
      .attr("y", -22)
      .attr("height", 44)
      .attr("width", NODE_WIDTH + 10)
      .attr("fill", "transparent");

    entered.append("rect")
      .attr("class", "cli-explorer-node-shell")
      .attr("x", -8)
      .attr("y", -16)
      .attr("height", NODE_HEIGHT)
      .attr("rx", node => node.data.kind === "command" || node.data.kind === "root" ? 8 : 5)
      .attr("width", NODE_WIDTH);

    entered.append("rect")
      .attr("class", "cli-explorer-node-marker")
      .attr("x", 3)
      .attr("y", -4)
      .attr("width", 8)
      .attr("height", 8)
      .attr("rx", node => node.data.kind === "positional" ? 1 : 4)
      .attr("transform", node => node.data.kind === "positional" ? "rotate(45 7 0)" : null);

    entered.append("text")
      .attr("x", 18)
      .attr("dy", "0.35em")
      .text(node => truncatedLabel(node.data));

    entered.append("text")
      .attr("class", "cli-explorer-node-count")
      .attr("x", NODE_COUNT_X)
      .attr("text-anchor", "end")
      .attr("dy", "0.35em")
      .text(node => node.data.originalChildren.length ? descendantsCount(node.data) : "");

    entered.append("path")
      .attr("class", "cli-explorer-node-disclosure")
      .attr("fill", "none")
      .attr("stroke-linecap", "round")
      .attr("stroke-linejoin", "round")
      .attr("d", node => disclosurePath(node.data, verticalOrientation));

    entered.append("title").text(node => nodeLabel(node.data));

    const merged = entered.merge(nodes)
      .attr("class", node => nodeClass(node.data))
      .attr("role", "treeitem")
      .attr("aria-label", node => `${node.data.kind}: ${nodeLabel(node.data)}${node.data.description ? `. ${node.data.description}` : ""}`)
      .attr("data-node-id", node => node.data.id)
      .attr("aria-level", node => node.depth + 1)
      .attr("aria-posinset", node => (node.parent ? node.parent.children.indexOf(node) : 0) + 1)
      .attr("aria-setsize", node => node.parent?.children?.length || 1)
      .attr("aria-selected", node => String(node.data === state.selected))
      .attr("tabindex", node => node.data === state.selected ? "0" : "-1")
      .attr("data-path", node => node.data.path)
      .attr("aria-expanded", node => node.data.originalChildren.length ? String(node.data.expanded) : null);

    merged.select(".cli-explorer-node-shell")
      .attr("x", verticalOrientation ? -106 : -8);
    merged.select(".cli-explorer-node-hit")
      .attr("x", verticalOrientation ? -110 : -12);
    merged.select(".cli-explorer-node-marker")
      .attr("x", verticalOrientation ? -95 : 3)
      .attr("transform", node => {
        if (node.data.kind !== "positional") return null;
        return verticalOrientation ? "rotate(45 -91 0)" : "rotate(45 7 0)";
      });
    merged.select("text:not(.cli-explorer-node-count)")
      .attr("x", verticalOrientation ? -80 : 18);
    merged.select("text.cli-explorer-node-count")
      .attr("x", NODE_COUNT_X + (verticalOrientation ? MOBILE_NODE_OFFSET_X : 0));
    merged.select(".cli-explorer-node-disclosure")
      .attr("d", node => disclosurePath(node.data, verticalOrientation));

    const animateNode = (element, node, removeAfter) => {
      const fromTransform = element.getAttribute("transform") || nodeTransform(node);
      const targetTransform = nodeTransform(node);
      const transformAt = d3.interpolateString(fromTransform, targetTransform);
      const values = {
        progress: 0,
        opacity: Number.parseFloat(element.getAttribute("opacity") || (removeAfter ? "1" : "0"))
      };
      animateState(
        element,
        values,
        { progress: 1, opacity: removeAfter ? 0 : 1 },
        duration,
        () => {
          element.setAttribute("transform", transformAt(values.progress));
          element.setAttribute("opacity", String(values.opacity));
        },
        removeAfter ? () => element.remove() : null
      );
    };
    if (duration && animeApi()) {
      merged.each(function(node) { animateNode(this, node, false); });
      nodes.exit().each(function(node) { animateNode(this, node, true); });
    } else {
      merged.attr("opacity", 1).attr("transform", nodeTransform);
      nodes.exit().remove();
    }
    treeRoot.each(node => {
      node.data._x = node.x;
      node.data._y = node.y;
    });

    if (visibleCount) visibleCount.textContent = `${treeRoot.descendants().length} nodes visible`;
    if (!state.initialized) {
      state.initialized = true;
      requestAnimationFrame(() => fitTree(false));
    }
  }

  function ancestors(node) {
    const result = [];
    let cursor = node.parentNode;
    while (cursor) {
      result.push(cursor);
      cursor = cursor.parentNode;
    }
    return result;
  }

  function selectNode(node, toggle = false) {
    state.selected = node;
    let openedFocusedBranch = false;
    if (toggle && node.originalChildren.length) {
      const opening = !node.expanded;
      if (opening && isNarrow() && node.kind !== "root") {
        openedFocusedBranch = true;
        index.forEach(item => { if (item.kind !== "root") item.expanded = false; });
        const path = [node, ...ancestors(node)];
        path.forEach(item => { item.expanded = true; });
        state.focusPath = new Set(path);
        state.focusBranch = node;
      } else {
        node.expanded = opening;
        state.focusPath = null;
        state.focusBranch = null;
      }
    }
    renderInspector(node);
    update(node, openedFocusedBranch);
    if (state.focusBranch || (toggle && node.originalChildren.length)) {
      const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 210;
      window.setTimeout(() => requestAnimationFrame(() => fitTree()), delay);
    }
  }

  function displayValue(value) {
    if (Array.isArray(value)) return value.join(", ");
    if (value === true) return "yes";
    if (value === false || value === undefined || value === null || value === "") return "";
    return String(value);
  }

  function renderInspector(node) {
    const inspector = app.querySelector("[data-inspector]");
    if (!inspector) return;
    const details = [
      ["Aliases", displayValue(node.aliases)],
      ["Short", displayValue(node.short)],
      ["Long", displayValue(node.long)],
      ["Value names", displayValue(node.valueNames)],
      ["Default", displayValue(node.defaultValues)],
      ["Possible values", displayValue(node.possibleValues)],
      ["Conflicts", displayValue(node.conflicts)],
      ["Repeatable", displayValue(node.repeatable)],
      ["Required", displayValue(node.required)],
      ["Global", displayValue(node.global)],
      ["Inventory id", displayValue(node.inventoryId)]
    ].filter(([, value]) => value !== "");
    const badges = [
      node.hidden ? ["hidden", "hidden"] : null,
      node.global ? ["global", "global"] : null,
      node.required ? ["required", "required"] : null,
      node.synthetic ? ["synthetic", "synthetic"] : null,
      node.external ? ["plugin catch-all", "external"] : null,
      node.deprecated ? ["deprecated", "deprecated"] : null
    ].filter(Boolean);
    const invocation = invocationFor(node);
    const patternNote = node.external
      ? `<p class="cli-explorer-pattern-note">Pattern only — the installed plugin supplies the command name and arguments.</p>`
      : "";
    const copyButton = node.external
      ? ""
      : `<button class="cli-explorer-copy-button" type="button" aria-label="Copy command path" title="Copy command path" data-copy="${escapeAttr(invocation)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"></rect><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8h2"></path></svg></button>`;
    inspector.innerHTML = `
      <div class="cli-explorer-kind-line"><i class="cli-explorer-kind-marker ${escapeHtml(node.kind)}"></i>${escapeHtml(node.kind === "root" ? "command root" : node.kind)}</div>
      <h2>${escapeHtml(nodeLabel(node))}</h2>
      <p class="cli-explorer-inspector-copy">${escapeHtml(node.description || "No additional description is attached to this node.")}</p>
      ${badges.length ? `<div class="cli-explorer-badge-row">${badges.map(([label, cls]) => `<span class="cli-explorer-badge ${cls}">${escapeHtml(label)}</span>`).join("")}</div>` : ""}
      <div class="cli-explorer-command-path"><span>${escapeHtml(invocation)}</span>${copyButton}</div>
      ${patternNote}
      ${details.length ? `<dl class="cli-explorer-meta-list">${details.map(([key, value]) => `<div class="cli-explorer-meta-row"><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : ""}
    `;
    const button = inspector.querySelector("[data-copy]");
    if (button) button.addEventListener("click", () => copyInvocation(button.dataset.copy || invocation, button));
  }

  function invocationFor(node) {
    if (node.external) return "mesh-llm <plugin-command> [args…]";
    const commands = [];
    let cursor = node;
    while (cursor) {
      if (cursor.kind === "root" || cursor.kind === "command") commands.unshift(cursor.name);
      cursor = cursor.parentNode;
    }
    let invocation = commands.join(" ");
    if (node.kind === "option" || node.kind === "positional") {
      const token = node.kind === "option"
        ? (node.long ? `--${String(node.long).replace(/^-+/, "")}` : optionToken(node))
        : node.name;
      invocation = `${invocation} ${token}`.trim();
      if (node.valueNames.length) invocation += ` ${node.valueNames[0]}`;
    }
    return invocation;
  }

  async function copyInvocation(value, button) {
    let copied = false;
    try {
      await navigator.clipboard.writeText(value);
      copied = true;
    } catch {
      try {
        const textarea = document.createElement("textarea");
        textarea.value = value;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        copied = document.execCommand("copy");
        textarea.remove();
      } catch {
        copied = false;
      }
    }
    if (button) button.setAttribute("aria-label", copied ? "Copied command path" : "Copy failed");
    if (copyStatus) copyStatus.textContent = copied ? `Copied ${value}` : "Copy failed — select the command path manually.";
    window.setTimeout(() => {
      if (button) button.setAttribute("aria-label", "Copy command path");
      if (copyStatus) copyStatus.textContent = "";
    }, 1500);
  }

  function revealNode(node) {
    if (node.hidden && !state.showHidden) {
      state.showHidden = true;
      app.querySelector('[data-action="hidden"]')?.setAttribute("aria-pressed", "true");
    }
    if (node.kind === "option" && node.parentNode?.kind === "root") {
      state.showRootOptions = true;
      state.focusedRootOption = node;
      app.querySelector('[data-action="root-options"]')?.setAttribute("aria-pressed", "true");
    } else {
      state.focusedRootOption = null;
    }
    index.forEach(item => { if (item.kind !== "root") item.expanded = false; });
    const path = [node, ...ancestors(node)];
    path.forEach(item => { item.expanded = true; });
    state.focusPath = new Set(path);
    state.focusBranch = null;
    state.selected = node;
    renderInspector(node);
    update(node, true);
    requestAnimationFrame(() => fitTree(false));
  }

  function fitTree(animate = true) {
    const bounds = state.layoutBounds;
    if (!bounds) return;
    if (!bounds.width || !bounds.height) return;
    const container = app.querySelector(".cli-explorer-canvas-wrap");
    if (!container) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (!width || !height) return;
    const computedScale = 0.88 / Math.max(bounds.width / width, bounds.height / height);
    const overviewScaleFloor = !state.focusPath && !state.focusBranch && !hasExpandedNonRoot() ? 0.9 : 0.72;
    const scale = Math.max(overviewScaleFloor, Math.min(1.15, computedScale));
    const scaledWidth = bounds.width * scale;
    const scaledHeight = bounds.height * scale;
    const verticalOrientation = isVerticalOrientation();
    const x = verticalOrientation && scaledWidth > width - 56
      ? width / 2 - (cliRoot._x ?? 0) * scale
      : scaledWidth > width - 56
      ? 28 - bounds.x * scale
      : (width - scaledWidth) / 2 - bounds.x * scale;
    const legendSafeTop = 64;
    const centeredY = (height - scaledHeight) / 2 - bounds.y * scale;
    const overviewLayout = !state.focusPath && !state.focusBranch && !hasExpandedNonRoot();
    const y = overviewLayout
      ? legendSafeTop - bounds.y * scale
      : scaledHeight > height - 56
      ? legendSafeTop - bounds.y * scale
      : Math.max(legendSafeTop - bounds.y * scale, centeredY);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const targetTransform = d3.zoomIdentity.translate(x, y).scale(scale);
    if (!animate || reduced || !animeApi()) {
      stopAnimation(svgNode);
      svg.call(zoom.transform, targetTransform);
      return;
    }
    const currentTransform = d3.zoomTransform(svgNode);
    const values = { x: currentTransform.x, y: currentTransform.y, k: currentTransform.k };
    animateState(
      svgNode,
      values,
      { x: targetTransform.x, y: targetTransform.y, k: targetTransform.k },
      260,
      () => svg.call(zoom.transform, d3.zoomIdentity.translate(values.x, values.y).scale(values.k))
    );
  }

  function collapseAll() {
    index.forEach(node => { node.expanded = node.kind === "root"; });
    state.focusedRootOption = null;
    state.focusPath = null;
    state.focusBranch = null;
    state.showRootOptions = false;
    app.querySelector('[data-action="root-options"]')?.setAttribute("aria-pressed", "false");
    update(cliRoot);
    requestAnimationFrame(() => fitTree());
  }

  function matches(node, query) {
    const haystack = [
      node.name,
      node.path,
      node.description,
      node.aliases,
      node.short,
      node.long,
      node.valueNames,
      node.defaultValues,
      node.possibleValues,
      node.conflicts
    ].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(query.toLowerCase());
  }

  function setSearchOpen(open) {
    const results = app.querySelector("[role=listbox]");
    const search = app.querySelector("[data-search-input]") || app.querySelector("#cli-explorer-search");
    if (!results || !search) return;
    results.classList.toggle("is-open", open);
    results.setAttribute("aria-hidden", String(!open));
    search.setAttribute("aria-expanded", String(open));
    if (!open) search.removeAttribute("aria-activedescendant");
  }

  function renderSearch(query) {
    const results = app.querySelector("[role=listbox]");
    const search = app.querySelector("#cli-explorer-search");
    if (!results || !search) return;
    state.query = query.trim();
    state.activeResult = -1;
    search.removeAttribute("aria-activedescendant");
    if (!state.query) {
      state.focusPath = null;
      state.focusBranch = null;
      state.focusedRootOption = null;
      results.innerHTML = "";
      setSearchOpen(false);
      update(state.selected, true);
      return;
    }
    const found = index.filter(node => matches(node, state.query)).slice(0, 24);
    results.innerHTML = found.length
      ? found.map(node => `<button id="cli-search-result-${node.id}" class="cli-explorer-search-result" type="button" role="option" aria-selected="false" data-id="${escapeAttr(node.id)}" data-kind="${escapeAttr(node.kind)}" data-path="${escapeAttr(node.path)}"><i class="cli-explorer-result-dot"></i><span class="cli-explorer-result-main"><span class="cli-explorer-result-name">${highlight(nodeLabel(node), state.query)}</span><span class="cli-explorer-result-path">${escapeHtml(node.path)}</span></span><span class="cli-explorer-result-kind">${escapeHtml(node.kind)}</span></button>`).join("")
      : `<div class="cli-explorer-search-result" role="status"><span></span><span class="cli-explorer-result-main"><span class="cli-explorer-result-name">No matching command or option</span><span class="cli-explorer-result-path">Try a shorter search term.</span></span></div>`;
    setSearchOpen(true);
    results.querySelectorAll("[data-id]").forEach(button => button.addEventListener("click", () => {
      const node = index.find(item => item.id === button.dataset.id);
      if (!node) return;
      revealNode(node);
      setSearchOpen(false);
      search.focus();
    }));
    update(state.selected, true);
  }

  function updateActiveResult(items) {
    items.forEach((item, itemIndex) => {
      const active = itemIndex === state.activeResult;
      item.classList.toggle("is-active", active);
      item.setAttribute("aria-selected", String(active));
    });
    const active = items[state.activeResult];
    const search = app.querySelector("#cli-explorer-search");
    if (active && search) search.setAttribute("aria-activedescendant", active.id);
  }

  function highlight(text, query) {
    const escaped = escapeHtml(text);
    const safeQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return escaped.replace(new RegExp(`(${safeQuery})`, "ig"), "<mark>$1</mark>");
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;"
    }[character]));
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#096;");
  }

  const searchInput = app.querySelector("#cli-explorer-search");
  searchInput?.addEventListener("input", event => renderSearch(event.target.value));
  searchInput?.addEventListener("keydown", event => {
    const items = [...app.querySelectorAll(".cli-explorer-search-result[data-id]")];
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!items.length) return;
      state.activeResult = event.key === "ArrowDown"
        ? Math.min(items.length - 1, state.activeResult + 1)
        : Math.max(0, state.activeResult - 1);
      updateActiveResult(items);
      items[state.activeResult]?.scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter" && state.activeResult >= 0) {
      event.preventDefault();
      items[state.activeResult]?.click();
    } else if (event.key === "Escape") {
      setSearchOpen(false);
    }
  });

  document.addEventListener("keydown", event => {
    const target = event.target;
    if (event.key === "/" && target !== searchInput && !(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement) && !target?.isContentEditable) {
      event.preventDefault();
      searchInput?.focus();
    }
  });

  document.addEventListener("click", event => {
    if (!event.target.closest(".cli-explorer-search-wrap")) setSearchOpen(false);
  });

  app.querySelectorAll('[data-action="orientation"]').forEach(button => {
    button.addEventListener("click", event => {
      const nextOrientation = event.currentTarget.dataset.orientation;
      if (nextOrientation !== "horizontal" && nextOrientation !== "vertical") return;
      state.orientationPreference = nextOrientation;
      syncOrientationControls(nextOrientation);
      update(state.selected, true);
      // Give the new shell geometry one paint before fitting.  The follow-up
      // timer also covers browsers that flush SVG bounds after the animation
      // frame, so a switch cannot retain the previous orientation's pan.
      requestAnimationFrame(() => {
        fitTree(false);
        window.setTimeout(() => fitTree(false), 0);
      });
    });
  });

  app.querySelector('[data-action="collapse"]')?.addEventListener("click", collapseAll);
  app.querySelector('[data-action="fit"]')?.addEventListener("click", () => fitTree());
  app.querySelector('[data-action="root-options"]')?.addEventListener("click", event => {
    state.showRootOptions = !state.showRootOptions;
    state.focusedRootOption = null;
    state.focusPath = null;
    state.focusBranch = null;
    event.currentTarget.setAttribute("aria-pressed", String(state.showRootOptions));
    update(state.selected);
    requestAnimationFrame(() => fitTree());
  });
  app.querySelector('[data-action="hidden"]')?.addEventListener("click", event => {
    state.showHidden = !state.showHidden;
    state.focusPath = null;
    state.focusBranch = null;
    event.currentTarget.setAttribute("aria-pressed", String(state.showHidden));
    update(state.selected);
    requestAnimationFrame(() => fitTree());
  });

  window.addEventListener("resize", () => {
    if (state.resizeFrame) cancelAnimationFrame(state.resizeFrame);
    state.resizeFrame = requestAnimationFrame(() => {
      state.resizeFrame = 0;
      update(state.selected, true);
      fitTree(false);
    });
  });

  renderInspector(cliRoot);
  update(cliRoot, true);
})();
