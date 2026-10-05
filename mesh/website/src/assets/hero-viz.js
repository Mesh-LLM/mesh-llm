(function () {
  'use strict';

  var S = 'http://www.w3.org/2000/svg';
  var DEFAULT_VIEWBOX_WIDTH = 1440;
  var DEFAULT_VIEWBOX_HEIGHT = 740;
  var activeViewBox = { width: DEFAULT_VIEWBOX_WIDTH, height: DEFAULT_VIEWBOX_HEIGHT };
  var EDGE_DASH = '7 13';
  var TOKEN_START = 8580;
  var TOKEN_STAGGER = 300;
  var TOKEN_DURATION = 520;
  var RETURN_PACKET_DURATION = 520;
  var PANEL_COLLAPSE_DELAY = 860;
  var RIM_TRACE_START = 7200;
  var RIM_TRACE_DURATION = 640;
  var TITLE_FINAL_PREFIX = 'Run any model across nodes ';
  var TITLE_PHRASES = ['in your business', 'in your homelab', 'on the internet'];
  var SUBTITLE_DOT_REST = 'rgba(212,212,216,0.88)';
  var SUBTITLE_DOT_HOT = '#ff4b4b';
  var SUBTITLE_CHAR_STAGGER = 25;
  var SUBTITLE_CHAR_IN_START = 100;
  var SUBTITLE_CHAR_IN_DURATION = 150;
  var SUBTITLE_DOT_LERP = 0.84;
  var SUBTITLE_DOT_SNAP = 0.35;
  var SUBTITLE_LOOP_DELAY = 2400;
  var SUBTITLE_LOOP_RETRY_DELAY = 900;
  var SUBTITLE_LOOP_FRAME_BUDGET = 48;
  var SUBTITLE_LOOP_PERF_COOLDOWN = 2600;
  var NODE_PLATE_RATIO = 0.48;
  var NODE_ICON_DIAMETER_RATIO = 0.576;
  var SVG_MOTION_EVENT = 'mesh:hero-viz:svg-motion';
  var SVG_MOTION_CLASS = 'is-svg-motion-active';
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var isSafari = /^((?!chrome|android|crios|fxios).)*safari/i.test(window.navigator.userAgent);

  if (isSafari) document.documentElement.classList.add('is-safari');

  var nodes = [
    { id: 'workstation', type: '96GB VRAM', label: 'workstation', x: 1128, y: 392, r: 34, color: '#60a5fa', icon: 'Monitor', primary: true },
    { id: 'laptop', type: '24GB VRAM', label: 'laptop', x: 396, y: 168, r: 26, color: '#93c5fd', icon: 'Laptop' },
    { id: 'server', type: '768GB VRAM', label: 'server', x: 666, y: 274, r: 42, color: '#22c55e', icon: 'Server', target: true },
    { id: 'mini', type: '8GB VRAM', label: 'mini pc', x: 746, y: 122, r: 28, color: '#38bdf8', icon: 'Router' },
    { id: 'gpu', type: '32GB VRAM', label: 'gpu rig', x: 210, y: 284, r: 36, color: '#86efac', icon: 'Gpu' },
    { id: 'cloud', type: '192GB VRAM', label: 'cloud node', x: 1240, y: 214, r: 30, color: '#a5b4fc', icon: 'Cloud' },
  ];

  var edges = [
    { id: 'e1', from: 'workstation', to: 'cloud', d: 'M 1128 392 Q 1278 326 1240 214', color: '#60a5fa', opacity: 0.48 },
    { id: 'e2', from: 'workstation', to: 'server', d: 'M 1128 392 Q 904 414 666 274', color: '#60a5fa', opacity: 0.44 },
    { id: 'e3', from: 'laptop', to: 'server', d: 'M 396 168 Q 536 148 666 274', color: '#93c5fd', opacity: 0.38 },
    { id: 'e4', from: 'server', to: 'mini', d: 'M 666 274 Q 702 182 746 122', color: '#22c55e', opacity: 0.42 },
    { id: 'e5', from: 'server', to: 'cloud', d: 'M 666 274 Q 940 140 1240 214', color: '#cbd5e1', opacity: 0.30 },
    { id: 'e6', from: 'gpu', to: 'laptop', d: 'M 210 284 Q 270 184 396 168', color: '#86efac', opacity: 0.34 },
    { id: 'e7', from: 'server', to: 'gpu', d: 'M 666 274 Q 438 414 210 284', color: '#22c55e', opacity: 0.28 },
  ];

  var topologyLayouts = [
    {
      name: 'phone',
      width: 560,
      height: 900,
      minViewport: 320,
      nodes: {
        workstation: { x: 500, y: 500, r: 50 },
        laptop: { x: 150, y: 154, r: 40 },
        server: { x: 285, y: 280, r: 60 },
        mini: { x: 365, y: 126, r: 38 },
        gpu: { x: 115, y: 426, r: 48 },
        cloud: { x: 455, y: 350, r: 42 },
      },
      controls: {
        e1: { x: 535, y: 420 },
        e2: { x: 355, y: 535 },
        e3: { x: 202, y: 196 },
        e4: { x: 356, y: 210 },
        e5: { x: 410, y: 335 },
        e6: { x: 72, y: 288 },
        e7: { x: 182, y: 378 },
      },
      bg: {
        blue: { x: 280, y: 330, r: 280 },
        green: { x: 385, y: 545, r: 220 },
      },
      panel: {
        x: 285,
        y: 460,
        anchor: 'top-center',
        connector: 'vertical',
        exitAngle: 1.5708,
        traceShape: 'circle',
        traceStart: -1.5708,
        traceEnd: 1.5708,
        traceSweep: 1,
      },
    },
    {
      name: 'compact',
      width: 820,
      height: 850,
      minViewport: 640,
      nodes: {
        workstation: { x: 628, y: 520, r: 40 },
        laptop: { x: 248, y: 188, r: 32 },
        server: { x: 430, y: 275, r: 48 },
        mini: { x: 505, y: 120, r: 32 },
        gpu: { x: 190, y: 415, r: 40 },
        cloud: { x: 655, y: 250, r: 35 },
      },
      controls: {
        e1: { x: 730, y: 396 },
        e2: { x: 595, y: 386 },
        e3: { x: 318, y: 198 },
        e4: { x: 480, y: 190 },
        e5: { x: 550, y: 230 },
        e6: { x: 170, y: 280 },
        e7: { x: 296, y: 426 },
      },
      bg: {
        blue: { x: 410, y: 310, r: 300 },
        green: { x: 560, y: 430, r: 245 },
      },
      panel: {
        x: 430,
        y: 535,
        anchor: 'top-center',
        connector: 'vertical',
        exitAngle: 1.5708,
        traceShape: 'circle',
        traceStart: -1.5708,
        traceEnd: 1.5708,
        traceSweep: 1,
      },
    },
    {
      name: 'balanced',
      width: 1080,
      height: 780,
      minViewport: 900,
      nodes: {
        workstation: { x: 895, y: 398, r: 36 },
        laptop: { x: 300, y: 190, r: 30 },
        server: { x: 558, y: 302, r: 46 },
        mini: { x: 640, y: 146, r: 31 },
        gpu: { x: 170, y: 360, r: 38 },
        cloud: { x: 955, y: 214, r: 33 },
      },
      controls: {
        e1: { x: 1000, y: 318 },
        e2: { x: 754, y: 430 },
        e3: { x: 414, y: 174 },
        e4: { x: 592, y: 198 },
        e5: { x: 750, y: 190 },
        e6: { x: 226, y: 236 },
        e7: { x: 340, y: 420 },
      },
      bg: {
        blue: { x: 530, y: 300, r: 320 },
        green: { x: 780, y: 320, r: 270 },
      },
      panel: {
        x: 558,
        y: 470,
        anchor: 'top-center',
        connector: 'vertical',
        exitAngle: 1.5708,
        traceShape: 'circle',
        traceStart: -1.5708,
        traceEnd: 1.5708,
        traceSweep: 1,
      },
    },
    {
      name: 'wide',
      width: 1440,
      height: 740,
      minViewport: 1280,
      nodes: {
        workstation: { x: 1128, y: 392, r: 34 },
        laptop: { x: 396, y: 168, r: 26 },
        server: { x: 666, y: 274, r: 42 },
        mini: { x: 746, y: 122, r: 28 },
        gpu: { x: 210, y: 284, r: 36 },
        cloud: { x: 1240, y: 214, r: 30 },
      },
      controls: {
        e1: { x: 1278, y: 326 },
        e2: { x: 904, y: 414 },
        e3: { x: 536, y: 148 },
        e4: { x: 702, y: 182 },
        e5: { x: 940, y: 140 },
        e6: { x: 270, y: 184 },
        e7: { x: 438, y: 414 },
      },
      bg: {
        blue: { x: 680, y: 270, r: 380 },
        green: { x: 1010, y: 320, r: 310 },
      },
      panel: {
        x: 1160,
        y: 490,
        anchor: 'left-center',
        connector: 'elbow',
        exitAngle: 0.7854,
        traceShape: 'circle',
        traceStart: -0.7854,
        traceEnd: 0.7854,
      },
    },
  ];

  var routes = [
    { edge: 'e2', source: '#60a5fa', target: '#22c55e', delay: 0, duration: 4800 },
    { edge: 'e3', source: '#93c5fd', target: '#22c55e', delay: 1200, duration: 5200 },
    { edge: 'e5', source: '#22c55e', target: '#a5b4fc', delay: 2600, duration: 5600 },
    { edge: 'e4', source: '#22c55e', target: '#38bdf8', delay: 3900, duration: 4600 },
    { edge: 'e7', source: '#22c55e', target: '#86efac', delay: 5200, duration: 5400 },
  ];

  function el(name, attrs, parent) {
    var node = document.createElementNS(S, name);
    for (var key in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) node.setAttribute(key, attrs[key]);
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function nodeById(id) {
    return nodes.filter(function (node) { return node.id === id; })[0];
  }

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function mixHex(a, b, progress) {
    function channel(hex, offset) { return parseInt(hex.slice(offset, offset + 2), 16); }
    var r = Math.round(channel(a, 1) + (channel(b, 1) - channel(a, 1)) * progress);
    var g = Math.round(channel(a, 3) + (channel(b, 3) - channel(a, 3)) * progress);
    var c = Math.round(channel(a, 5) + (channel(b, 5) - channel(a, 5)) * progress);
    return 'rgb(' + r + ',' + g + ',' + c + ')';
  }

  function copyAttrs(attrs) {
    var copy = {};
    for (var key in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, key)) copy[key] = attrs[key];
    }
    return copy;
  }

  function roundSvg(value) {
    return Math.round(value * 100) / 100;
  }

  function lerp(a, b, progress) {
    return a + (b - a) * progress;
  }

  function blendPoint(a, b, progress) {
    return {
      x: lerp(a.x, b.x, progress),
      y: lerp(a.y, b.y, progress),
      r: lerp(a.r || 0, b.r || 0, progress),
    };
  }

  function blendLayout(a, b, progress) {
    var fixed = progress < 0.5 ? a : b;
    var layout = {
      name: fixed.name,
      width: lerp(a.width, b.width, progress),
      height: lerp(a.height, b.height, progress),
      nodes: {},
      controls: {},
      panel: fixed.panel,
      bg: {
        blue: blendPoint(a.bg.blue, b.bg.blue, progress),
        green: blendPoint(a.bg.green, b.bg.green, progress),
      },
    };

    nodes.forEach(function (node) {
      layout.nodes[node.id] = blendPoint(a.nodes[node.id], b.nodes[node.id], progress);
    });

    edges.forEach(function (edge) {
      layout.controls[edge.id] = blendPoint(a.controls[edge.id], b.controls[edge.id], progress);
    });

    return layout;
  }

  function resolveTopologyLayout(viz) {
    var width = Math.max(320, viz.clientWidth || window.innerWidth || DEFAULT_VIEWBOX_WIDTH);

    if (width <= topologyLayouts[0].minViewport) return topologyLayouts[0];

    for (var i = 0; i < topologyLayouts.length - 1; i += 1) {
      var from = topologyLayouts[i];
      var to = topologyLayouts[i + 1];

      if (width <= to.minViewport) {
        return blendLayout(from, to, (width - from.minViewport) / (to.minViewport - from.minViewport));
      }
    }

    return topologyLayouts[topologyLayouts.length - 1];
  }

  function pathForEdge(edge, layout) {
    var from = layout.nodes[edge.from];
    var to = layout.nodes[edge.to];
    var control = layout.controls[edge.id];
    return 'M ' + roundSvg(from.x) + ' ' + roundSvg(from.y) + ' Q ' + roundSvg(control.x) + ' ' + roundSvg(control.y) + ' ' + roundSvg(to.x) + ' ' + roundSvg(to.y);
  }

  function iconScaleFor(item) {
    return ((item.r * NODE_PLATE_RATIO * 2) * NODE_ICON_DIAMETER_RATIO) / 24;
  }

  function setCircle(circle, attrs) {
    if (!circle) return;
    circle.setAttribute('cx', roundSvg(attrs.x));
    circle.setAttribute('cy', roundSvg(attrs.y));
    circle.setAttribute('r', roundSvg(attrs.r));
  }

  function cachePathLength(path, length) {
    var measured = typeof length === 'number' ? length : path.getTotalLength();
    path.__meshPathLength = measured;
    return measured;
  }

  function getPathLength(path) {
    return typeof path.__meshPathLength === 'number' ? path.__meshPathLength : cachePathLength(path);
  }

  function setResponsiveTopology(viz, svg, scene) {
    var layout = resolveTopologyLayout(viz);
    activeViewBox.width = layout.width;
    activeViewBox.height = layout.height;
    svg.setAttribute('viewBox', '0 0 ' + roundSvg(layout.width) + ' ' + roundSvg(layout.height));
    viz.setAttribute('data-mesh-layout', layout.name);

    nodes.forEach(function (item) {
      var next = layout.nodes[item.id];
      item.x = roundSvg(next.x);
      item.y = roundSvg(next.y);
      item.r = roundSvg(next.r);

      if (!scene || !item.group) return;
      item.group.setAttribute('transform', 'translate(' + item.x + ' ' + item.y + ')');
      item.halo.setAttribute('r', item.r);
      item.plate.setAttribute('r', roundSvg(item.r * NODE_PLATE_RATIO));
      item.icon.setAttribute('transform', 'scale(' + roundSvg(iconScaleFor(item)) + ')');
      item.labelEl.setAttribute('y', roundSvg(item.r * 0.92));
      item.typeEl.setAttribute('y', roundSvg(item.r * 0.92 + 13));
    });

    edges.forEach(function (edge) {
      edge.d = pathForEdge(edge, layout);

      if (!scene || !edge.path) return;
      edge.path.setAttribute('d', edge.d);
      edge.solidPath.setAttribute('d', edge.d);
      edge.length = cachePathLength(edge.path);
      cachePathLength(edge.solidPath, edge.length);
      edge.solidPath.style.strokeDasharray = edge.length;
    });

    if (scene) {
      scene.layout = layout;
      if (scene.bgPlane) {
        scene.bgPlane.setAttribute('width', roundSvg(layout.width));
        scene.bgPlane.setAttribute('height', roundSvg(layout.height));
      }
      setCircle(scene.bgBlue, layout.bg.blue);
      setCircle(scene.bgGreen, layout.bg.green);
      scene.focus.setAttribute('cx', scene.primary.x);
      scene.focus.setAttribute('cy', scene.primary.y);
      scene.focus.setAttribute('r', roundSvg(scene.primary.r * 0.7));
      scene.request.setAttribute('cx', scene.primary.x);
      scene.request.setAttribute('cy', scene.primary.y);
      scene.returnPackets.forEach(function (packet) {
        packet.setAttribute('cx', scene.target.x);
        packet.setAttribute('cy', scene.target.y);
      });
    }

    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('mesh:layout'));
    }

    return layout;
  }

  function fallbackIcon(group) {
    el('rect', { x: -8, y: -8, width: 16, height: 16, rx: 3 }, group);
    el('path', { d: 'M -4 -2 L 4 -2 M -4 3 L 4 3' }, group);
  }

  function drawLucideIcon(iconName, group) {
    var icon = window.lucide && window.lucide.icons && window.lucide.icons[iconName];
    if (!icon) {
      fallbackIcon(group);
      return;
    }

    var inner = el('g', { transform: 'translate(-12 -12)' }, group);
    icon.forEach(function (shape) {
      el(shape[0], copyAttrs(shape[1]), inner);
    });
  }

  function ensureGlow(defs, item) {
    var id = 'node-glow-' + item.id;
    var old = defs.querySelector('#' + id);
    if (old) old.remove();

    var gradient = el('radialGradient', { id: id, cx: '50%', cy: '50%', r: '50%' }, defs);
    el('stop', { offset: '0%', 'stop-color': item.color, 'stop-opacity': '0.36' }, gradient);
    el('stop', { offset: '52%', 'stop-color': item.color, 'stop-opacity': '0.13' }, gradient);
    el('stop', { offset: '100%', 'stop-color': item.color, 'stop-opacity': '0' }, gradient);
    return id;
  }

  function activateEdge(edge) {
    if (edge.solidPath) {
      edge.solidPath.style.opacity = 0;
      edge.solidPath.style.strokeDashoffset = 0;
    }
    edge.path.style.strokeDasharray = EDGE_DASH;
    edge.path.style.strokeDashoffset = 0;
    edge.path.style.opacity = 1;
    edge.path.classList.add('is-active');
  }

  function svgPointToViz(svg, x, y) {
    var layout = getSvgLayout(svg);
    return {
      x: layout.padX + x * layout.scale,
      y: layout.padY + y * layout.scale,
    };
  }

  function getSvgLayout(svg) {
    var rect = svg.getBoundingClientRect();
    var viewBox = svg.viewBox && svg.viewBox.baseVal;
    var width = viewBox && viewBox.width ? viewBox.width : activeViewBox.width;
    var height = viewBox && viewBox.height ? viewBox.height : activeViewBox.height;
    var scale = Math.min(rect.width / width, rect.height / height);
    var renderedWidth = width * scale;
    var renderedHeight = height * scale;
    return {
      scale: scale,
      padX: (rect.width - renderedWidth) / 2,
      padY: (rect.height - renderedHeight) / 2,
    };
  }

  function vizPointToSvg(svg, x, y) {
    var layout = getSvgLayout(svg);
    return {
      x: (x - layout.padX) / layout.scale,
      y: (y - layout.padY) / layout.scale,
    };
  }

  function setPathLine(path, a, b) {
    path.setAttribute('d', 'M ' + a.x + ' ' + a.y + ' L ' + b.x + ' ' + b.y);
  }

  function polarPoint(center, radius, angle) {
    return {
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
    };
  }

  function setArcPath(path, center, radius, startAngle, endAngle, sweep) {
    var start = polarPoint(center, radius, startAngle);
    var end = polarPoint(center, radius, endAngle);
    var largeArc = Math.abs(endAngle - startAngle) > Math.PI ? 1 : 0;
    path.setAttribute('d', 'M ' + start.x + ' ' + start.y + ' A ' + radius + ' ' + radius + ' 0 ' + largeArc + ' ' + sweep + ' ' + end.x + ' ' + end.y);
  }

  function arcSamplePath(center, radius, startAngle, endAngle, direction) {
    var end = endAngle;
    var points = [];
    var steps;

    if (direction > 0) {
      while (end <= startAngle) end += Math.PI * 2;
    } else {
      while (end >= startAngle) end -= Math.PI * 2;
    }

    steps = Math.max(14, Math.ceil(Math.abs(end - startAngle) / (Math.PI / 20)));

    for (var i = 0; i <= steps; i += 1) {
      var progress = i / steps;
      var point = polarPoint(center, radius, startAngle + (end - startAngle) * progress);
      points.push((i === 0 ? 'M ' : 'L ') + roundSvg(point.x) + ' ' + roundSvg(point.y));
    }

    return points.join(' ');
  }

  function setCircleTracePaths(clockwisePath, counterPath, center, radius, startAngle, endAngle) {
    clockwisePath.setAttribute('d', arcSamplePath(center, radius, startAngle, endAngle, 1));
    counterPath.setAttribute('d', arcSamplePath(center, radius, startAngle, endAngle, -1));
  }

  function updatePanelConnector(viz, svg, scene, left, top, width, height) {
    if (!scene.panelAngle || !scene.panelStraight) return;

    var panelLeft = vizPointToSvg(svg, left, top + height / 2);
    var panelRight = vizPointToSvg(svg, left + width, top + height / 2);
    var panelTop = vizPointToSvg(svg, left, top);
    var panelBottom = vizPointToSvg(svg, left, top + height);
    var traceRadius = scene.target.r * 0.62;
    var panelConfig = scene.layout && scene.layout.panel;

    if (panelConfig) {
      var panelAnchor = panelConfig.anchor === 'left-center'
        ? { x: panelLeft.x - 3, y: (panelTop.y + panelBottom.y) / 2 }
        : { x: (panelLeft.x + panelRight.x) / 2, y: panelTop.y - 3 };
      var exitAngle = typeof panelConfig.exitAngle === 'number'
        ? panelConfig.exitAngle
        : Math.atan2(panelAnchor.y - scene.target.y, panelAnchor.x - scene.target.x);
      var exitPoint = polarPoint(scene.target, traceRadius, exitAngle);
      var elbow;

      if (panelConfig.connector === 'vertical') {
        elbow = { x: panelAnchor.x, y: panelAnchor.y - 16 };
      } else {
        var directionX = panelAnchor.x >= scene.target.x ? 1 : -1;
        var directionY = panelAnchor.y >= scene.target.y ? 1 : -1;
        var diagonalRun = Math.abs(panelAnchor.y - scene.target.y);
        var horizontalElbow = {
          x: scene.target.x + directionX * diagonalRun,
          y: panelAnchor.y,
        };
        var hasHorizontalRun = directionX > 0
          ? panelAnchor.x - horizontalElbow.x >= 24
          : horizontalElbow.x - panelAnchor.x >= 24;

        elbow = hasHorizontalRun
          ? horizontalElbow
          : {
            x: scene.target.x + (panelAnchor.x - scene.target.x) * 0.7,
            y: scene.target.y + (panelAnchor.y - scene.target.y) * 0.7,
          };
      }

      if (scene.nodeTraceCw && scene.nodeTraceCcw) {
        var startAngle = typeof panelConfig.traceStart === 'number' ? panelConfig.traceStart : exitAngle - Math.PI * 0.62;
        var endAngle = typeof panelConfig.traceEnd === 'number' ? panelConfig.traceEnd : exitAngle + Math.PI * 0.2;
        var sweep = typeof panelConfig.traceSweep === 'number' ? panelConfig.traceSweep : 1;

        if (panelConfig.traceShape === 'circle') {
          setCircleTracePaths(scene.nodeTraceCw, scene.nodeTraceCcw, scene.target, traceRadius, startAngle, endAngle);
        } else {
          setArcPath(scene.nodeTraceCw, scene.target, traceRadius, startAngle, endAngle, sweep);
          setArcPath(scene.nodeTraceCcw, scene.target, traceRadius, startAngle, endAngle, sweep ? 0 : 1);
        }
      }

      setPathLine(scene.panelAngle, exitPoint, elbow);
      setPathLine(scene.panelStraight, elbow, panelAnchor);
      return;
    }

    if (viz.clientWidth <= 700) {
      var panelCenterX = (panelLeft.x + panelRight.x) / 2;
      var panelAnchor = { x: panelCenterX, y: panelTop.y - 3 };
      var exitAngle = Math.atan2(panelAnchor.y - scene.target.y, panelAnchor.x - scene.target.x);
      var exitPoint = polarPoint(scene.target, traceRadius, exitAngle);
      var availableY = Math.max(36, panelAnchor.y - scene.target.y);
      var elbow = {
        x: panelAnchor.x,
        y: Math.min(panelAnchor.y - 12, scene.target.y + availableY * 0.72),
      };

      if (scene.nodeTraceCw && scene.nodeTraceCcw) {
        setArcPath(scene.nodeTraceCw, scene.target, traceRadius, exitAngle - Math.PI * 0.62, exitAngle + Math.PI * 0.2, 1);
        setArcPath(scene.nodeTraceCcw, scene.target, traceRadius, exitAngle - Math.PI * 0.62, exitAngle + Math.PI * 0.2, 0);
      }

      setPathLine(scene.panelAngle, exitPoint, elbow);
      setPathLine(scene.panelStraight, elbow, panelAnchor);
      return;
    }

    var panelOnRight = (panelLeft.x + panelRight.x) / 2 >= scene.target.x;
    var panelIsBelow = panelTop.y > scene.target.y;
    var preferredY = scene.target.y + (panelOnRight ? 86 : panelIsBelow ? 76 : -76);
    var lineY = clamp(preferredY, panelTop.y + 18, panelBottom.y - 18);
    var delta = Math.max(24, Math.abs(lineY - scene.target.y));
    var exitDirectionX = panelOnRight ? 1 : -1;
    var exitDirectionY = panelOnRight || lineY >= scene.target.y ? 1 : -1;
    var elbow = {
      x: scene.target.x + exitDirectionX * delta,
      y: scene.target.y + exitDirectionY * delta,
    };
    var exitAngle = Math.atan2(elbow.y - scene.target.y, elbow.x - scene.target.x);
    var exitPoint = polarPoint(scene.target, traceRadius, exitAngle);
    var panelAnchor = { x: (panelOnRight ? panelLeft.x - 3 : panelRight.x + 3), y: elbow.y };

    if (scene.nodeTraceCw && scene.nodeTraceCcw) {
      setArcPath(scene.nodeTraceCw, scene.target, traceRadius, -Math.PI * 0.75, Math.PI / 4, 1);
      setArcPath(scene.nodeTraceCcw, scene.target, traceRadius, -Math.PI * 0.75, Math.PI / 4, 0);
    }

    setPathLine(scene.panelAngle, exitPoint, elbow);
    setPathLine(scene.panelStraight, elbow, panelAnchor);
  }

  function positionPanel(viz, svg, scene) {
    if (!scene.panel || !scene.target) return;

    var panel = scene.panel;
    var width = panel.offsetWidth || 260;
    var height = panel.offsetHeight || 112;
    var layout = getSvgLayout(svg);
    var panelConfig = scene.layout && scene.layout.panel;
    var primaryPoint = {
      x: layout.padX + scene.primary.x * layout.scale,
      y: layout.padY + scene.primary.y * layout.scale,
    };
    var targetPoint = {
      x: layout.padX + scene.target.x * layout.scale,
      y: layout.padY + scene.target.y * layout.scale,
    };
    var panelBias = primaryPoint.x > targetPoint.x ? 0.36 : 0.40;
    var anchorX = primaryPoint.x + (targetPoint.x - primaryPoint.x) * panelBias;
    var requestEdge = edges.filter(function (edge) { return edge.id === 'e2'; })[0];
    var linkBottom = Math.max(primaryPoint.y, targetPoint.y);
    var lineClearance = Math.max(24, Math.min(54, viz.clientWidth * 0.045));
    var left = clamp(anchorX - width / 2, 18, Math.max(18, viz.clientWidth - width - 18));

    if (panelConfig) {
      var fixedPoint = svgPointToViz(svg, panelConfig.x, panelConfig.y);
      var fixedCenterX = panelConfig.connector === 'vertical' ? targetPoint.x : fixedPoint.x;
      var isShortLandscape = window.innerWidth > window.innerHeight && window.innerHeight <= 480;

      if (isShortLandscape) {
        fixedCenterX = Math.max(fixedCenterX, viz.clientWidth - width / 2 - 18);
      }

      left = clamp(fixedCenterX - width / 2, 18, Math.max(18, viz.clientWidth - width - 18));
      top = clamp(fixedPoint.y, 18, Math.max(18, viz.clientHeight - height - 18));

      panel.style.left = Math.round(left) + 'px';
      panel.style.top = Math.round(top) + 'px';
      panel.style.right = 'auto';
      updatePanelConnector(viz, svg, scene, left, top, width, height);
      return;
    }

    if (viz.clientWidth <= 700) {
      var nodeBottom = targetPoint.y + scene.target.r * layout.scale * 0.7;

      nodes.forEach(function (node) {
        var point = {
          x: layout.padX + node.x * layout.scale,
          y: layout.padY + node.y * layout.scale,
        };

        nodeBottom = Math.max(nodeBottom, point.y + node.r * layout.scale * 0.62);
      });

      if (viz.clientWidth > 560) {
        left = clamp(viz.clientWidth - width - 18, 18, Math.max(18, viz.clientWidth - width - 18));
      } else {
        left = clamp(targetPoint.x - width / 2, 18, Math.max(18, viz.clientWidth - width - 18));
      }

      top = clamp(nodeBottom + Math.max(24, viz.clientWidth * 0.06), 18, Math.max(18, viz.clientHeight - height - 18));

      if (viz.clientWidth <= 560) {
        top = Math.min(top, Math.max(18, viz.clientHeight - height - Math.max(148, viz.clientHeight * 0.2)));
      }

      panel.style.left = Math.round(left) + 'px';
      panel.style.top = Math.round(top) + 'px';
      panel.style.right = 'auto';
      updatePanelConnector(viz, svg, scene, left, top, width, height);
      return;
    }

    if (requestEdge && requestEdge.solidPath && requestEdge.solidPath.getBBox) {
      var linkBox = requestEdge.solidPath.getBBox();
      linkBottom = layout.padY + (linkBox.y + linkBox.height) * layout.scale;
    }

    nodes.forEach(function (node) {
      if (node.id === scene.target.id) return;

      var point = {
        x: layout.padX + node.x * layout.scale,
        y: layout.padY + node.y * layout.scale,
      };
      var nodeClearance = Math.max(18, (node.r + 15) * layout.scale);
      var overlapsPanelX = point.x > left - nodeClearance && point.x < left + width + nodeClearance;
      var sitsBelowLink = point.y > linkBottom - nodeClearance * 0.8;

      if (overlapsPanelX && sitsBelowLink) {
        linkBottom = Math.max(linkBottom, point.y + nodeClearance);
      }
    });

    var top = clamp(linkBottom + lineClearance, 18, Math.max(18, viz.clientHeight - height - 18));

    panel.style.left = Math.round(left) + 'px';
    panel.style.top = Math.round(top) + 'px';
    panel.style.right = 'auto';
    updatePanelConnector(viz, svg, scene, left, top, width, height);
  }

  function titleTextRect(scene) {
    var headingRect = scene.titleHeading ? scene.titleHeading.getBoundingClientRect() : null;
    var subtitleRect = scene.subtitle ? scene.subtitle.getBoundingClientRect() : null;

    if (!headingRect && !subtitleRect) {
      return scene.title ? scene.title.getBoundingClientRect() : null;
    }

    if (!headingRect) return subtitleRect;
    if (!subtitleRect) return headingRect;

    return {
      left: Math.min(headingRect.left, subtitleRect.left),
      right: Math.max(headingRect.right, subtitleRect.right),
      top: Math.min(headingRect.top, subtitleRect.top),
      bottom: Math.max(headingRect.bottom, subtitleRect.bottom),
      width: Math.max(headingRect.right, subtitleRect.right) - Math.min(headingRect.left, subtitleRect.left),
      height: Math.max(headingRect.bottom, subtitleRect.bottom) - Math.min(headingRect.top, subtitleRect.top),
    };
  }

  function shortPhoneTitleFooterReserve(viewportWidth, stageHeight, footerGap, isTouchLandscape) {
    var shortStageProgress;

    if (isTouchLandscape || viewportWidth > 390) return 0;

    shortStageProgress = clamp((660 - stageHeight) / 48, 0, 1);
    if (shortStageProgress <= 0) return 0;

    return Math.round(clamp(footerGap * 0.5 + shortStageProgress * 8, 14, 20));
  }

  function positionTitle(viz, svg, scene) {
    var fallbackFooterTop;
    var footer;
    var footerGap;
    var footerRect;
    var footerTop;
    var headingRect;
    var isTouchLandscape;
    var layout;
    var maxTop;
    var overlapsPanelX;
    var panelClearance;
    var panelRect;
    var responsive;
    var section;
    var sectionRect;
    var stage;
    var stageHeight;
    var stageRect;
    var targetRadius;
    var targetY;
    var titleHeight;
    var titleMin;
    var titleRect;
    var titleVizClearance;
    var top;
    var vizRect;
    var viewportWidth;
    var titleFooterReserve;

    if (!scene.title || !scene.target) return;

    section = scene.title.closest('section.hero');
    if (!section) return;

    layout = getSvgLayout(svg);
    vizRect = viz.getBoundingClientRect();
    sectionRect = section.getBoundingClientRect();
    stage = section.querySelector('.hero-stage-sticky');
    footer = section.querySelector('.hero-footer');
    stageRect = stage ? stage.getBoundingClientRect() : sectionRect;
    stageHeight = stage ? stage.clientHeight : section.clientHeight;
    viewportWidth = window.innerWidth;
    isTouchLandscape = window.matchMedia && window.matchMedia('(pointer: coarse) and (orientation: landscape)').matches;
    targetY = vizRect.top - stageRect.top + layout.padY + scene.target.y * layout.scale;
    targetRadius = scene.target.r * layout.scale;
    responsive = clamp((viewportWidth - 360) / 760, 0, 1);
    titleVizClearance = clamp(viz.clientWidth * 0.014, 10, 18);
    top = targetY + Math.max(70, targetRadius + 54) + 22 * responsive + titleVizClearance;
    titleRect = titleTextRect(scene);
    titleHeight = titleRect && titleRect.height > 0 ? titleRect.height : scene.title.offsetHeight || 124;
    titleMin = isTouchLandscape ? clamp(stageHeight * 0.28, 120, 220) : clamp(stageHeight * 0.34, 220, 292);
    fallbackFooterTop = stageHeight - clamp(stageHeight * 0.16, 128, 160);
    footerRect = footer ? footer.getBoundingClientRect() : null;
    footerTop = footerRect && footerRect.height > 0 ? footerRect.top - stageRect.top : fallbackFooterTop;
    footerGap = clamp(stageHeight * 0.04, 24, 42);
    titleFooterReserve = shortPhoneTitleFooterReserve(viewportWidth, stageHeight, footerGap, isTouchLandscape);
    maxTop = Math.max(18, footerTop - footerGap - titleHeight - titleFooterReserve);

    if (scene.panel) {
      panelRect = scene.panel.getBoundingClientRect();
      headingRect = scene.titleHeading ? scene.titleHeading.getBoundingClientRect() : scene.title.getBoundingClientRect();
      overlapsPanelX = Math.min(panelRect.right, headingRect.right) - Math.max(panelRect.left, headingRect.left) > 8;
      panelClearance = clamp(viz.clientWidth * 0.03, 18, 30);

      if (overlapsPanelX) {
        top = Math.max(top, panelRect.bottom - stageRect.top + panelClearance);
        maxTop = Math.min(maxTop, stageHeight - titleHeight - panelClearance);
      }
    }

    scene.title.style.setProperty('--hero-title-top', Math.round(clamp(top, titleMin, maxTop)) + 'px');
  }

  function buildViz(viz, svg) {
    var layout = setResponsiveTopology(viz, svg);
    Array.prototype.slice.call(svg.children).forEach(function (child) {
      if (child.tagName.toLowerCase() !== 'defs') child.remove();
    });
    var defs = svg.querySelector('defs') || el('defs', {}, svg);
    var meshGridPattern = defs.querySelector('#mesh-grid');
    if (meshGridPattern) meshGridPattern.setAttribute('patternTransform', 'translate(18 18)');

    var bg = el('g', { class: 'mesh-bg' }, svg);
    var bgPlane = el('rect', { class: 'mesh-grid-plane', width: roundSvg(layout.width), height: roundSvg(layout.height), fill: 'url(#mesh-grid)' }, bg);
    var bgBlue = el('circle', { cx: roundSvg(layout.bg.blue.x), cy: roundSvg(layout.bg.blue.y), r: roundSvg(layout.bg.blue.r), fill: 'url(#mesh-blue-field)' }, bg);
    var bgGreen = el('circle', { cx: roundSvg(layout.bg.green.x), cy: roundSvg(layout.bg.green.y), r: roundSvg(layout.bg.green.r), fill: 'url(#mesh-green-field)' }, bg);

    var edgeGroup = el('g', { class: 'mesh-edges', fill: 'none', 'stroke-linecap': 'round' }, svg);
    edges.forEach(function (edge) {
      var solidPath = el('path', {
        class: 'mesh-edge-trace',
        d: edge.d,
        stroke: edge.color,
        'stroke-width': 1.35,
        'stroke-opacity': edge.opacity,
      }, edgeGroup);
      var dashPath = el('path', {
        class: 'mesh-edge',
        d: edge.d,
        stroke: edge.color,
        'stroke-width': 1.35,
        'stroke-opacity': edge.opacity,
        opacity: 0,
      }, edgeGroup);
      edge.solidPath = solidPath;
      edge.path = dashPath;
      edge.length = cachePathLength(dashPath);
      cachePathLength(solidPath, edge.length);
      solidPath.style.strokeDasharray = edge.length;
      solidPath.style.strokeDashoffset = edge.length;
      dashPath.style.strokeDasharray = EDGE_DASH;
      dashPath.style.strokeDashoffset = 0;
      dashPath.style.opacity = 0;
    });

    var panelConnectorGroup = el('g', { class: 'panel-connectors', fill: 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, svg);
    var panelAngle = el('path', { class: 'panel-connector panel-connector-angle', d: 'M 0 0 L 0 0', stroke: '#86efac', 'stroke-width': 1, 'stroke-opacity': 0.56, opacity: 0 }, panelConnectorGroup);
    var panelStraight = el('path', { class: 'panel-connector panel-connector-straight', d: 'M 0 0 L 0 0', stroke: '#86efac', 'stroke-width': 1, 'stroke-opacity': 0.48, opacity: 0 }, panelConnectorGroup);
    var packetGroup = el('g', { class: 'mesh-packets' }, svg);
    var nodeGroup = el('g', { class: 'mesh-nodes' }, svg);

    nodes.forEach(function (item) {
      var glowId = ensureGlow(defs, item);
      var group = el('g', { class: 'mesh-node', 'data-node-id': item.id, transform: 'translate(' + item.x + ' ' + item.y + ')', opacity: 0 }, nodeGroup);
      var halo = el('circle', { class: 'node-halo', r: item.r, fill: 'url(#' + glowId + ')', opacity: 1 }, group);
      var plate = el('circle', { class: 'node-plate', r: item.r * NODE_PLATE_RATIO, fill: '#0e1014', stroke: item.color, 'stroke-opacity': 0.72, 'stroke-width': 1.2 }, group);
      var iconScale = iconScaleFor(item);
      var icon = el('g', { class: 'node-icon', transform: 'scale(' + iconScale + ')', fill: 'none', stroke: item.color, 'stroke-width': 1.8, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, group);
      drawLucideIcon(item.icon, icon);
      var label = el('text', { class: 'node-label', y: item.r * 0.92, 'text-anchor': 'middle' }, group);
      label.textContent = item.label;
      var type = el('text', { class: 'node-type', y: item.r * 0.92 + 13, 'text-anchor': 'middle' }, group);
      type.textContent = item.type;
      item.group = group;
      item.halo = halo;
      item.plate = plate;
      item.icon = icon;
      item.labelEl = label;
      item.typeEl = type;
    });

    var nodeTraceGroup = el('g', { class: 'node-border-traces', fill: 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, svg);
    var nodeTraceCw = el('path', { class: 'node-border-trace node-border-trace-cw', d: 'M 0 0 L 0 0', stroke: '#86efac', 'stroke-width': 1.35, 'stroke-opacity': 0.82, opacity: 0 }, nodeTraceGroup);
    var nodeTraceCcw = el('path', { class: 'node-border-trace node-border-trace-ccw', d: 'M 0 0 L 0 0', stroke: '#86efac', 'stroke-width': 1.35, 'stroke-opacity': 0.82, opacity: 0 }, nodeTraceGroup);

    var primary = nodeById('workstation');
    var target = nodeById('server');
    var focus = el('circle', { class: 'primary-ping', cx: primary.x, cy: primary.y, r: roundSvg(primary.r * 0.7), fill: 'none', stroke: primary.color, 'stroke-width': 1.2, opacity: 0 }, svg);
    var request = el('circle', { class: 'inference-packet request-packet', cx: primary.x, cy: primary.y, r: 5, fill: primary.color, opacity: 0 }, packetGroup);
    var work = el('circle', { class: 'inference-packet work-packet', r: 5.5, fill: primary.color, opacity: 0 }, packetGroup);
    var returnPackets = [];

    routes.forEach(function (route) {
      var edge = edges.filter(function (item) { return item.id === route.edge; })[0];
      route.path = edge.path;
      route.packet = el('circle', { class: 'ambient-packet', 'data-route-edge': route.edge, r: 3.5, fill: route.source, opacity: 0 }, packetGroup);
    });

    var tokenCount = viz.querySelectorAll('.stream-token').length;
    for (var i = 0; i < tokenCount; i += 1) {
      returnPackets.push(el('circle', { class: 'return-packet', r: 3.8, cx: target.x, cy: target.y, fill: target.color, opacity: 0 }, packetGroup));
    }

    var scene = {
      bg: bg,
      bgPlane: bgPlane,
      bgBlue: bgBlue,
      bgGreen: bgGreen,
      focus: focus,
      request: request,
      work: work,
      returnPackets: returnPackets,
      nodeTraceCw: nodeTraceCw,
      nodeTraceCcw: nodeTraceCcw,
      panelTraces: [nodeTraceCw, nodeTraceCcw],
      panelAngle: panelAngle,
      panelStraight: panelStraight,
      panelConnectors: [panelAngle, panelStraight],
      primary: primary,
      target: target,
      chrome: document.querySelectorAll('nav.top, .hero-footer'),
      title: document.querySelector('.hero-title-loop'),
      titleHeading: document.querySelector('.hero-title-heading'),
      subtitle: document.querySelector('.hero-title-subtitle'),
      subtitleCopy: document.querySelector('.hero-subtitle-copy'),
      subtitleAnimated: document.querySelector('.hero-subtitle-animated'),
      subtitlePrefix: document.querySelector('.hero-subtitle-prefix'),
      subtitleDynamic: document.querySelector('.hero-subtitle-dynamic'),
      subtitleDot: document.querySelector('.hero-subtitle-dot'),
      panel: viz.querySelector('.hero-inference'),
      tokens: viz.querySelectorAll('.stream-token'),
    };

    setResponsiveTopology(viz, svg, scene);
    return scene;
  }

  function setPacketOnPath(packet, path, progress, source, target) {
    var point = path.getPointAtLength(getPathLength(path) * progress);
    packet.setAttribute('cx', point.x);
    packet.setAttribute('cy', point.y);
    packet.setAttribute('fill', mixHex(source, target, progress));
  }

  function setReturnPacket(packet, path, progress) {
    setPacketOnPath(packet, path, progress, '#60a5fa', '#22c55e');
  }

  function releaseIntroLock() {
    document.body.classList.add('is-hero-complete');
    document.body.classList.remove('home-intro');
  }

  function splitTextChars(element) {
    if (!element) return [];
    if (element._meshSplit && element._meshSplit.revert) element._meshSplit.revert();
    element._meshSplit = null;

    if (typeof anime === 'undefined') return [];
    var splitter = (anime.text && (anime.text.splitText || anime.text.split)) || anime.splitText || anime.split;
    if (!splitter) return [];

    var split = splitter(element, {
      chars: { wrap: 'clip' },
      includeSpaces: true,
      accessible: false,
    });
    element._meshSplit = split;
    var chars = Array.prototype.slice.call(split.chars || []);
    chars.forEach(function (char) {
      char.classList.add('hero-title-char');
      char.style.width = '';
      char.style.transformOrigin = '50% 50%';
    });
    return chars;
  }

  function setText(element, text) {
    if (!element) return;
    if (element._meshSplit && element._meshSplit.revert) element._meshSplit.revert();
    element._meshSplit = null;
    element.textContent = text || '';
  }

  function setCharsTyped(chars) {
    chars.forEach(function (char) {
      char.style.width = '';
      char.style.opacity = 1;
      char.style.transform = 'translateX(0) scaleX(1)';
    });
  }

  function primeSubtitleChars(chars) {
    chars.forEach(function (char) {
      char.style.width = '';
      char.style.opacity = 0;
      char.style.transformOrigin = '0% 95%';
      char.style.transform = 'translateX(10px) scaleX(0)';
    });
  }

  function getStagger(interval, options) {
    var stagger = typeof anime !== 'undefined' && (anime.stagger || (anime.utils && anime.utils.stagger));
    if (stagger) return stagger(interval, options || {});

    options = options || {};
    return function (_el, i, length) {
      var start = typeof options.start === 'number' ? options.start : 0;
      var index = options.from === 'last' ? length - i - 1 : i;
      return start + index * interval;
    };
  }

  function animateSubtitleCharsIn(animate, chars, onComplete, onUpdate) {
    if (!chars.length) {
      if (onComplete) onComplete();
      return;
    }

    var completed = false;
    var startedAt = window.performance ? window.performance.now() : Date.now();
    var totalDuration = SUBTITLE_CHAR_IN_START + SUBTITLE_CHAR_IN_DURATION + Math.max(0, chars.length - 1) * SUBTITLE_CHAR_STAGGER;
    primeSubtitleChars(chars);
    if (onUpdate) onUpdate(true, 0);
    animate(chars, {
      opacity: [0, 1],
      scaleX: [0, 1],
      translateX: [10, 0],
      duration: SUBTITLE_CHAR_IN_DURATION,
      delay: getStagger(SUBTITLE_CHAR_STAGGER, { from: 'first', ease: 'in(3)', start: SUBTITLE_CHAR_IN_START }),
      onUpdate: function () {
        var now = window.performance ? window.performance.now() : Date.now();
        if (onUpdate) onUpdate(false, clampUnit((now - startedAt) / totalDuration));
      },
      onComplete: function () {
        if (completed) return;
        completed = true;
        setCharsTyped(chars);
        if (onUpdate) onUpdate(true, 1);
        if (onComplete) onComplete();
      },
    });
  }

  function animateSubtitleCharsOut(animate, chars, onComplete) {
    if (!chars.length) {
      if (onComplete) onComplete();
      return;
    }

    var completed = false;
    animate(chars, {
      opacity: 0,
      scaleX: 0,
      duration: 100,
      delay: getStagger(SUBTITLE_CHAR_STAGGER, { from: 'last', ease: 'in(3)' }),
      onComplete: function () {
        if (completed) return;
        completed = true;
        if (onComplete) onComplete();
      },
    });
  }

  function subtitleChars(scene, scope) {
    if (!scene.subtitle) return [];
    var target = scope === 'dynamic' ? scene.subtitleDynamic : scene.subtitleCopy;
    return Array.prototype.slice.call(target.querySelectorAll('.hero-title-char'));
  }

  function splitSubtitleChars(scene, scope) {
    if (scope === 'dynamic') return scene.subtitleDynamic && scene.subtitleDynamic.textContent ? splitTextChars(scene.subtitleDynamic) : [];

    var prefixChars = scene.subtitlePrefix && scene.subtitlePrefix.textContent ? splitTextChars(scene.subtitlePrefix) : [];
    var dynamicChars = scene.subtitleDynamic && scene.subtitleDynamic.textContent ? splitTextChars(scene.subtitleDynamic) : [];
    return prefixChars.concat(dynamicChars);
  }

  function setSubtitleText(scene, prefix, dynamic) {
    setText(scene.subtitlePrefix, prefix || '');
    setText(scene.subtitleDynamic, dynamic || '');
    var prefixChars = prefix ? splitTextChars(scene.subtitlePrefix) : [];
    var dynamicChars = dynamic ? splitTextChars(scene.subtitleDynamic) : [];
    return {
      prefix: prefixChars,
      dynamic: dynamicChars,
      all: prefixChars.concat(dynamicChars),
    };
  }

  function subtitleRestColor(scene) {
    return scene.subtitle ? window.getComputedStyle(scene.subtitle).color : SUBTITLE_DOT_REST;
  }

  function textWidth(element) {
    return element ? element.getBoundingClientRect().width : 0;
  }

  function subtitleDotTravel(scene) {
    return textWidth(scene.subtitleDynamic) || textWidth(scene.subtitleAnimated) || textWidth(scene.subtitleCopy);
  }

  function clampUnit(value) {
    return Math.min(Math.max(value, 0), 1);
  }

  function dotRestLeft(scene) {
    if (!scene.subtitleDot || !scene.subtitleAnimated) return 0;
    return scene.subtitleAnimated.getBoundingClientRect().left + scene.subtitleDot.offsetLeft;
  }

  function clearSubtitleDotLayout(scene) {
    scene.subtitleDotLayout = null;
  }

  function measureSubtitleDotLayout(scene, chars) {
    if (!scene.subtitleDot || !chars.length) return null;

    var firstLeft = null;
    var edges = [];

    chars.forEach(function (char) {
      var rect = char.getBoundingClientRect();
      if (firstLeft === null) firstLeft = rect.left;
      edges.push(rect.right);
    });

    if (firstLeft === null) return null;

    scene.subtitleDotLayout = {
      chars: chars,
      edges: edges,
      firstLeft: firstLeft,
      restLeft: dotRestLeft(scene),
    };

    return scene.subtitleDotLayout;
  }

  function prepareSubtitleDotLayout(scene, chars) {
    clearSubtitleDotLayout(scene);
    return measureSubtitleDotLayout(scene, chars);
  }

  function prepareSubtitleDotIn(scene) {
    if (!scene.subtitleDot) return false;
    var shouldFadeIn = true;
    scene.subtitleDotHasAppeared = true;
    scene.subtitleDotX = null;
    clearSubtitleDotLayout(scene);
    scene.subtitleDot.textContent = '.';
    scene.subtitleDot.style.visibility = 'visible';
    scene.subtitleDot.style.opacity = shouldFadeIn ? 0 : 1;
    scene.subtitleDot.style.color = subtitleRestColor(scene);
    scene.subtitleDot.style.transformOrigin = '0% 0%';
    return shouldFadeIn;
  }

  function hideSubtitleDot(scene) {
    if (!scene.subtitleDot) return;
    scene.subtitleDot.textContent = '.';
    scene.subtitleDot.style.visibility = 'hidden';
    scene.subtitleDot.style.opacity = 0;
    scene.subtitleDot.style.color = '';
    scene.subtitleDot.style.transformOrigin = '';
    scene.subtitleDot.style.transform = '';
    scene.subtitleDotX = null;
    scene.subtitleDotHasAppeared = false;
    clearSubtitleDotLayout(scene);
  }

  function setSubtitleDotX(scene, targetX, immediate) {
    if (immediate || typeof scene.subtitleDotX !== 'number') {
      scene.subtitleDotX = targetX;
    } else {
      var delta = targetX - scene.subtitleDotX;
      scene.subtitleDotX += Math.abs(delta) <= SUBTITLE_DOT_SNAP ? delta : delta * SUBTITLE_DOT_LERP;
    }

    scene.subtitleDot.style.transform = 'translateX(' + (Math.round(scene.subtitleDotX * 100) / 100) + 'px)';
  }

  function syncSubtitleDotToChars(scene, chars, shouldFadeIn, immediate, progress) {
    if (!scene.subtitleDot || !chars.length) return;

    var layout = scene.subtitleDotLayout && scene.subtitleDotLayout.chars === chars ? scene.subtitleDotLayout : measureSubtitleDotLayout(scene, chars);
    var visibleRight = null;
    var visibleCount;
    var edge;
    var x;

    if (!layout) return;

    progress = typeof progress === 'number' ? clampUnit(progress) : 1;
    visibleCount = Math.min(layout.edges.length, Math.max(0, Math.ceil(progress * layout.edges.length)));
    visibleRight = visibleCount ? layout.edges[visibleCount - 1] : null;
    edge = visibleRight === null ? layout.firstLeft : visibleRight;
    x = Math.round((edge - layout.restLeft) * 100) / 100;
    setSubtitleDotX(scene, x, immediate || visibleRight === null);
    if (shouldFadeIn) scene.subtitleDot.style.opacity = clampUnit((edge - layout.firstLeft) / 28);
  }

  function setSubtitleDotResting(scene) {
    if (!scene.subtitleDot) return;
    scene.subtitleDot.textContent = '.';
    scene.subtitleDot.style.display = '';
    scene.subtitleDot.style.position = '';
    scene.subtitleDot.style.right = '';
    scene.subtitleDot.style.bottom = '';
    scene.subtitleDot.style.top = '';
    scene.subtitleDot.style.visibility = 'visible';
    scene.subtitleDot.style.opacity = 1;
    scene.subtitleDot.style.color = '';
    scene.subtitleDot.style.transformOrigin = '';
    scene.subtitleDot.style.transform = '';
    scene.subtitleDotX = null;
    clearSubtitleDotLayout(scene);
  }

  function animateSubtitleDotOut(scene, animate, width, charCount, onComplete) {
    if (!scene.subtitleDot) {
      if (onComplete) onComplete();
      return;
    }
    scene.subtitleDot.textContent = '.';
    scene.subtitleDot.style.visibility = 'visible';
    scene.subtitleDot.style.opacity = 1;
    scene.subtitleDotX = null;
    var completed = false;
    animate(scene.subtitleDot, {
      x: -width,
      scaleX: [4, 1],
      transformOrigin: ['100% 0%', '100% 0%'],
      color: SUBTITLE_DOT_HOT,
      duration: charCount * SUBTITLE_CHAR_STAGGER + 75,
      delay: 0,
      ease: 'out(3)',
      onComplete: function () {
        if (completed) return;
        completed = true;
        hideSubtitleDot(scene);
        if (onComplete) onComplete();
      },
    });
  }

  function revealHeading(scene, animate) {
    if (!scene.title || !scene.titleHeading) return;

    scene.title.classList.add('is-visible');
    animate(scene.title, {
      opacity: [0, 1],
      translateY: [10, 0],
      duration: 360,
      ease: 'out(3)',
    });

    var headingChars = splitTextChars(scene.titleHeading);
    headingChars.forEach(function (char) {
      char.classList.add('hero-title-char');
      char.style.opacity = 0;
      char.style.transform = 'translateY(0.72em)';
    });
    animate(headingChars, {
      opacity: [0, 1],
      translateY: ['0.72em', '0em'],
      duration: 540,
      delay: function (_el, i) { return i * 13; },
      ease: 'out(4)',
    });
  }

  function revealSubtitleLine(scene, animate, prefix, dynamic, onComplete) {
    if (scene.subtitle) scene.subtitle.classList.add('is-visible');
    var parts = setSubtitleText(scene, prefix, dynamic);
    var shouldFadeIn = prepareSubtitleDotIn(scene);
    prepareSubtitleDotLayout(scene, parts.all);
    var syncDot = function (immediate, progress) { syncSubtitleDotToChars(scene, parts.all, shouldFadeIn, immediate, progress); };
    animateSubtitleCharsIn(animate, parts.all, function () {
      setSubtitleDotResting(scene);
      if (onComplete) onComplete();
    }, syncDot);
  }

  function eraseSubtitleLine(scene, animate, onComplete) {
    var chars = subtitleChars(scene);
    if (!chars.length) chars = splitSubtitleChars(scene);
    animateSubtitleDotOut(scene, animate, subtitleDotTravel(scene), subtitleChars(scene, 'dynamic').length || chars.length, function () {
      if (onComplete) onComplete(chars.length);
    });
    animateSubtitleCharsOut(animate, chars);
  }

  function eraseSubtitleDynamic(scene, animate, onComplete) {
    var chars = subtitleChars(scene, 'dynamic');
    if (!chars.length) chars = splitSubtitleChars(scene, 'dynamic');
    animateSubtitleDotOut(scene, animate, subtitleDotTravel(scene), chars.length, function () {
      if (onComplete) onComplete(chars.length);
    });
    animateSubtitleCharsOut(animate, chars);
  }

  function revealSubtitleDynamic(scene, animate, text, onComplete) {
    setText(scene.subtitleDynamic, text);
    var chars = splitTextChars(scene.subtitleDynamic);
    var shouldFadeIn = prepareSubtitleDotIn(scene);
    prepareSubtitleDotLayout(scene, chars);
    var syncDot = function (immediate, progress) { syncSubtitleDotToChars(scene, chars, shouldFadeIn, immediate, progress); };
    animateSubtitleCharsIn(animate, chars, function () {
      setSubtitleDotResting(scene);
      if (onComplete) onComplete();
    }, syncDot);
  }

  function setSubtitleFinal(scene) {
    var parts = setSubtitleText(scene, TITLE_FINAL_PREFIX, TITLE_PHRASES[0]);
    setCharsTyped(parts.all);
    setSubtitleDotResting(scene);
    if (scene.subtitle) scene.subtitle.classList.add('is-visible');
    scene.subtitleLoopIndex = 0;
  }

  function scheduleTitleStep(callback, delay) {
    window.setTimeout(callback, delay);
  }

  function subtitleLoopBaseActive(scene) {
    var rect;
    var titleStyle;
    var subtitleStyle;
    var section;

    if (!scene || !scene.title || !scene.subtitle || reduceMotion || document.hidden) return false;
    if (!scene.titleLoopStarted || scene.subtitleLoopAnimating || scene.subtitleLoopMeasuring) return false;
    if (scene.subtitleLoopMotionActive === false) return false;

    section = scene.title.closest('[data-stage2-scroll]');
    if (section && section.classList.contains('is-stage2-final-hook-visible')) return false;

    rect = scene.title.getBoundingClientRect();
    if (rect.bottom <= 0 || rect.top >= window.innerHeight || rect.right <= 0 || rect.left >= window.innerWidth) return false;

    titleStyle = window.getComputedStyle(scene.title);
    subtitleStyle = window.getComputedStyle(scene.subtitle);
    return Number.parseFloat(titleStyle.opacity || '1') > 0.08 && Number.parseFloat(subtitleStyle.opacity || '1') > 0.08;
  }

  function subtitleLoopPerfDelay(scene) {
    var now = window.performance ? window.performance.now() : Date.now();
    var blockedUntil = scene.subtitleLoopPerfBlockedUntil || 0;
    return Math.max(0, blockedUntil - now);
  }

  function clearSubtitleLoopTimer(scene) {
    if (!scene || !scene.subtitleLoopTimer) return;
    window.clearTimeout(scene.subtitleLoopTimer);
    scene.subtitleLoopTimer = null;
  }

  function scheduleSubtitleLoop(scene, animate, delay) {
    if (!scene || scene.subtitleLoopTimer || scene.subtitleLoopAnimating || scene.subtitleLoopMeasuring) return;
    if (!subtitleLoopBaseActive(scene)) return;

    delay = Math.max(0, delay || SUBTITLE_LOOP_DELAY);
    scene.subtitleLoopTimer = window.setTimeout(function () {
      scene.subtitleLoopTimer = null;
      advanceSubtitleLoop(scene, animate);
    }, delay);
  }

  function syncSubtitleLoop(scene, animate, delay) {
    var perfDelay;

    if (!subtitleLoopBaseActive(scene)) {
      clearSubtitleLoopTimer(scene);
      return;
    }

    perfDelay = subtitleLoopPerfDelay(scene);
    scheduleSubtitleLoop(scene, animate, perfDelay > 0 ? perfDelay + 80 : delay);
  }

  function requestSubtitleLoopSync(scene, animate, delay) {
    if (!scene || scene.subtitleLoopSyncRaf) return;

    scene.subtitleLoopSyncRaf = window.requestAnimationFrame(function () {
      scene.subtitleLoopSyncRaf = null;
      syncSubtitleLoop(scene, animate, delay || SUBTITLE_LOOP_RETRY_DELAY);
    });
  }

  function measureSubtitleLoopFrame(scene, onComplete) {
    var firstFrame;

    if (!window.requestAnimationFrame) {
      onComplete(true);
      return;
    }

    scene.subtitleLoopMeasuring = true;
    window.requestAnimationFrame(function (first) {
      firstFrame = first;
      window.requestAnimationFrame(function (second) {
        var delta = second - firstFrame;
        var now = window.performance ? window.performance.now() : Date.now();
        scene.subtitleLoopMeasuring = false;

        if (delta > SUBTITLE_LOOP_FRAME_BUDGET) {
          scene.subtitleLoopPerfBlockedUntil = now + SUBTITLE_LOOP_PERF_COOLDOWN;
          onComplete(false);
          return;
        }

        onComplete(true);
      });
    });
  }

  function rotateSubtitleLoop(scene, animate) {
    var nextIndex;

    if (!subtitleLoopBaseActive(scene)) return;

    nextIndex = ((scene.subtitleLoopIndex || 0) + 1) % TITLE_PHRASES.length;
    scene.subtitleLoopAnimating = true;

    eraseSubtitleDynamic(scene, animate, function () {
      revealSubtitleDynamic(scene, animate, TITLE_PHRASES[nextIndex], function () {
        scene.subtitleLoopIndex = nextIndex;
        scene.subtitleLoopAnimating = false;
        syncSubtitleLoop(scene, animate, SUBTITLE_LOOP_DELAY);
      });
    });
  }

  function advanceSubtitleLoop(scene, animate) {
    var perfDelay;

    if (!subtitleLoopBaseActive(scene)) return;

    perfDelay = subtitleLoopPerfDelay(scene);
    if (perfDelay > 0) {
      scheduleSubtitleLoop(scene, animate, perfDelay + 80);
      return;
    }

    measureSubtitleLoopFrame(scene, function (allowed) {
      if (!allowed) {
        syncSubtitleLoop(scene, animate, SUBTITLE_LOOP_PERF_COOLDOWN);
        return;
      }

      rotateSubtitleLoop(scene, animate);
    });
  }

  function installSubtitleLoopGates(scene, animate) {
    if (scene.subtitleLoopGatesInstalled) return;
    scene.subtitleLoopGatesInstalled = true;
    scene.subtitleLoopMotionActive = true;

    window.addEventListener(SVG_MOTION_EVENT, function (event) {
      scene.subtitleLoopMotionActive = !event.detail || event.detail.active !== false;
      syncSubtitleLoop(scene, animate, SUBTITLE_LOOP_RETRY_DELAY);
    });

    document.addEventListener('visibilitychange', function () {
      syncSubtitleLoop(scene, animate, SUBTITLE_LOOP_RETRY_DELAY);
    });

    window.addEventListener('scroll', function () {
      requestSubtitleLoopSync(scene, animate);
    }, { passive: true });
    window.addEventListener('resize', function () {
      requestSubtitleLoopSync(scene, animate);
    });
    window.addEventListener('pageshow', function () {
      requestSubtitleLoopSync(scene, animate, SUBTITLE_LOOP_RETRY_DELAY);
    });
  }

  function startTitleHeadingSequence(scene, animate) {
    if (!scene.title || scene.titleHeadingStarted) return;
    scene.titleHeadingStarted = true;

    revealHeading(scene, animate);
  }

  function startSubtitleIntroSequence(scene, animate) {
    if (!scene.title || scene.subtitleIntroStarted) return;
    scene.subtitleIntroStarted = true;

    scheduleTitleStep(function () {
      revealSubtitleLine(scene, animate, '', 'Run large models', function () {
        scheduleTitleStep(function () {
          eraseSubtitleLine(scene, animate, function () {
            scheduleTitleStep(function () {
              revealSubtitleLine(scene, animate, '', 'Run diverse models', function () {
                scheduleTitleStep(function () {
                  eraseSubtitleLine(scene, animate, function () {
                    scheduleTitleStep(function () {
                      revealSubtitleLine(scene, animate, TITLE_FINAL_PREFIX, TITLE_PHRASES[0], function () {
                        startTitleLoop(scene, animate);
                      });
                    }, 260);
                  });
                }, 720);
              });
            }, 260);
          });
        }, 720);
      });
    }, 520);
  }

  function startTitleLoop(scene, animate) {
    if (!scene.title || reduceMotion || scene.titleLoopStarted) return;

    scene.titleLoopStarted = true;
    scene.title.classList.add('is-visible', 'is-settled');
    setSubtitleFinal(scene);
    installSubtitleLoopGates(scene, animate);
    syncSubtitleLoop(scene, animate, SUBTITLE_LOOP_DELAY);
  }

  function activateFinalState(scene) {
    scene.bg.setAttribute('opacity', 1);
    nodes.forEach(function (node) {
      node.group.setAttribute('opacity', 1);
      node.group.setAttribute('transform', 'translate(' + node.x + ' ' + node.y + ')');
    });
    edges.forEach(function (edge) {
      activateEdge(edge);
    });
    scene.primary.halo.setAttribute('opacity', 0.18);
    scene.primary.halo.setAttribute('r', scene.primary.r);
    scene.target.halo.setAttribute('opacity', 0.24);
    scene.target.halo.setAttribute('r', scene.target.r * 1.08);
    scene.focus.setAttribute('opacity', 0);
    scene.request.setAttribute('opacity', 0);
    scene.work.setAttribute('opacity', 0);
    scene.returnPackets.forEach(function (packet) {
      packet.setAttribute('opacity', 0);
    });
    routes.forEach(function (route) {
      if (route.packet) route.packet.setAttribute('opacity', 0);
    });
    scene.target.group.classList.add('is-processing');
    scene.panel.classList.add('is-visible', 'is-collapsing', 'is-collapsed');
    scene.panel.style.opacity = 0;
    scene.panel.style.clipPath = 'inset(49% 0% 49% 0%)';
    scene.panel.style.transform = 'scaleY(0.025)';
    for (var i = 0; i < scene.tokens.length; i += 1) scene.tokens[i].classList.add('is-visible');
    scene.panelConnectors.forEach(function (connector) {
      connector.setAttribute('opacity', 0);
      connector.style.strokeDasharray = '0px, 1010px';
      connector.style.strokeDashoffset = '-1000px';
    });
    scene.panelTraces.forEach(function (trace) {
      trace.setAttribute('opacity', 0);
      trace.style.strokeDasharray = '0px, 1010px';
      trace.style.strokeDashoffset = '-1000px';
    });
    if (scene.title) {
      if (scene.titleHeading) setCharsTyped(splitTextChars(scene.titleHeading));
      setSubtitleFinal(scene);
      scene.title.classList.add('is-visible', 'is-settled');
      scene.title.style.opacity = 1;
      scene.title.style.transform = 'translateY(0)';
    }
    releaseIntroLock();
    scene.chrome.forEach(function (item) {
      item.style.opacity = 1;
      item.style.transform = 'translateY(0)';
    });
  }

  function quietAmbient(scene) {
    routes.forEach(function (route) {
      if (route.packet) route.packet.setAttribute('opacity', 0);
    });
    if (scene.target && scene.target.halo) {
      scene.target.halo.setAttribute('r', scene.target.r * 1.08);
      scene.target.halo.setAttribute('opacity', 0.24);
    }
  }

  function pauseAnimation(animation) {
    if (animation && typeof animation.pause === 'function') animation.pause();
  }

  function resumeAnimation(animation) {
    if (animation && typeof animation.resume === 'function') {
      animation.resume();
    } else if (animation && typeof animation.play === 'function') {
      animation.play();
    }
  }

  function createAmbientAnimations(scene, animate) {
    var animations = [];

    animations.push(animate(edges.map(function (edge) { return edge.path; }), {
      strokeDashoffset: [0, -20],
      duration: 5400,
      loop: true,
      ease: 'linear',
      autoplay: false,
    }));

    animations.push(animate(scene.target.halo, {
      r: [scene.target.r, scene.target.r * 1.26, scene.target.r],
      opacity: [0.2, 0.34, 0.2],
      duration: 2400,
      loop: true,
      ease: 'inOutQuad',
      autoplay: false,
    }));

    routes.forEach(function (route) {
      var state = { p: 0 };
      animations.push(animate(state, {
        p: [0, 1],
        duration: route.duration,
        delay: route.delay,
        loopDelay: 3200,
        loop: true,
        ease: 'inOut(3)',
        autoplay: false,
        onUpdate: function () {
          var opacity = state.p < 0.14 ? state.p / 0.14 : state.p > 0.86 ? (1 - state.p) / 0.14 : 0.66;
          route.packet.setAttribute('opacity', Math.max(0, opacity));
          setPacketOnPath(route.packet, route.path, state.p, route.source, route.target);
        },
      }));
    });

    return animations;
  }

  function createSvgMotionSwitch(viz, scene, animate) {
    var animations = null;
    var ready = false;
    var requestedActive = true;
    var active = false;

    function ensureAnimations() {
      if (!animations) animations = createAmbientAnimations(scene, animate);
    }

    function setMotionActive(nextActive) {
      nextActive = Boolean(nextActive);
      if (active === nextActive) return;

      active = nextActive;
      viz.classList.toggle(SVG_MOTION_CLASS, active);

      if (active) {
        ensureAnimations();
        animations.forEach(resumeAnimation);
      } else {
        if (animations) animations.forEach(pauseAnimation);
        quietAmbient(scene);
      }
    }

    function sync() {
      if (!ready) {
        active = false;
        viz.classList.remove(SVG_MOTION_CLASS);
        return;
      }
      setMotionActive(ready && requestedActive && !reduceMotion);
    }

    window.addEventListener(SVG_MOTION_EVENT, function (event) {
      requestedActive = !event.detail || event.detail.active !== false;
      sync();
    });

    return {
      ready: function () {
        ready = true;
        sync();
      },
      stop: function () {
        requestedActive = false;
        sync();
      },
    };
  }

  function addReturnPacket(tl, packet, path, position) {
    if (!packet || !path) return;

    var state = { p: 1, r: 3.4 };
    tl.add(state, {
      p: [1, 0],
      r: [3.2, 4.8, 3.4],
      duration: RETURN_PACKET_DURATION,
      ease: 'out(4)',
      onBegin: function () {
        packet.setAttribute('opacity', 0);
        setReturnPacket(packet, path, 1);
      },
      onUpdate: function () {
        var traveled = 1 - state.p;
        var opacity = traveled < 0.16 ? traveled / 0.16 : traveled > 0.84 ? (1 - traveled) / 0.16 : 0.95;
        packet.setAttribute('opacity', Math.max(0, opacity));
        packet.setAttribute('r', state.r);
        setReturnPacket(packet, path, state.p);
      },
      onComplete: function () {
        packet.setAttribute('opacity', 0);
      },
    }, position);
  }

  function createDrawables(paths) {
    var createDrawable = anime.svg && anime.svg.createDrawable ? anime.svg.createDrawable : anime.createDrawable;
    if (!createDrawable) return [];

    return paths.map(function (path) {
      var drawable = createDrawable(path)[0];
      drawable.draw = '0 0';
      return drawable;
    });
  }

  function createConnectorDrawables(scene) {
    return createDrawables(scene.panelConnectors);
  }

  function createTraceDrawables(scene) {
    return createDrawables(scene.panelTraces);
  }

  function addPanelCollapse(tl, scene, drawables, traceDrawables, position) {
    tl.add(scene.panelConnectors.concat(scene.panelTraces), {
      opacity: [1, 0],
      duration: 340,
      ease: 'out(3)',
    }, position - 80);

    if (drawables.length) {
      tl.add(drawables, {
        draw: ['0 1', '1 1'],
        duration: 340,
        ease: 'inOut(3)',
      }, position - 80);
    }

    if (traceDrawables.length) {
      tl.add(traceDrawables, {
        draw: ['0 1', '1 1'],
        duration: 340,
        ease: 'inOut(3)',
      }, position - 80);
    }

    tl.add(scene.panel, {
      clipPath: ['inset(0% 0% 0% 0%)', 'inset(49% 0% 49% 0%)'],
      scaleY: [1, 0.025],
      opacity: [1, 0],
      duration: 520,
      ease: 'inOut(3)',
      onBegin: function () {
        scene.panel.classList.add('is-collapsing');
      },
      onComplete: function () {
        scene.panel.classList.add('is-collapsed');
      },
    }, position);
  }

  function runSequence(viz, scene) {
    var animate = anime.animate;
    var createTimeline = anime.createTimeline;
    var connectorDrawables = createConnectorDrawables(scene);
    var traceDrawables = createTraceDrawables(scene);
    var connectorStart = RIM_TRACE_START + RIM_TRACE_DURATION;
    var straightStart = connectorStart + 300;
    var panelRevealStart = connectorStart + 220;
    var panelCollapseStart = TOKEN_START + (scene.tokens.length - 1) * TOKEN_STAGGER + Math.max(TOKEN_DURATION, RETURN_PACKET_DURATION) + PANEL_COLLAPSE_DELAY;
    var stableStateStart = panelCollapseStart + 620;
    var stableStarted = false;
    var svgMotion = createSvgMotionSwitch(viz, scene, animate);
    function beginStableState() {
      if (stableStarted) return;
      stableStarted = true;
      svgMotion.ready();
      startSubtitleIntroSequence(scene, animate);
    }

    var tl = createTimeline({
      autoplay: false,
      defaults: { ease: 'outExpo' },
      onComplete: beginStableState,
    });

    function forceStableState() {
      if (tl && typeof tl.pause === 'function') tl.pause();
      activateFinalState(scene);
      beginStableState();
    }

    window.addEventListener('mesh:hero-viz:stable', forceStableState);

    tl.add(scene.bg, { opacity: [0, 1], duration: 900 }, 520);

    nodes.forEach(function (node, index) {
      tl.add(node.group, {
        opacity: [0, 1],
        duration: 460,
      }, 1600 + index * 180);
    });

    tl.call(function () {
      startTitleHeadingSequence(scene, animate);
    }, 3140);

    edges.forEach(function (edge, index) {
      var drawStart = 2200 + index * 185;
      var drawDuration = index > 3 ? 620 : 760;
      var fadeStart = drawStart + drawDuration - 40;
      var fadeDuration = 1120;

      tl.add(edge.solidPath, {
        strokeDashoffset: [edge.length, 0],
        duration: drawDuration,
        ease: 'out(3)',
      }, drawStart);

      tl.add(edge.path, {
        opacity: [0, 1],
        duration: fadeDuration,
        ease: 'inOut(2)',
        onBegin: function () {
          edge.path.style.strokeDasharray = EDGE_DASH;
          edge.path.style.strokeDashoffset = 0;
          edge.path.classList.add('is-active');
        },
      }, fadeStart);

      tl.add(edge.solidPath, {
        opacity: [1, 0],
        duration: fadeDuration,
        ease: 'inOut(2)',
      }, fadeStart);
    });

    tl.add(scene.focus, {
      opacity: [0, 0.7, 0],
      r: [scene.primary.r * 0.48, scene.primary.r * 2.24],
      duration: 1180,
      ease: 'outQuart',
    }, 4300);

    tl.add(scene.primary.halo, {
      opacity: [0.13, 0.34, 0.18],
      r: [scene.primary.r, scene.primary.r * 1.22, scene.primary.r],
      duration: 900,
    }, 4400);

    tl.add(scene.request, {
      opacity: [0, 1],
      r: [2, 5],
      duration: 360,
      ease: 'outQuart',
    }, 5300);

    var dispatch = { p: 0 };
    var dispatchEdge = edges.filter(function (edge) { return edge.id === 'e2'; })[0];
    tl.add(dispatch, {
      p: [0, 1],
      duration: 1650,
      ease: 'inOut(3)',
      onBegin: function () {
        scene.request.setAttribute('opacity', 0);
        scene.work.setAttribute('opacity', 1);
      },
      onUpdate: function () {
        setPacketOnPath(scene.work, dispatchEdge.path, dispatch.p, '#60a5fa', '#22c55e');
      },
      onComplete: function () {
        scene.work.setAttribute('opacity', 0);
        scene.target.group.classList.add('is-processing');
      },
    }, 5800);

    tl.add(scene.target.halo, {
      opacity: [0.13, 0.42, 0.24],
      r: [scene.target.r, scene.target.r * 1.34, scene.target.r * 1.08],
      duration: 820,
      ease: 'outQuart',
    }, RIM_TRACE_START);

    if (traceDrawables.length) {
      tl.add(scene.panelTraces, {
        opacity: [0, 1],
        duration: 160,
        ease: 'out(3)',
      }, RIM_TRACE_START);

      tl.add(traceDrawables, {
        draw: ['0 0', '0 1'],
        duration: RIM_TRACE_DURATION,
        ease: 'out(3)',
      }, RIM_TRACE_START);
    }

    if (connectorDrawables.length) {
      tl.add(scene.panelAngle, {
        opacity: [1, 1],
        duration: 1,
        ease: 'out(3)',
      }, connectorStart);

      tl.add(connectorDrawables[0], {
        draw: ['0 0', '0 1'],
        duration: 390,
        ease: 'out(3)',
      }, connectorStart);

      tl.add(scene.panelStraight, {
        opacity: [0, 1],
        duration: 180,
        ease: 'out(3)',
      }, straightStart);

      tl.add(connectorDrawables[1], {
        draw: ['0 0', '0 1'],
        duration: 360,
        ease: 'out(3)',
      }, straightStart);
    }

    tl.add(scene.panel, {
      opacity: [0, 1],
      translateY: [10, 0],
      scaleY: [1, 1],
      clipPath: ['inset(0% 0% 0% 0%)', 'inset(0% 0% 0% 0%)'],
      duration: 520,
      onBegin: function () {
        scene.panel.classList.remove('is-collapsed', 'is-collapsing');
        scene.panel.classList.add('is-visible');
      },
    }, panelRevealStart);

    var returnEdge = edges.filter(function (edge) { return edge.id === 'e2'; })[0];
    Array.prototype.forEach.call(scene.tokens, function (token, index) {
      var tokenTime = TOKEN_START + index * TOKEN_STAGGER;
      tl.add(token, {
        opacity: [0, 1],
        translateY: [6, 0],
        duration: TOKEN_DURATION,
        ease: 'out(3)',
        onComplete: function () {
          token.classList.add('is-visible');
        },
      }, tokenTime);

      addReturnPacket(tl, scene.returnPackets[index], returnEdge.path, tokenTime);
    });

    addPanelCollapse(tl, scene, connectorDrawables, traceDrawables, panelCollapseStart);

    tl.add({ p: 0 }, {
      p: [0, 1],
      duration: 1,
      onBegin: beginStableState,
    }, stableStateStart);

    tl.add(scene.chrome, {
      opacity: [0, 1],
      translateY: [10, 0],
      duration: 760,
      delay: function (_el, i) { return i * 140; },
      onBegin: function () { document.body.classList.add('is-hero-complete'); },
      onComplete: releaseIntroLock,
    }, stableStateStart);

    tl.play();
  }

  function init() {
    if (!reduceMotion && typeof anime === 'undefined') {
      requestAnimationFrame(init);
      return;
    }

    var viz = document.querySelector('.hero-viz');
    if (!viz) return;
    var svg = viz.querySelector('svg');
    if (!svg) return;

    var scene = buildViz(viz, svg);
    positionPanel(viz, svg, scene);
    positionTitle(viz, svg, scene);
    var resizeRaf = null;
    var lastResizeWidth = window.innerWidth;
    window.addEventListener('resize', function () {
      var currentWidth = window.innerWidth;
      var widthChanged = currentWidth !== lastResizeWidth;
      lastResizeWidth = currentWidth;
      // Skip height-only resizes (iOS Safari tab bar collapse/expand)
      if (!widthChanged) return;
      if (resizeRaf) return;
      resizeRaf = window.requestAnimationFrame(function () {
        resizeRaf = null;
        setResponsiveTopology(viz, svg, scene);
        positionPanel(viz, svg, scene);
        positionTitle(viz, svg, scene);
      });
    });
    viz.classList.add('is-mounted');

    if (reduceMotion) {
      activateFinalState(scene);
      return;
    }

    runSequence(viz, scene);
  }

  function initTopo() {
    if (reduceMotion) return;

    if (typeof anime === 'undefined') {
      requestAnimationFrame(initTopo);
      return;
    }

    var topo = document.querySelector('.topo-stage');
    if (!topo) return;

    var animate = anime.animate;
    var edges = topo.querySelectorAll('.edge');
    if (edges.length) {
      animate(edges, {
        strokeDashoffset: [0, -10],
        duration: 800,
        loop: true,
        ease: 'linear',
      });
    }

    var rings = topo.querySelectorAll('.ring:not(.slow)');
    if (rings.length) {
      animate(rings, {
        scale: [0.7, 1.8],
        opacity: [0.35, 0],
        duration: 4000,
        loop: true,
        ease: 'out',
      });
    }

    var slowRings = topo.querySelectorAll('.ring.slow');
    if (slowRings.length) {
      animate(slowRings, {
        scale: [0.7, 1.8],
        opacity: [0.35, 0],
        duration: 6000,
        loop: true,
        ease: 'out',
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      init();
      initTopo();
    });
  } else {
    init();
    initTopo();
  }
})();
