export type TopSection = "dashboard" | "chat" | "playground";

export type AppRoute = {
  section: TopSection;
  chatId: string | null;
};

export function sectionFromPathname(pathname: string): TopSection | null {
  if (pathname === "/dashboard" || pathname === "/dashboard/") {
    return "dashboard";
  }
  if (
    pathname === "/chat" ||
    pathname === "/chat/" ||
    pathname.startsWith("/chat/")
  ) {
    return "chat";
  }
  if (pathname === "/playground" || pathname === "/playground/") {
    return "playground";
  }
  return null;
}

export function readRouteFromLocation(): AppRoute {
  if (typeof window === "undefined") {
    return { section: "dashboard", chatId: null };
  }

  const pathname = window.location.pathname;
  if (pathname === "/dashboard" || pathname === "/dashboard/") {
    return { section: "dashboard", chatId: null };
  }
  if (pathname === "/chat" || pathname === "/chat/") {
    return { section: "chat", chatId: null };
  }
  if (pathname.startsWith("/chat/")) {
    const raw = pathname.slice("/chat/".length);
    const chatId = raw ? decodeURIComponent(raw.split("/")[0]) : null;
    return { section: "chat", chatId };
  }
  if (pathname === "/playground" || pathname === "/playground/") {
    return { section: "playground", chatId: null };
  }

  return { section: "dashboard", chatId: null };
}

export function pathnameForRoute(route: AppRoute): string {
  if (route.section === "dashboard") {
    return "/dashboard";
  }
  if (route.section === "playground") {
    return "/playground";
  }
  return route.chatId ? `/chat/${encodeURIComponent(route.chatId)}` : "/chat";
}

export function pushRoute(route: AppRoute) {
  if (typeof window === "undefined") return;

  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath) {
    window.history.pushState({}, "", nextPath);
  }
}

export function replaceRoute(route: AppRoute) {
  if (typeof window === "undefined") return;

  const nextPath = pathnameForRoute(route);
  if (window.location.pathname !== nextPath) {
    window.history.replaceState({}, "", nextPath);
  }
}
