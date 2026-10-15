// The page's own elements carry this attribute, and the injected stylesheet
// applies only to them (see scope-css.ts). The page mounts inside the console's
// DOM, so an unscoped utility such as `.hidden` would also restyle the
// console's own elements.
export const SCOPE_ATTRIBUTE = 'data-capsules-ui'

let portalRoot: HTMLElement | null = null

/** Where the page's popovers, menus, tooltips and dialogs render: a container
 *  at the end of `document.body` carrying the scope attribute, so portaled
 *  content is styled like the rest of the page. */
export function portalContainer(): HTMLElement {
  if (!portalRoot?.isConnected) {
    portalRoot = document.createElement('div')
    portalRoot.setAttribute(SCOPE_ATTRIBUTE, '')
    document.body.appendChild(portalRoot)
  }
  return portalRoot
}
