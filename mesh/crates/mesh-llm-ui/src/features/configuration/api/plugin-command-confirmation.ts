/** Ask the user to review every executable and argument before approval. */
export function confirmPluginCommands(commands: string[]): Promise<boolean> {
  return new Promise((resolve) => {
    const dialog = document.createElement('dialog')
    dialog.setAttribute('aria-labelledby', 'plugin-command-confirmation-title')
    dialog.className =
      'rounded-lg border border-border bg-background p-6 text-foreground shadow-lg backdrop:bg-black/60'
    const title = document.createElement('h2')
    title.id = 'plugin-command-confirmation-title'
    title.className = 'text-lg font-semibold'
    title.textContent = 'Approve plugin commands?'
    const description = document.createElement('p')
    description.textContent =
      'These commands and arguments will run with the host service account. Review the complete list before approving.'
    const list = document.createElement('pre')
    list.className =
      'my-4 max-h-80 max-w-2xl overflow-auto whitespace-pre-wrap break-all rounded border border-border p-3'
    list.textContent = commands.join('\n')
    const actions = document.createElement('div')
    actions.className = 'flex justify-end gap-2'
    const cancel = document.createElement('button')
    cancel.className = 'rounded border border-input px-3 py-2'
    cancel.textContent = 'Cancel'
    const approve = document.createElement('button')
    approve.className = 'rounded border border-input px-3 py-2 font-semibold'
    approve.textContent = 'Approve'
    actions.append(cancel, approve)
    dialog.append(title, description, list, actions)
    document.body.append(dialog)
    const finish = (approved: boolean) => {
      dialog.close()
      dialog.remove()
      resolve(approved)
    }
    cancel.addEventListener('click', () => finish(false))
    approve.addEventListener('click', () => finish(true))
    dialog.addEventListener('cancel', (event) => {
      event.preventDefault()
      finish(false)
    })
    dialog.showModal()
    cancel.focus()
  })
}
