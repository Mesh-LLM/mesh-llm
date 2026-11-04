let credential: string | null = null
let pending: Promise<string> | null = null

/** Store the credential in memory only for this browser session. */
export function setOwnerControlCredential(value: string | null): void {
  credential = value?.trim() || null
}

export function getOwnerControlCredential(): Promise<string> {
  if (credential) return Promise.resolve(credential)
  if (pending) return pending

  pending = new Promise<string>((resolve, reject) => {
    const dialog = document.createElement('dialog')
    dialog.setAttribute('aria-labelledby', 'owner-control-credential-title')
    dialog.className =
      'rounded-lg border border-border bg-background p-6 text-foreground shadow-lg backdrop:bg-black/60'
    const form = document.createElement('form')
    form.className = 'flex max-w-lg flex-col gap-4'
    const title = document.createElement('h2')
    title.id = 'owner-control-credential-title'
    title.className = 'text-lg font-semibold'
    title.textContent = 'Unlock owner controls'
    const description = document.createElement('p')
    description.textContent =
      'Paste the owner-control token from the host’s owner-only token file. The default path is ~/.mesh-llm/owner-control-token-v2. It stays in this browser tab’s memory.'
    const label = document.createElement('label')
    label.className = 'flex flex-col gap-2'
    label.textContent = 'Owner-control token'
    const input = document.createElement('input')
    input.type = 'password'
    input.autocomplete = 'off'
    input.required = true
    input.className = 'rounded border border-input bg-background p-2'
    label.append(input)
    const actions = document.createElement('div')
    actions.className = 'flex justify-end gap-2'
    const cancel = document.createElement('button')
    cancel.type = 'button'
    cancel.className = 'rounded border border-input px-3 py-2'
    cancel.textContent = 'Cancel'
    const submit = document.createElement('button')
    submit.type = 'submit'
    submit.className = 'rounded border border-input px-3 py-2 font-semibold'
    submit.textContent = 'Unlock'
    actions.append(cancel, submit)
    form.append(title, description, label, actions)
    dialog.append(form)
    document.body.append(dialog)
    const finish = (value?: string) => {
      dialog.close()
      dialog.remove()
      if (value) {
        setOwnerControlCredential(value)
        resolve(value)
      } else {
        reject(new Error('Owner-control credential was not provided.'))
      }
    }
    cancel.addEventListener('click', () => finish())
    dialog.addEventListener('cancel', (event) => {
      event.preventDefault()
      finish()
    })
    form.addEventListener('submit', (event) => {
      event.preventDefault()
      const value = input.value.trim()
      if (value) finish(value)
    })
    dialog.showModal()
    input.focus()
  }).finally(() => {
    pending = null
  })
  return pending
}
