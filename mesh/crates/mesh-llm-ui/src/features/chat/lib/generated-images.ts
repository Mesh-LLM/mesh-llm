export const MAX_GENERATED_IMAGE_CHARS = 4 * 1024 * 1024

export function isGeneratedImageUrl(url: string): boolean {
  return (
    url.length <= MAX_GENERATED_IMAGE_CHARS && /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/i.test(url)
  )
}

export function hasGeneratedImage(text: string): boolean {
  return /data:image\/(?:png|jpeg|webp);base64,/i.test(text)
}

/** Avoid a wall of literal pixels until an inline image's Markdown destination closes. */
export function hideUnfinishedGeneratedImages(text: string): string {
  return text.replace(
    /!\[([^[\]\r\n]{0,512})\]\(<?data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]*={0,2}>?\s*$/gi,
    (_image, alt: string) => `[${alt || 'Generated image'}]`
  )
}

/** Keep captions in text history without sending generated pixels back to the model. */
export function summarizeGeneratedImages(text: string, suffix = ''): string {
  return text
    .replace(
      /!\[([^[\]\r\n]{0,512})\]\(data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}\)/gi,
      (_image, alt: string) => `[${alt || 'Generated image'}${suffix}]`
    )
    .replace(/data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]*={0,2}/gi, `[Generated image${suffix}]`)
}
