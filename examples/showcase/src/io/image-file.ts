// Which files are images. Apart from height-image.ts, which reads them: every
// file dropped on the page is asked.

/** Whether a file is an image, by its type or else its extension. */
export function isImageFile(file: { name: string; type: string }): boolean {
  return file.type.startsWith("image/") || /\.(?:png|jpe?g|webp|bmp|gif|avif)$/i.test(file.name);
}
