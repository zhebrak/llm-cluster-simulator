import { toBlob } from 'html-to-image';

function getThemeBackgroundColor(): string {
  const theme = document.documentElement.getAttribute('data-theme');
  return theme === 'light' ? '#f8f9fc' : '#111827';
}

async function captureAsBlob(el: HTMLElement): Promise<Blob> {
  const blob = await toBlob(el, {
    pixelRatio: 2,
    backgroundColor: getThemeBackgroundColor(),
  });
  if (!blob) throw new Error('Capture failed');
  return blob;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Copy PNG to clipboard. Returns true if clipboard succeeded, false if fell back to download. */
export async function copyImageToClipboard(el: HTMLElement): Promise<boolean> {
  // Pass a Promise<Blob> to ClipboardItem so clipboard.write() is called
  // synchronously within the user gesture. Chrome revokes clipboard access
  // if an await (like captureAsBlob) runs before write() — the gesture expires.
  // With a promise-valued ClipboardItem, write() is called immediately and
  // the blob is resolved lazily by the browser.
  try {
    const blobPromise = captureAsBlob(el);
    await navigator.clipboard.write([
      new ClipboardItem({ 'image/png': blobPromise }),
    ]);
    return true;
  } catch {
    // Non-secure context (HTTP) or unsupported browser — fall back to download
    try {
      const blob = await captureAsBlob(el);
      downloadBlob(blob, 'llm-cluster-simulator.png');
    } catch { /* capture failed entirely */ }
    return false;
  }
}

export async function downloadImage(el: HTMLElement, filename: string): Promise<void> {
  const blob = await captureAsBlob(el);
  downloadBlob(blob, filename);
}
