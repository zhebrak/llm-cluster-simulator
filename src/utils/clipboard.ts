/**
 * Copy text to clipboard with textarea fallback for non-HTTPS contexts.
 */
export function copyToClipboard(text: string): void {
  try {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
  } catch {
    navigator.clipboard?.writeText(text);
  }
}
