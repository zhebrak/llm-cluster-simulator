/**
 * ModalBackdrop — fixed full-screen overlay for modal dialogs.
 * Centralizes z-index, positioning, and optional backdrop click handler.
 * Inner panel div (with stopPropagation, sizing, borders) stays in each consumer.
 */

interface ModalBackdropProps {
  children: React.ReactNode;
  backdropClass?: string;        // default "bg-black/60"
  onBackdropClick?: () => void;
}

export function ModalBackdrop({ children, backdropClass = 'bg-black/60', onBackdropClick }: ModalBackdropProps) {
  return (
    <div
      className={`fixed inset-0 ${backdropClass} flex items-center justify-center z-50`}
      onClick={onBackdropClick}
    >
      {children}
    </div>
  );
}
