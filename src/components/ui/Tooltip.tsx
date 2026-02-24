import { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  text: string;
  children: React.ReactElement;
  className?: string;
}

export function Tooltip({ text, children, className }: TooltipProps) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const triggerRef = useRef<HTMLDivElement | null>(null);
  const tipRef = useRef<HTMLDivElement | null>(null);
  const isTouchRef = useRef(false);

  const handleEnter = () => {
    if (isTouchRef.current) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setShow(true), 150);
  };

  const handleLeave = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    setShow(false);
    setPos(null);
  };

  const handleTouchStart = useCallback(() => {
    isTouchRef.current = true;
  }, []);

  useEffect(() => {
    if (!show) return;
    const wrapper = triggerRef.current;
    const tip = tipRef.current;
    if (!wrapper || !tip) return;
    // display:contents makes the wrapper invisible to layout — use first child instead
    const trigger = (wrapper.firstElementChild as HTMLElement) ?? wrapper;
    const r = trigger.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    const pad = 8;
    let x = r.left + r.width / 2 - t.width / 2;
    x = Math.max(pad, Math.min(x, window.innerWidth - t.width - pad));
    // Place above by default; flip below if clipped at top
    let y = r.top - t.height - 6;
    if (y < pad) {
      y = r.bottom + 6;
    }
    setPos({ x, y });
  }, [show]);

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
        onTouchStart={handleTouchStart}
        className={className ?? 'inline-flex'}
      >
        {children}
      </div>
      {show && createPortal(
        <div
          ref={tipRef}
          className="fixed px-2.5 py-1.5 text-xs text-gray-200 bg-gray-900 border border-gray-700 rounded-lg shadow-xl pointer-events-none"
          style={{
            zIndex: 9999,
            left: pos ? pos.x : -9999,
            top: pos ? pos.y : -9999,
            opacity: pos ? 1 : 0,
            maxWidth: 280,
            whiteSpace: 'normal',
          }}
        >
          {text}
        </div>,
        document.body,
      )}
    </>
  );
}
