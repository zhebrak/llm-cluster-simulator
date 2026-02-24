import { useState, useEffect, useCallback } from 'react';

type Theme = 'light' | 'dark';

const STORAGE_KEY = 'llm-sim-theme';

function getInitialTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === 'light' || stored === 'dark') return stored;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme: Theme) {
  // Suppress CSS transitions during theme switch to prevent color flashes
  document.documentElement.classList.add('no-transitions');
  document.documentElement.setAttribute('data-theme', theme);
  // Re-enable after browser paints the new styles (double-rAF ensures paint)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.documentElement.classList.remove('no-transitions');
    });
  });
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    applyTheme(t);
    localStorage.setItem(STORAGE_KEY, t);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  }, [theme, setTheme]);

  // Apply on mount (in case SSR/hydration mismatch)
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  // Listen for OS preference changes when no explicit choice stored
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => {
      if (!localStorage.getItem(STORAGE_KEY)) {
        const t = e.matches ? 'dark' : 'light';
        setThemeState(t);
        applyTheme(t);
      }
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  return { theme, setTheme, toggleTheme } as const;
}
