/**
 * Main application shell layout
 */

import { useState, type ReactNode } from 'react';
import { ChevronsRight, Monitor, X } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip.tsx';
import { Header } from './Header.tsx';
import { Sidebar } from './Sidebar.tsx';

interface AppShellProps {
  children: ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [smallScreenDismissed, setSmallScreenDismissed] = useState(false);

  return (
    <div className="flex flex-col h-dvh bg-gray-950">
      <Header />
      {!smallScreenDismissed && (
        <div className="small-screen-banner md:hidden flex items-center gap-2 px-4 py-2 border-b text-xs">
          <Monitor className="w-4 h-4 shrink-0" />
          <span className="flex-1">This simulator is designed for larger screens. For the best experience, use a desktop or tablet in landscape mode.</span>
          <button
            onClick={() => setSmallScreenDismissed(true)}
            className="small-screen-dismiss shrink-0 p-0.5 rounded transition-colors cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar wrapper — always mounted, animated width */}
        <div
          className="shrink-0 overflow-hidden transition-[width] duration-200 ease-in-out border-r border-gray-800"
          style={{ width: sidebarOpen ? 320 : 40 }}
        >
          {sidebarOpen ? (
            <div className="w-80 h-full">
              <Sidebar onCollapse={() => setSidebarOpen(false)} />
            </div>
          ) : (
            <div className="w-10 h-full bg-gray-900/30 flex flex-col items-center pt-3">
              <Tooltip text="Show sidebar">
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="p-1.5 rounded-lg text-gray-500 hover:text-white hover:bg-white/[0.06] transition-colors cursor-pointer"
                >
                  <ChevronsRight className="w-5 h-5" />
                </button>
              </Tooltip>
            </div>
          )}
        </div>
        <main className="flex-1 overflow-auto p-6 flex flex-col" style={{ scrollbarGutter: 'stable' }}>
          {children}
          <div className="mt-auto pt-8 pb-2 text-xs text-gray-500 text-right">
            v1.1.0 · Alex Zhebrak · 2026
          </div>
        </main>
      </div>
    </div>
  );
}
