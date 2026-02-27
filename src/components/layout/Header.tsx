/**
 * Application header
 */

import { useState } from 'react';
import { Github, HelpCircle, Moon, Sun, X } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip.tsx';
import { GameModeButton } from '../game/GameModeButton.tsx';
import { useTheme } from '../../hooks/useTheme';

function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-white">Quick Start Guide</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-6 text-sm">
          <section>
            <h3 className="text-lg font-medium text-white mb-2">Getting Started</h3>
            <ol className="list-decimal list-inside space-y-2 text-gray-300">
              <li>Choose mode: <span className="text-indigo-400">Training</span> or <span className="text-cyan-400">Inference</span></li>
              <li>Select a model from the sidebar (e.g., LLaMA-2 7B)</li>
              <li>Choose a cluster configuration (e.g., 8x H100)</li>
              <li>Configure parallelism strategy, batch sizes, and precision</li>
              <li>Click "Run" to see results</li>
              <li>Share configuration via URL or export</li>
            </ol>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Parallelism Strategies</h3>
            <p className="text-gray-400 mb-2">Strategies (dropdown):</p>
            <ul className="space-y-2 text-gray-300 mb-3">
              <li><span className="text-indigo-400 font-medium">DDP</span> - Replicate model, sync gradients</li>
              <li><span className="text-indigo-400 font-medium">ZeRO-1</span> - Shard optimizer states (DeepSpeed)</li>
              <li><span className="text-indigo-400 font-medium">FSDP</span> - Fully shard params, grads, and optimizer (ZeRO-3)</li>
              <li><span className="text-indigo-400 font-medium">FSDP+TP, ZeRO-1+TP</span> - 2D: data + tensor parallelism</li>
              <li><span className="text-indigo-400 font-medium">3D (DDP/ZeRO-1/FSDP + TP + PP)</span> - For largest models</li>
            </ul>
            <p className="text-gray-400 mb-2">Additional dimensions (sidebar controls):</p>
            <ul className="space-y-2 text-gray-300">
              <li><span className="text-indigo-400 font-medium">TP</span> - Tensor parallelism, splits layers across GPUs (best within node)</li>
              <li><span className="text-indigo-400 font-medium">PP</span> - Pipeline parallelism, distributes layers across stages (1F1B, Interleaved, DualPipeV)</li>
              <li><span className="text-indigo-400 font-medium">CP</span> - Context parallelism for long sequences (ring or all-gather)</li>
              <li><span className="text-indigo-400 font-medium">SP</span> - Sequence parallelism toggle (reduces activation memory)</li>
              <li><span className="text-indigo-400 font-medium">EP</span> - Expert parallelism for MoE models</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Training Configuration</h3>
            <ul className="space-y-2 text-gray-300">
              <li><span className="text-orange-400 font-medium">Precision</span> - BF16, FP8 (Hopper+ with Transformer Engine)</li>
              <li><span className="text-orange-400 font-medium">Activation Checkpointing</span> - Selective (attention only) / Full / Off</li>
              <li><span className="text-orange-400 font-medium">Training Method</span> - Full / LoRA / QLoRA</li>
              <li><span className="text-orange-400 font-medium">Pipeline Schedule</span> - 1F1B, Interleaved 1F1B (virtual stages), DualPipe-V</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Training Scale</h3>
            <ul className="space-y-2 text-gray-300">
              <li><span className="text-orange-400 font-medium">Chinchilla Optimal</span> - ~20 tokens per parameter (active params for MoE; compute-optimal)</li>
              <li><span className="text-orange-400 font-medium">Heavy Overtrain</span> - ~200x active params (LLaMA-style, inference-optimized)</li>
              <li><span className="text-orange-400 font-medium">Fine-tune</span> - 1B tokens (for fine-tuning scenarios)</li>
              <li><span className="text-orange-400 font-medium">Custom</span> - Set your own token count</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Training Metrics</h3>
            <ul className="space-y-2 text-gray-300">
              <li><span className="text-yellow-400 font-medium">MFU</span> - Model FLOPS Utilization: 6PD / (time × peak). Always measured against BF16 peak for cross-model comparability</li>
              <li><span className="text-yellow-400 font-medium">FP8 HW Utilization</span> - Primary metric for FP8 training: MFU recomputed against FP8 peak TFLOPS, reflecting actual hardware utilization</li>
              <li><span className="text-yellow-400 font-medium">HFU</span> - Hardware FLOPS Utilization: includes recompute (8PD with checkpointing)</li>
              <li><span className="text-yellow-400 font-medium">Model FLOPs MFU</span> - Actual model FLOPs; shown when quadratic attention causes &gt;10% divergence from PaLM MFU</li>
              <li><span className="text-green-400 font-medium">Throughput</span> - Tokens processed per second</li>
              <li><span className="text-blue-400 font-medium">Step Time</span> - Time per training iteration</li>
              <li><span className="text-purple-400 font-medium">Memory</span> - GPU memory usage per device</li>
              <li><span className="text-gray-400 font-medium">vs Chinchilla</span> - How your training compares to compute-optimal</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Inference Metrics</h3>
            <ul className="space-y-2 text-gray-300">
              <li><span className="text-cyan-400 font-medium">TTFT</span> - Time to First Token (prefill latency)</li>
              <li><span className="text-blue-400 font-medium">TPOT</span> - Time Per Output Token (decode latency)</li>
              <li><span className="text-green-400 font-medium">Throughput</span> - Tokens generated per second</li>
              <li><span className="text-purple-400 font-medium">KV Cache</span> - Memory for cached key/value states</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Inference Optimizations</h3>
            <ul className="space-y-2 text-gray-300">
              <li><span className="text-cyan-400 font-medium">Weight Quantization</span> - FP8, INT8, INT4/GPTQ/AWQ, GGUF formats (Q4_K_M, Q8_0, etc.)</li>
              <li><span className="text-cyan-400 font-medium">KV Cache Quantization</span> - FP8, INT8, FP4</li>
              <li><span className="text-cyan-400 font-medium">Flash Attention</span> - Memory-efficient attention</li>
              <li><span className="text-cyan-400 font-medium">Continuous Batching</span> - Iteration-level scheduling</li>
              <li><span className="text-cyan-400 font-medium">Speculative Decoding</span> - Use draft model for faster generation</li>
              <li><span className="text-cyan-400 font-medium">Tensor Parallel</span> - Split model across GPUs</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Keyboard Shortcuts</h3>
            <ul className="space-y-2 text-gray-300">
              <li><kbd className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-gray-200">Ctrl+Enter</kbd> — Run simulation</li>
              <li><kbd className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-xs font-mono text-gray-200">Ctrl+Shift+Enter</kbd> — Auto-optimize</li>
            </ul>
          </section>

          <section>
            <h3 className="text-lg font-medium text-white mb-2">Learning Mode</h3>
            <p className="text-gray-300">
              Click <span className="text-green-400 font-medium">Learn</span> in the header to enter interactive challenges.
              Configure training runs to hit target metrics — MFU, throughput, memory — across increasing difficulty levels.
              Auto-optimize is disabled so you learn the trade-offs yourself.
            </p>
          </section>

        </div>

        <div className="mt-6 pt-4 border-t border-gray-800 text-xs text-gray-500">
          Theoretical estimates validated against published benchmarks. Actual performance will vary.
        </div>
      </div>
    </div>
  );
}

export function Header() {
  const [showHelp, setShowHelp] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const handleHomeClick = () => {
    window.location.reload();
  };

  return (
    <>
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/50">
        <button
          onClick={handleHomeClick}
          className="flex items-center gap-3 hover:opacity-80 transition-opacity cursor-pointer"
        >
          <img src="/favicon.svg" alt="" className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg" />
          <div className="text-left">
            <h1 className="text-sm sm:text-lg font-semibold text-white">
              LLM Cluster Simulator
            </h1>
            <p className="text-[10px] sm:text-xs text-gray-400 hidden sm:block">
              Plan distributed training & inference
            </p>
          </div>
        </button>

        <div className="flex items-center gap-2">
          <GameModeButton />
          <button
            onClick={() => setShowHelp(true)}
            className="flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-white hover:bg-gray-800/50 rounded-lg transition-colors cursor-pointer"
          >
            <HelpCircle className="w-4 h-4" />
            <span className="hidden sm:inline">Help</span>
          </button>
          <a
            href="https://github.com/zhebrak/llm-cluster-simulator"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-white hover:bg-gray-800/50 rounded-lg transition-colors cursor-pointer"
          >
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
          <Tooltip text={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}>
            <button
              onClick={toggleTheme}
              className="flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-white hover:bg-gray-800/50 rounded-lg transition-colors cursor-pointer"
            >
              {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>
          </Tooltip>
        </div>
      </header>

      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
    </>
  );
}
