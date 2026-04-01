/**
 * Citation modal — displays BibTeX for citing the simulator.
 */

import { useState } from 'react';
import { X, Copy, Check } from 'lucide-react';
import { ModalBackdrop } from './ModalBackdrop.tsx';
import { copyToClipboard } from '../../utils/clipboard.ts';

const BIBTEX = `@misc{zhebrak2026llmclustersim,
  author       = {Zhebrak, Alex},
  title        = {{LLM Cluster Simulator}: Interactive Distributed Training and Inference Planning},
  year         = {2026},
  url          = {https://github.com/zhebrak/llm-cluster-simulator},
  doi          = {10.5281/zenodo.19365122},
  note         = {Browser-based simulator for GPU cluster parallelism strategies, calibrated against published benchmarks from Meta, DeepSeek, and NVIDIA}
}`;

export function CitationModal({ onClose }: { onClose: () => void }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    copyToClipboard(BIBTEX);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <ModalBackdrop onBackdropClick={onClose}>
      <div
        className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-lg w-full mx-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Cite this tool</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-300 cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        <p className="text-sm text-gray-400 mb-3">
          If you use LLM Cluster Simulator in your work, please cite:
        </p>

        <pre className="bg-gray-950 border border-gray-800 rounded-lg p-4 text-xs font-mono text-gray-300 overflow-x-auto whitespace-pre-wrap">
          {BIBTEX}
        </pre>

        <div className="mt-3 flex justify-end">
          <button
            onClick={handleCopy}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer ${
              copied
                ? 'text-green-400 bg-green-500/10'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied!' : 'Copy BibTeX'}
          </button>
        </div>
      </div>
    </ModalBackdrop>
  );
}
