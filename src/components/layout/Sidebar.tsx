/**
 * Sidebar with configuration panels
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, ChevronsLeft, Box, Server, Layers, Settings, Play, Zap, Plus, X } from 'lucide-react';
import { Tooltip } from '../ui/Tooltip.tsx';
import { CustomModelEditor } from './CustomModelEditor.tsx';
import { useConfigStore, getNextDemoPreset, getPrevDemoPreset, type AppMode, type TrainingGoal } from '../../stores/config.ts';
import type { FinetuningMethod, LoraTargetModules } from '../../core/strategies/lora.ts';
import { computeLoraTrainableParams } from '../../core/strategies/lora.ts';
import { useSimulationStore } from '../../stores/simulation.ts';
import { MODEL_FAMILIES, ALL_MODEL_CONFIGS } from '../../core/models/architectures.ts';
import { modelRegistry } from '../../core/models/registry.ts';
import { PRESET_LABELS, PRESET_CATEGORIES, INFERENCE_PRESET_CATEGORIES, INFERENCE_PRESET_LABELS } from '../../core/hardware/presets.ts';
import { ALL_GPUS, GPU_CATEGORIES, supportsFlashAttention } from '../../core/hardware/gpu.ts';
import { AVAILABLE_STRATEGIES, STRATEGY_GROUPS } from '../../core/strategies/index.ts';
import { getGPUHourlyRate } from '../../core/cost/index.ts';
import { formatNumber } from '../../types/base.ts';
import { useGameStore } from '../../stores/game.ts';
import { useRPGStore } from '../../stores/rpg.ts';
import { getAvailableHardware } from '../../rpg/hardware.ts';
import { getMissionById } from '../../rpg/missions/index.ts';

// Format numbers with commas for readability
function formatWithCommas(n: number): string {
  return Math.round(n).toLocaleString();
}

// Format hours into human-readable duration (hours, days, months, years)
function formatDuration(hours: number): string {
  if (!isFinite(hours)) return 'OOM';
  if (hours < 24) return `${parseFloat(hours.toFixed(1))} hours`;
  if (hours < 24 * 30) return `${parseFloat((hours / 24).toFixed(1))} days`;
  if (hours < 24 * 365) return `${parseFloat((hours / 24 / 30).toFixed(1))} months`;
  return `${parseFloat((hours / 24 / 365).toFixed(1))} years`;
}

// Human-readable labels for optimizer change-tracking fields
const CHANGELOG_LABELS: Record<string, string> = {
  strategyType: 'Strategy',
  globalBatchSize: 'Global Batch Size',
  microBatchSize: 'Micro Batch Size',
  mixedPrecision: 'Precision',
  activationCheckpointing: 'Activation Checkpointing',
  flashAttention: 'Flash Attention',
  tp: 'Tensor Parallel',
  pp: 'Pipeline Parallel',
  ep: 'Expert Parallel',
  cp: 'Context Parallel',
  cpImplementation: 'CP Implementation',
  sequenceParallel: 'Sequence Parallel',
  pipelineSchedule: 'Pipeline Schedule',
  interleavedStages: 'Virtual Stages (v)',
};

const STRATEGY_LABELS: Record<string, string> = {
  ddp: 'DDP',
  'zero-1': 'ZeRO-1',
  fsdp: 'FSDP',
  'fsdp-tp': 'FSDP + TP',
  'zero1-tp': 'ZeRO-1 + TP',
  'ddp-tp-pp': 'DDP + TP + PP',
  'zero1-tp-pp': 'ZeRO-1 + TP + PP',
  'fsdp-tp-pp': 'FSDP + TP + PP',
};

function changelogLabel(field: string): string {
  return CHANGELOG_LABELS[field] ?? field;
}

function formatChangelogValue(field: string, value: string): string {
  if (field === 'strategyType') return STRATEGY_LABELS[value] ?? value;
  if (field === 'flashAttention' || field === 'sequenceParallel') {
    return value === 'true' ? 'On' : 'Off';
  }
  if (field === 'activationCheckpointing') {
    if (value === 'off') return 'Off';
    if (value === 'selective') return 'Selective';
    if (value === 'full') return 'Full';
    // Legacy boolean fallback
    return value === 'true' ? 'On' : 'Off';
  }
  if (field === 'cpImplementation') {
    return value === 'all-gather' ? 'All-Gather' : 'Ring';
  }
  if (field === 'pipelineSchedule') {
    if (value === '1f1b') return '1F1B';
    if (value === 'interleaved-1f1b') return 'Interleaved 1F1B';
    if (value === 'dualpipe-v') return 'DualPipeV';
    if (value === 'gpipe') return 'GPipe';
    return value;
  }
  return value;
}

// Format token counts nicely
function formatTokens(tokens: number): string {
  if (tokens >= 1e12) return `${(tokens / 1e12).toFixed(1)}T tokens`;
  if (tokens >= 1e9) return `${(tokens / 1e9).toFixed(0)}B tokens`;
  if (tokens >= 1e6) return `${(tokens / 1e6).toFixed(0)}M tokens`;
  return `${tokens} tokens`;
}

interface CollapsibleSectionProps {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

function CollapsibleSection({ title, icon, defaultOpen = true, children }: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-gray-800">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800/50 transition-colors cursor-pointer"
      >
        {icon}
        <span className="flex-1">{title}</span>
        {isOpen ? (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </button>
      {isOpen && (
        <div className="px-3 pb-3 space-y-2">
          {children}
        </div>
      )}
    </div>
  );
}

interface SelectOption {
  value: string;
  label: string;
  group?: string;
}

interface SelectProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: SelectOption[];
}

function Select({ label, value, onChange, options }: SelectProps) {
  // Group options by their group field (undefined = ungrouped)
  const hasGroups = options.some(o => o.group);

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-0.5">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent cursor-pointer"
      >
        {hasGroups ? (() => {
          const groups: { label: string | undefined; opts: SelectOption[] }[] = [];
          for (const opt of options) {
            const last = groups[groups.length - 1];
            if (last && last.label === opt.group) {
              last.opts.push(opt);
            } else {
              groups.push({ label: opt.group, opts: [opt] });
            }
          }
          return groups.map((g) =>
            g.label ? (
              <optgroup key={g.label} label={g.label}>
                {g.opts.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </optgroup>
            ) : (
              g.opts.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))
            )
          );
        })() : options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

interface SegmentedControlProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; tooltip?: string }[];
  disabled?: boolean;
}

function SegmentedControl({ label, value, onChange, options, disabled }: SegmentedControlProps) {
  return (
    <div className={disabled ? 'opacity-50' : ''}>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      <div className="flex gap-0.5 bg-gray-800 p-0.5 rounded-lg">
        {options.map((opt) => {
          const btn = (
            <button
              key={opt.value}
              onClick={() => !disabled && onChange(opt.value)}
              disabled={disabled}
              className={`flex-1 px-2 py-1 text-xs font-medium rounded-md transition-colors focus:outline-none ${
                disabled ? 'cursor-not-allowed' : 'cursor-pointer'
              } ${
                value === opt.value
                  ? 'bg-accent text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {opt.label}
            </button>
          );
          return opt.tooltip ? (
            <Tooltip key={opt.value} text={opt.tooltip} className="flex-1 flex">
              {btn}
            </Tooltip>
          ) : btn;
        })}
      </div>
    </div>
  );
}

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}

function NumberInput({ label, value, onChange, min, max, step = 1, disabled }: NumberInputProps) {
  const [localValue, setLocalValue] = useState(String(value));

  // Sync local value when external value changes
  useEffect(() => {
    setLocalValue(String(value));
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalValue(newValue);

    // Don't update parent while field is empty (user is still typing)
    if (newValue === '') return;

    // Only update parent if it's a valid finite number
    const num = Number(newValue);
    if (!Number.isFinite(num)) return;
    // Don't update parent if below min — user may still be typing (e.g. "1" → "1024")
    if (min !== undefined && num < min) return;
    const clamped = Math.min(max ?? Infinity, num);
    // For integer fields (step >= 1), truncate to integer
    if (step >= 1) {
      onChange(Math.trunc(clamped));
    } else {
      onChange(clamped);
    }
  };

  const handleBlur = () => {
    const num = Number(localValue);
    if (localValue === '' || !Number.isFinite(num)) {
      setLocalValue(String(value));
    } else if (min !== undefined && num < min) {
      setLocalValue(String(min));
      onChange(min);
    }
  };

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-0.5">{label}</label>
      <input
        type="number"
        value={localValue}
        onChange={handleChange}
        onBlur={handleBlur}
        onFocus={(e) => e.target.select()}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        className={`w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent${disabled ? ' opacity-50 cursor-not-allowed' : ''}`}
      />
    </div>
  );
}

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  description?: string;
  disabled?: boolean;
}

function Toggle({ label, checked, onChange, description, disabled }: ToggleProps) {
  return (
    <div
      className={`flex items-center justify-between py-2 px-3 -mx-3 rounded-lg transition-colors group ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-800/50 cursor-pointer'}`}
      onClick={() => !disabled && onChange(!checked)}
    >
      <div className="flex-1 min-w-0">
        <span className="text-sm text-gray-300 group-hover:text-white transition-colors">{label}</span>
        {description && (
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        )}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={(e) => {
          e.stopPropagation();
          if (!disabled) onChange(!checked);
        }}
        className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 ring-accent focus:ring-offset-2 focus:ring-offset-gray-900 ${disabled ? 'cursor-not-allowed' : 'cursor-pointer'} ${
          checked ? 'bg-accent' : 'bg-gray-600'
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
            checked ? 'translate-x-4' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  );
}

interface ModelOption {
  value: string;
  label: string;
  isAction?: boolean;
  deletable?: boolean;
}

interface ModelGroup {
  label: string;
  options: ModelOption[];
}

function ModelSelector({ value, onChange, onDeleteOption, groupedModels, label = 'Model', placeholder = 'Search models...' }: {
  value: string;
  onChange: (id: string) => void;
  onDeleteOption?: (id: string) => void;
  groupedModels: ModelGroup[];
  label?: string;
  placeholder?: string;
}) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const currentLabel = useMemo(() => {
    for (const group of groupedModels) {
      const opt = group.options.find(o => o.value === value);
      if (opt) return opt.label;
    }
    return value;
  }, [value, groupedModels]);

  const filtered = useMemo(() => {
    if (!query) return groupedModels;
    const q = query.toLowerCase();
    return groupedModels
      .map(group => ({
        label: group.label,
        options: group.options.filter(o =>
          o.label.toLowerCase().includes(q) ||
          o.value.toLowerCase().includes(q) ||
          group.label.toLowerCase().includes(q)
        ),
      }))
      .filter(group => group.options.length > 0);
  }, [query, groupedModels]);

  const flatOptions = useMemo(() =>
    filtered.flatMap(g => g.options),
    [filtered]
  );

  // Click outside to close
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
        setConfirmDeleteId(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [isOpen]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (highlightIndex < 0 || !listRef.current) return;
    const items = listRef.current.querySelectorAll('[data-option]');
    items[highlightIndex]?.scrollIntoView({ block: 'nearest' });
  }, [highlightIndex]);

  const handleSelect = (id: string) => {
    onChange(id);
    setIsOpen(false);
    setQuery('');
    setConfirmDeleteId(null);
    inputRef.current?.blur();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        e.preventDefault();
        setIsOpen(true);
        setHighlightIndex(0);
      }
      return;
    }
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlightIndex(i => Math.min(i + 1, flatOptions.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setHighlightIndex(i => Math.max(i - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (highlightIndex >= 0 && highlightIndex < flatOptions.length) {
          handleSelect(flatOptions[highlightIndex].value);
        }
        break;
      case 'Escape':
        e.preventDefault();
        setIsOpen(false);
        setQuery('');
        setConfirmDeleteId(null);
        inputRef.current?.blur();
        break;
    }
  };

  let optionIdx = -1;

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-xs text-gray-400 mb-0.5">{label}</label>
      <input
        ref={inputRef}
        type="text"
        value={isOpen ? query : currentLabel}
        onChange={(e) => {
          setQuery(e.target.value);
          setHighlightIndex(0);
          if (!isOpen) setIsOpen(true);
        }}
        onFocus={() => {
          setQuery('');
          setIsOpen(true);
          setHighlightIndex(-1);
        }}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent"
      />
      {isOpen && (
        <div
          ref={listRef}
          className="absolute w-full mt-1 bg-gray-800 border border-gray-700 rounded-md shadow-lg max-h-64 overflow-y-auto z-50"
        >
          {filtered.length === 0 ? (
            <div className="px-2 py-2 text-sm text-gray-500">No models found</div>
          ) : (
            filtered.map(group => (
              <div key={group.label}>
                <div className="text-xs text-gray-500 px-2 py-1 font-semibold sticky top-0 bg-gray-800">
                  {group.label}
                </div>
                {group.options.map(opt => {
                  optionIdx++;
                  const idx = optionIdx;
                  const isHighlighted = idx === highlightIndex;
                  const isCurrent = opt.value === value;
                  const isConfirmingDelete = confirmDeleteId === opt.value;

                  if (isConfirmingDelete) {
                    return (
                      <div
                        key={opt.value}
                        data-option
                        onMouseEnter={() => setHighlightIndex(idx)}
                        className={`flex items-center justify-between px-2 py-1.5 text-sm ${isHighlighted ? 'bg-gray-700' : ''}`}
                      >
                        <span className="text-gray-400 truncate">Delete?</span>
                        <div className="flex items-center gap-2 ml-2">
                          <button
                            onMouseDown={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              onDeleteOption?.(opt.value);
                              setConfirmDeleteId(null);
                            }}
                            className="text-xs text-red-400 hover:text-red-300 font-medium cursor-pointer"
                          >
                            Yes
                          </button>
                          <button
                            onMouseDown={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setConfirmDeleteId(null);
                            }}
                            className="text-xs text-gray-400 hover:text-white font-medium cursor-pointer"
                          >
                            No
                          </button>
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div
                      key={opt.value}
                      data-option
                      onMouseDown={(e) => {
                        e.preventDefault();
                        if (!opt.isAction) {
                          handleSelect(opt.value);
                        } else {
                          onChange(opt.value);
                          setIsOpen(false);
                          setQuery('');
                        }
                      }}
                      onMouseEnter={() => setHighlightIndex(idx)}
                      className={`flex items-center justify-between px-2 py-1.5 text-sm cursor-pointer ${
                        isHighlighted ? 'bg-gray-700' : ''
                      } ${opt.isAction ? 'text-accent font-medium' : isCurrent ? 'text-accent font-semibold' : 'text-white'}`}
                    >
                      <span className="truncate flex items-center gap-1.5">
                        {opt.isAction && <Plus className="w-3.5 h-3.5" />}
                        {opt.label}
                      </span>
                      <div className="flex items-center gap-1.5 ml-2 flex-shrink-0">
                        {!opt.isAction && modelRegistry.getMetadata(opt.value)?.isMoE && (
                          <span className="text-xs text-gray-500">MoE</span>
                        )}
                        {opt.deletable && onDeleteOption && (
                          <button
                            onMouseDown={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setConfirmDeleteId(opt.value);
                            }}
                            className="p-0.5 rounded hover:bg-gray-600 text-gray-500 hover:text-red-400 transition-colors cursor-pointer"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function Sidebar({ onCollapse }: { onCollapse?: () => void }) {
  const config = useConfigStore();
  const simulation = useSimulationStore();
  const gameActive = useGameStore(s => s.active);
  const inTask = useGameStore(s => !!s.activeTaskId);
  const inMission = useRPGStore(s => !!s.activeMissionId);
  const rpgActive = useRPGStore(s => s.active);
  const rpgCompletedMissions = useRPGStore(s => s.completedMissions);
  const rpgActiveMissionId = useRPGStore(s => s.activeMissionId);
  const blurred = inTask || inMission;
  const blurLabel = inTask ? 'Hidden in learning mode' : 'Hidden in game mode';
  const [showCustomEditor, setShowCustomEditor] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const optBannerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll sidebar to show optimization result banner when it appears
  useEffect(() => {
    if (simulation.optimizationResult && optBannerRef.current) {
      optBannerRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [simulation.optimizationResult]);

  const customGroup: ModelGroup = {
    label: 'Custom',
    options: [
      { value: 'custom-new', label: 'Define Custom Model...', isAction: true },
      ...Object.entries(config.customModels).map(([id, cfg]) => ({
        value: id, label: cfg.name, deletable: true,
      })),
    ],
  };

  const groupedModels = [customGroup, ...MODEL_FAMILIES.map(family => ({
    label: family.name,
    options: family.models.map(id => ({
      value: id,
      label: ALL_MODEL_CONFIGS[id]?.name ?? id,
    })),
  }))];

  const presetCategories = config.mode === 'inference' ? INFERENCE_PRESET_CATEGORIES : PRESET_CATEGORIES;
  const presetLabels = config.mode === 'inference' ? INFERENCE_PRESET_LABELS : PRESET_LABELS;

  const groupedPresets = presetCategories.map(cat => ({
    label: cat.name,
    options: cat.presets.map(id => ({
      value: id,
      label: presetLabels[id] || id,
    })),
  }));

  const GPU_CATEGORY_LABELS: Record<string, string> = {
    datacenter: 'Datacenter',
    nextGen: 'Next Gen',
    amd: 'AMD',
    professional: 'Professional',
    consumer: 'Consumer',
    legacy: 'Legacy',
  };

  // In RPG mode, only show GPUs from unlocked hardware tiers + active mission's setup GPU
  const rpgAllowedGPUs = rpgActive
    ? (() => {
        const set = new Set(getAvailableHardware(rpgCompletedMissions).map(s => s.gpuId));
        if (rpgActiveMissionId) {
          const mission = getMissionById(rpgActiveMissionId);
          if (mission?.setup.gpuId) set.add(mission.setup.gpuId);
        }
        return set;
      })()
    : null;

  const groupedGPUs = Object.entries(GPU_CATEGORIES).map(([catId, gpuIds]) => ({
    label: GPU_CATEGORY_LABELS[catId] || catId,
    options: (rpgAllowedGPUs ? gpuIds.filter(id => rpgAllowedGPUs.has(id)) : gpuIds).map(id => ({
      value: id,
      label: ALL_GPUS[id]?.name || id,
    })),
  })).filter(g => g.options.length > 0);

  const groupedStrategies = STRATEGY_GROUPS.map(group => ({
    label: group.name,
    options: AVAILABLE_STRATEGIES
      .filter(s => s.group === group.id)
      .map(s => ({ value: s.id, label: s.name })),
  }));

  const gpuSupportsFA = config.clusterConfig ? supportsFlashAttention(config.clusterConfig.node.gpu) : true;

  const precisionOptions = [
    { value: 'fp32', label: 'FP32' },
    { value: 'tf32', label: 'TF32' },
    { value: 'fp16', label: 'FP16' },
    { value: 'bf16', label: 'BF16' },
    { value: 'fp8', label: 'FP8' },
  ];

  const quantizationOptions: SelectOption[] = [
    // Standard / server quantization
    { value: 'bf16', label: 'BF16 (no quant)', group: 'Standard' },
    { value: 'fp16', label: 'FP16 (no quant)', group: 'Standard' },
    { value: 'fp8', label: 'FP8 (native)', group: 'Standard' },
    { value: 'fp4', label: 'FP4 (native)', group: 'Standard' },
    { value: 'int8', label: 'INT8 (W8A8)', group: 'Standard' },
    { value: 'int4', label: 'INT4 / GPTQ / AWQ', group: 'Standard' },
    // GGUF quantization (llama.cpp / Ollama)
    { value: 'q8_0', label: 'Q8_0 (8.5 bpw)', group: 'GGUF' },
    { value: 'q6_k', label: 'Q6_K (6.6 bpw)', group: 'GGUF' },
    { value: 'q5_k_m', label: 'Q5_K_M (5.7 bpw)', group: 'GGUF' },
    { value: 'q4_k_m', label: 'Q4_K_M (4.8 bpw)', group: 'GGUF' },
    { value: 'q3_k_m', label: 'Q3_K_M (3.9 bpw)', group: 'GGUF' },
    { value: 'q2_k', label: 'Q2_K (3.0 bpw)', group: 'GGUF' },
  ];

  const modeButtons: { mode: AppMode; label: string; icon: React.ReactNode }[] = [
    { mode: 'training', label: 'Training', icon: <Settings className="w-4 h-4" /> },
    { mode: 'inference', label: 'Inference', icon: <Zap className="w-4 h-4" /> },
  ];

  return (
    <aside className="w-80 h-full border-r border-gray-800 bg-gray-900/30 flex flex-col">
      <div ref={scrollAreaRef} className="flex-1 overflow-y-auto">
      {/* Mode Selector */}
      <div className="p-3 border-b border-gray-800">
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs text-gray-400">Mode</label>
          {onCollapse && (
            <Tooltip text="Hide sidebar">
              <button
                onClick={onCollapse}
                className="p-1 -mr-1 rounded-lg text-gray-500 hover:text-white hover:bg-white/[0.06] transition-colors cursor-pointer"
              >
                <ChevronsLeft className="w-5 h-5" />
              </button>
            </Tooltip>
          )}
        </div>
        <div className="flex gap-1 bg-gray-800 p-0.5 rounded-lg">
          {modeButtons.map(({ mode, label, icon }) => (
            <button
              key={mode}
              onClick={() => config.setMode(mode)}
              className={`flex-1 flex items-center justify-center gap-1 px-2 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer focus:outline-none ${
                config.mode === mode
                  ? 'bg-accent text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {icon}
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Example preset cycler — prev / next */}
      <div className="px-3 pt-1.5 pb-3 border-b border-gray-800 relative">
        {blurred && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <span className="text-xs text-gray-500">{blurLabel}</span>
          </div>
        )}
        <div className={blurred ? 'blur-sm pointer-events-none select-none' : ''}>
        <label className="text-xs text-gray-400 mb-1 block">{config.mode === 'training' ? 'Training' : 'Inference'} Presets</label>
        <div className="flex items-center gap-1.5">
        <Tooltip text={`${getPrevDemoPreset(config.mode).modelLabel} · ${getPrevDemoPreset(config.mode).clusterLabel}`}>
          <button
            onClick={(e) => { config.resetPrev(); (e.target as HTMLElement).blur(); }}
            className="group/prev flex items-center justify-center p-1.5 text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white rounded-md transition-colors cursor-pointer"
          >
            <ChevronLeft className="w-3.5 h-3.5" />
          </button>
        </Tooltip>
        <button
          onClick={(e) => { config.reset(); (e.target as HTMLElement).blur(); }}
          className="group/next flex items-center justify-center gap-1.5 flex-1 min-w-0 px-2 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 text-gray-500 hover:text-white rounded-md transition-colors cursor-pointer"
        >
          <span className="truncate">{getNextDemoPreset(config.mode).modelLabel} · {getNextDemoPreset(config.mode).clusterLabel}</span>
          <ChevronRight className="w-3.5 h-3.5 flex-shrink-0" />
        </button>
        </div>
        </div>
      </div>

      {/* Model Configuration */}
      <CollapsibleSection
        title="Model"
        icon={<Box className="w-4 h-4 text-accent" />}
      >
        <ModelSelector
          value={showCustomEditor ? 'custom-new' : config.modelId}
          onChange={(id) => {
            if (id === 'custom-new') {
              setShowCustomEditor(true);
            } else {
              setShowCustomEditor(false);
              config.setModel(id);
            }
          }}
          onDeleteOption={(id) => config.deleteCustomModel(id)}
          groupedModels={groupedModels}
        />
        {showCustomEditor && (
          <CustomModelEditor onClose={() => setShowCustomEditor(false)} />
        )}
        {config.modelSpec && (
          <div className="text-xs text-gray-500 space-y-1">
            <div>
              Parameters: {(config.modelSpec.totalParams / 1e9).toFixed(2)}B
              {config.modelSpec.isMoE && config.modelSpec.activeParams && (
                <span> ({(config.modelSpec.activeParams / 1e9).toFixed(2)}B active)</span>
              )}
            </div>
            <div>Layers: {config.modelSpec.numLayers}</div>
            <div>Hidden: {formatWithCommas(config.modelSpec.hiddenSize)}</div>
            {config.modelSpec.isMoE && config.modelSpec.numExperts && (
              <div>Experts: {config.modelSpec.numExperts} ({config.modelSpec.numActiveExperts ?? 2} active)</div>
            )}
          </div>
        )}
      </CollapsibleSection>

      {/* Cluster Configuration */}
      <CollapsibleSection
        title="Cluster"
        icon={<Server className="w-4 h-4 text-accent" />}
      >
        <div className="relative">
          {rpgActive && (
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <span className="text-xs text-gray-500">Hidden in game mode</span>
            </div>
          )}
          <div className={rpgActive ? 'blur-sm pointer-events-none select-none' : ''}>
            <label className="block text-xs text-gray-400 mb-0.5">Preset</label>
            <select
              value={config.clusterId}
              onChange={(e) => config.setCluster(e.target.value)}
              className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent cursor-pointer"
            >
              <option value="custom">Custom</option>
              {groupedPresets.map((group) => (
                <optgroup key={group.label} label={group.label}>
                  {group.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-0.5">GPU Type</label>
          <select
            value={config.gpuId}
            onChange={(e) => config.setCustomCluster(e.target.value, config.numGPUs, config.gpusPerNode)}
            className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent cursor-pointer"
          >
            {groupedGPUs.map((group) => (
              <optgroup key={group.label} label={group.label}>
                {group.options.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <NumberInput
            label="Total GPUs"
            value={config.numGPUs}
            onChange={(value) => config.setCustomCluster(config.gpuId, value, config.gpusPerNode)}
            min={1}
          />
          <NumberInput
            label="GPUs/Node"
            value={config.gpusPerNode}
            onChange={(value) => config.setCustomCluster(config.gpuId, config.numGPUs, value)}
            min={1}
          />
        </div>
        <div className="relative">
          {blurred && (
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <span className="text-xs text-gray-500">{blurLabel}</span>
            </div>
          )}
          <div className={blurred ? 'blur-sm pointer-events-none select-none' : ''}>
            <NumberInput
              label="$/GPU-hour"
              value={config.pricePerGPUHour ?? getGPUHourlyRate(config.gpuId).rate}
              onChange={(value) => config.setPricePerGPUHour(value)}
              min={0}
              step={0.01}
            />
            {config.pricePerGPUHour !== null && (
              <Tooltip text={`Reset to default ($${getGPUHourlyRate(config.gpuId).rate.toFixed(2)}/hr)`} className="contents">
                <button
                  onClick={() => config.setPricePerGPUHour(null)}
                  className="absolute top-0 right-0 text-xs text-gray-400 hover:text-accent cursor-pointer"
                >
                  reset to default
                </button>
              </Tooltip>
            )}
          </div>
        </div>
        {config.clusterConfig && (
          <div className="text-xs text-gray-500 space-y-1">
            <div>Nodes: {config.clusterConfig.numNodes}</div>
            <div>Memory: {formatWithCommas(config.clusterConfig.totalMemoryGB)} GB</div>
            <div>Peak TFLOPS: {formatWithCommas(config.clusterConfig.totalTFLOPS)}</div>
          </div>
        )}
      </CollapsibleSection>

      {/* Strategy Configuration - Training Mode Only */}
      {(config.mode === 'training') && (
        <CollapsibleSection
          title="Strategy"
          icon={<Layers className="w-4 h-4 text-accent" />}
        >
          <div>
            <label className="block text-xs text-gray-400 mb-0.5">Parallelism Strategy</label>
            <select
              value={config.training.strategyType}
              onChange={(e) => config.setStrategy(e.target.value as typeof config.training.strategyType)}
              className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent cursor-pointer"
            >
              {groupedStrategies.map((group) => (
                <optgroup key={group.label} label={group.label}>
                  {group.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>
          {['fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(config.training.strategyType) && (
            <NumberInput
              label="Tensor Parallel Degree"
              value={config.training.tpDegree}
              onChange={(value) => config.setStrategyParams({ tpDegree: value })}
              min={1}
            />
          )}
          {['ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(config.training.strategyType) && (
            <>
              <NumberInput
                label="Pipeline Parallel Degree"
                value={config.training.ppDegree}
                onChange={(value) => config.setStrategyParams({ ppDegree: value })}
                min={1}
              />
              {config.training.ppDegree > 1 && (
                <>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Pipeline Schedule</label>
                    <select
                      className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-white"
                      value={config.training.pipelineSchedule}
                      onChange={(e) => {
                        const schedule = e.target.value as '1f1b' | 'interleaved-1f1b' | 'dualpipe-v';
                        config.setStrategyParams({
                          pipelineSchedule: schedule,
                          ...(schedule === 'interleaved-1f1b' ? { interleavedStages: config.training.interleavedStages || 2 } : {}),
                        });
                      }}
                    >
                      <option value="1f1b">1F1B</option>
                      <option value="interleaved-1f1b">Interleaved 1F1B</option>
                      <option value="dualpipe-v">DualPipeV</option>
                    </select>
                  </div>
                  {config.training.pipelineSchedule === 'interleaved-1f1b' && (
                    <>
                      <NumberInput
                        label="Virtual Stages (v)"
                        value={config.training.interleavedStages}
                        onChange={(value) => config.setStrategyParams({ interleavedStages: value })}
                        min={2}
                      />
                      {config.modelSpec && config.modelSpec.numLayers % (config.training.ppDegree * config.training.interleavedStages) !== 0 && (
                        <p className="text-xs text-yellow-400">
                          Layers ({config.modelSpec.numLayers}) not divisible by PP×v ({config.training.ppDegree}×{config.training.interleavedStages}={config.training.ppDegree * config.training.interleavedStages}). Stages will get uneven layer counts, causing load imbalance.
                        </p>
                      )}
                    </>
                  )}
                  {config.training.pipelineSchedule === 'dualpipe-v' && config.training.gradientAccumulationSteps < 2 * config.training.ppDegree && (
                    <p className="text-xs text-yellow-400">
                      DualPipeV needs m {'\u2265'} 2×PP ({2 * config.training.ppDegree}), currently {config.training.gradientAccumulationSteps}. Bubble reduction degraded.
                    </p>
                  )}
                </>
              )}
            </>
          )}
          {config.modelSpec?.isMoE && ['fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(config.training.strategyType) && (
            <NumberInput
              label="Expert Parallel Degree"
              value={config.training.epDegree}
              onChange={(value) => config.setStrategyParams({ epDegree: value })}
              min={1}
            />
          )}
          {['ddp', 'fsdp', 'zero-1', 'fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(config.training.strategyType) && (
            <NumberInput
              label="Context Parallel Degree"
              value={config.training.cpDegree}
              onChange={(value) => config.setStrategyParams({ cpDegree: value })}
              min={1}
            />
          )}
          {config.training.cpDegree > 1 && (
            <div>
              <label className="block text-xs text-gray-400 mb-1">CP Implementation</label>
              <select
                className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-white"
                value={config.training.cpImplementation}
                onChange={(e) => {
                  config.setStrategyParams({
                    cpImplementation: e.target.value as 'ring' | 'all-gather',
                  });
                }}
              >
                <option value="ring">Ring Attention</option>
                <option value="all-gather">All-Gather (Megatron)</option>
              </select>
            </div>
          )}
          {['fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'].includes(config.training.strategyType) && (
            <>
              <Toggle
                label="Sequence Parallelism"
                description="Reduce activation memory across TP group"
                checked={config.training.sequenceParallel}
                onChange={(checked) => config.setStrategyParams({ sequenceParallel: checked })}
              />
              <div className="text-xs text-gray-500">
                DP Degree: {config.training.dpDegree}
                {config.training.cpDegree > 1 && ` · CP: ${config.training.cpDegree}`}
                {config.training.epDegree > 1 && ` · EP: ${config.training.epDegree}`}
              </div>
            </>
          )}
        </CollapsibleSection>
      )}

      {/* Training Parameters - Training Mode Only */}
      {(config.mode === 'training') && (
        <CollapsibleSection
          title="Configuration"
          icon={<Settings className="w-4 h-4 text-accent" />}
        >
          <NumberInput
            label="Global Batch Size"
            value={config.training.globalBatchSize}
            onChange={(value) => config.setTrainingParams({ globalBatchSize: value })}
            min={1}
          />
          <NumberInput
            label="Micro Batch Size"
            value={config.training.microBatchSize}
            onChange={(value) => config.setTrainingParams({ microBatchSize: value })}
            min={1}
          />
          {config.training.microBatchSize > config.training.globalBatchSize && (
            <p className="text-xs text-yellow-400">
              Micro batch size exceeds global batch size.
            </p>
          )}
          <NumberInput
            label="Sequence Length"
            value={config.sequenceLength}
            onChange={(value) => config.setSequenceLength(value)}
            min={128}
            step={128}
          />
          <Select
            label="Precision"
            value={config.precision}
            onChange={(value) => config.setPrecision(value as typeof config.precision)}
            options={precisionOptions}
          />
          <Toggle
            label="Flash Attention"
            description={gpuSupportsFA ? 'Memory-efficient attention kernel (FA2)' : `Not supported on ${config.clusterConfig?.node.gpu.name ?? 'this GPU'}`}
            checked={config.training.flashAttention}
            onChange={(checked) => config.setTrainingParams({ flashAttention: checked })}
            disabled={!gpuSupportsFA}
          />
          <SegmentedControl
            label="Activation Checkpointing"
            value={!config.training.activationCheckpointing ? 'none' : config.training.checkpointingGranularity}
            onChange={(value) => {
              if (value === 'none') {
                config.setTrainingParams({ activationCheckpointing: false });
              } else {
                config.setTrainingParams({
                  activationCheckpointing: true,
                  checkpointingGranularity: value as 'full' | 'selective',
                });
              }
            }}
            options={[
              { value: 'selective', label: 'Selective', tooltip: 'Recompute attention only' },
              { value: 'full', label: 'Full', tooltip: 'Recompute every layer' },
              { value: 'none', label: 'Off', tooltip: 'Keep all activations in memory' },
            ]}
          />
          {config.training.activationCheckpointing &&
           config.training.checkpointingGranularity === 'selective' && (() => {
            const isAuto = config.training.selectiveStoredLayers === 'auto';
            const maxLayers = config.modelSpec?.numLayers ?? 999;
            const resolved = simulation.metrics?.resolvedStoredLayers;

            if (isAuto) {
              return (
                <div className="relative">
                  <label className="block text-xs text-gray-400 mb-0.5">Stored Layers</label>
                  <div className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-gray-400">
                    {resolved != null ? `Auto (${resolved} of ${maxLayers})` : 'Auto'}
                  </div>
                  <button
                    onClick={() => config.setTrainingParams({
                      selectiveStoredLayers: resolved ?? maxLayers,
                    })}
                    className="absolute top-0 right-0 text-xs text-gray-400 hover:text-accent cursor-pointer"
                  >
                    override
                  </button>
                </div>
              );
            }

            return (
              <div className="relative">
                <NumberInput
                  label={`Stored Layers (of ${maxLayers})`}
                  value={config.training.selectiveStoredLayers as number}
                  onChange={(value) => config.setTrainingParams({ selectiveStoredLayers: value })}
                  min={0}
                  max={maxLayers}
                />
                <Tooltip text="Auto-pick stored layers to fill available memory" className="contents">
                  <button
                    onClick={() => config.setTrainingParams({ selectiveStoredLayers: 'auto' })}
                    className="absolute top-0 right-0 text-xs text-gray-400 hover:text-accent cursor-pointer"
                  >
                    reset to auto
                  </button>
                </Tooltip>
              </div>
            );
          })()}
          <div className="space-y-2">
            <SegmentedControl
              label="Training Method"
              value={config.training.finetuningMethod}
              onChange={(value) => config.setTrainingParams({ finetuningMethod: value as FinetuningMethod })}
              options={[
                { value: 'full', label: 'Full', tooltip: 'Train all parameters' },
                { value: 'lora', label: 'LoRA', tooltip: 'Low-rank adapters' },
                { value: 'qlora', label: 'QLoRA', tooltip: 'LoRA + 4-bit base weights' },
              ]}
            />
            {config.training.finetuningMethod !== 'full' && (
              <>
                <Select
                  label="LoRA Rank"
                  value={String(config.training.loraRank)}
                  onChange={(value) => config.setTrainingParams({ loraRank: Number(value) })}
                  options={[
                    { value: '4', label: 'r = 4' },
                    { value: '8', label: 'r = 8' },
                    { value: '16', label: 'r = 16' },
                    { value: '32', label: 'r = 32' },
                    { value: '64', label: 'r = 64' },
                  ]}
                />
                <Select
                  label="Target Modules"
                  value={config.training.loraTargetModules}
                  onChange={(value) => config.setTrainingParams({ loraTargetModules: value as LoraTargetModules })}
                  options={[
                    { value: 'q_v', label: 'Q, V only' },
                    { value: 'q_k_v_o', label: 'Q, K, V, O' },
                    { value: 'all_linear', label: 'All linear layers' },
                  ]}
                />
                {config.modelSpec && (() => {
                  const trainable = computeLoraTrainableParams(
                    config.modelSpec!, config.training.loraRank, config.training.loraTargetModules
                  );
                  const total = config.modelSpec!.totalParams;
                  const pct = (trainable / total * 100);
                  return (
                    <div className="text-xs text-gray-400 mt-1 px-1">
                      Trainable: {formatNumber(trainable)} / {formatNumber(total)} ({pct < 0.01 ? '<0.01' : pct.toFixed(2)}%)
                    </div>
                  );
                })()}
              </>
            )}
          </div>
          <div className="space-y-2">
            <Select
              label="Training Scale"
              value={config.training.trainingGoal}
              onChange={(value) => config.setTrainingGoal(value as TrainingGoal)}
              options={[
                { value: 'chinchilla', label: 'Chinchilla Optimal (~20x params)' },
                { value: 'heavy-overtrain', label: 'Heavy Overtrain (~200x params)' },
                { value: 'finetune', label: 'Fine-tune (1B tokens)' },
                { value: 'custom', label: 'Custom' },
              ]}
            />
            {config.training.trainingGoal === 'custom' && (
              <NumberInput
                label="Target Tokens (B)"
                value={Math.round(config.training.targetTokens / 1e9)}
                onChange={(value) => config.setTargetTokens(value * 1e9)}
                min={1}
                step={10}
              />
            )}
            <div className="text-xs text-gray-500 space-y-1 mt-2">
              <div>Target: {formatTokens(config.training.targetTokens)}</div>
              <div>Steps: {formatNumber(config.training.globalBatchSize > 0 && config.sequenceLength > 0
                ? Math.ceil(config.training.targetTokens / (config.training.globalBatchSize * config.sequenceLength))
                : 0)}</div>
            </div>
          </div>
        </CollapsibleSection>
      )}

      {/* Inference Parameters - Inference Mode Only */}
      {(config.mode === 'inference') && (
        <CollapsibleSection
          title="Configuration"
          icon={<Zap className="w-4 h-4 text-accent" />}
        >
          {config.numGPUs > 1 && (
            <NumberInput
              label="Tensor Parallel Degree"
              value={config.inference.tensorParallel || 1}
              onChange={(value) => config.setInferenceParams({ tensorParallel: value })}
              min={1}
            />
          )}
          {config.numGPUs > 1 && config.modelSpec?.isMoE && (
            <NumberInput
              label="Expert Parallel Degree"
              value={config.inference.expertParallel || 1}
              onChange={(value) => config.setInferenceParams({ expertParallel: value })}
              min={1}
            />
          )}
          <NumberInput
            label="Batch Size"
            value={config.inference.batchSize}
            onChange={(value) => config.setInferenceParams({ batchSize: value })}
            min={1}
          />
          <NumberInput
            label="Input Sequence Length"
            value={config.inference.inputSeqLen}
            onChange={(value) => config.setInferenceParams({ inputSeqLen: value })}
            min={1}
            step={64}
          />
          <NumberInput
            label="Output Tokens"
            value={config.inference.outputSeqLen}
            onChange={(value) => config.setInferenceParams({ outputSeqLen: value })}
            min={1}
            step={64}
          />
          <Select
            label="Weight Precision"
            value={config.inference.weightPrecision}
            onChange={(value) => config.setInferenceParams({ weightPrecision: value as typeof config.inference.weightPrecision })}
            options={quantizationOptions}
          />
          <Select
            label="KV Cache Precision"
            value={config.inference.kvCachePrecision}
            onChange={(value) => config.setInferenceParams({ kvCachePrecision: value as typeof config.inference.kvCachePrecision })}
            options={[
              { value: 'bf16', label: 'BF16' },
              { value: 'fp16', label: 'FP16' },
              { value: 'fp8', label: 'FP8' },
              { value: 'int8', label: 'INT8' },
              { value: 'fp4', label: 'FP4' },
            ]}
          />
          <Toggle
            label="Flash Attention"
            description={gpuSupportsFA ? 'Memory-efficient attention kernel (FA2)' : `Not supported on ${config.clusterConfig?.node.gpu.name ?? 'this GPU'}`}
            checked={config.inference.flashAttention}
            onChange={(checked) => config.setInferenceParams({ flashAttention: checked })}
            disabled={!gpuSupportsFA}
          />
          <Toggle
            label="Continuous Batching"
            description="Iteration-level scheduling"
            checked={config.inference.continuousBatching}
            onChange={(checked) => config.setInferenceParams({ continuousBatching: checked })}
          />
          <Toggle
            label="Speculative Decoding"
            description="Use draft model for faster generation"
            checked={config.inference.speculativeDecoding}
            onChange={(checked) => config.setSpeculativeDecoding(checked)}
          />
          {config.inference.speculativeDecoding && (
            <>
              <ModelSelector
                label="Draft Model"
                placeholder="Search draft models..."
                value={config.inference.draftModelId || ''}
                onChange={(id) => config.setDraftModel(id || null)}
                groupedModels={groupedModels
                  .map(g => ({ ...g, options: g.options.filter(o => o.value !== config.modelId) }))
                  .filter(g => g.options.length > 0)}
              />
              <NumberInput
                label="Speculative Tokens (K)"
                value={config.inference.numSpeculativeTokens}
                onChange={(value) => config.setInferenceParams({ numSpeculativeTokens: value })}
                min={1}
              />
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Acceptance Rate: {(config.inference.acceptanceRate * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={config.inference.acceptanceRate * 100}
                  onChange={(e) => config.setInferenceParams({ acceptanceRate: Number(e.target.value) / 100 })}
                  className="w-full"
                />
              </div>
            </>
          )}
        </CollapsibleSection>
      )}

      {/* Optimization result banner — full-width section */}
      {simulation.optimizationResult && simulation.optimizationResult.changelog.length > 0 && (
        <div ref={optBannerRef} className="px-3 py-3 opt-banner border-b border-gray-800 text-xs">
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-3.5 h-3.5 opt-banner-icon" />
            <span className="text-sm font-medium opt-banner-title">Auto-Optimized</span>
          </div>
          <div className="text-gray-300 space-y-0.5">
            {simulation.optimizationResult.changelog.map((e) => (
              <div key={e.field}><span className="font-semibold">{changelogLabel(e.field)}:</span> {formatChangelogValue(e.field, e.from)} → {formatChangelogValue(e.field, e.to)}</div>
            ))}
          </div>
          <div className="mt-1.5 opt-banner-metric font-medium">
            {simulation.optimizationResult.target === 'training' ? (
              !isFinite(simulation.optimizationResult.beforeMetric)
                ? `Training time: OOM → ${formatDuration(simulation.optimizationResult.afterMetric)}`
                : (() => {
                    const before = simulation.optimizationResult.beforeMetric;
                    const after = simulation.optimizationResult.afterMetric;
                    const saved = before - after;
                    const pct = ((saved / before) * 100).toFixed(0);
                    return `${formatDuration(saved)} faster (−${pct}%)`;
                  })()
            ) : simulation.optimizationResult.target === 'throughput' ? (
              (() => {
                const before = simulation.optimizationResult.beforeMetric;
                const after = simulation.optimizationResult.afterMetric;
                const gain = after - before;
                const pct = ((gain / Math.max(1, before)) * 100).toFixed(0);
                return `+${formatWithCommas(Math.round(gain))} tok/s (+${pct}%)`;
              })()
            ) : (
              (() => {
                const before = simulation.optimizationResult.beforeMetric;
                const after = simulation.optimizationResult.afterMetric;
                const saved = before - after;
                const pct = ((saved / before) * 100).toFixed(0);
                return `${saved.toFixed(1)}ms faster (−${pct}%)`;
              })()
            )}
          </div>
          {simulation.optimizationResult.target === 'training' && (
            <div className="mt-1 text-[10px] text-gray-500">
              Optimizes for training time — actual convergence depends on model architecture, data, and hyperparameters.
            </div>
          )}
        </div>
      )}

      </div>{/* end scrollable area */}

      {/* Run Button — sticky footer */}
      <div className="shrink-0 border-t border-gray-800 p-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              if (config.mode === 'training') {
                simulation.runTrainingSimulation();
              } else {
                simulation.runInferenceSimulation();
              }
            }}
            disabled={simulation.isOptimizing}
            className="relative flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-accent hover:bg-accent disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors cursor-pointer overflow-hidden"
          >
            {simulation.runCounter > 0 && (
              <div
                key={simulation.runCounter}
                className="absolute inset-0 bg-white/20 animate-run-shade"
              />
            )}
            <Play className="w-4 h-4 relative z-10" />
            <span className="relative z-10">Run {config.mode === 'training' ? 'Training' : 'Inference'}</span>
            <span className="relative z-10 text-[10px] text-white/40 ml-1 translate-y-[2px]">Ctrl+Enter</span>
          </button>
          {config.mode === 'training' && !gameActive && (
            <Tooltip text="Auto-Optimize  Ctrl+Shift+Enter">
              <button
                onClick={() => simulation.autoOptimizeTraining()}
                disabled={simulation.isOptimizing || simulation.status === 'running'}
                className="flex items-center px-2 py-2 bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors cursor-pointer"
              >
                <Zap className="w-4 h-4" />
              </button>
            </Tooltip>
          )}
        </div>
      </div>
    </aside>
  );
}
