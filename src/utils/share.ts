/**
 * Shareable URL encoding/decoding for simulator configurations.
 *
 * Wire format uses short, stable key names decoupled from internal field names.
 * All fields are included (no default-omission) so URLs are deterministic.
 * Versioned with migrations for backward compatibility.
 */

import { useConfigStore } from '../stores/config.ts';

// ---------------------------------------------------------------------------
// Wire format types
// ---------------------------------------------------------------------------

export interface ShareConfigV1Training {
  v: 1;
  mode: 't';
  model: string;
  gpu: string;
  n: number;
  gpn: number;
  gbs: number;
  mbs: number;
  seq: number;
  st: string;
  tp: number;
  pp: number;
  dp: number;
  ep: number;
  cp: number;
  pr: string;
  ckpt: boolean;
  fa: boolean;
  sp: boolean;
  ps: string;
  is: number;
  goal: string;
  tok: string; // exponent notation, e.g. "15e12"
  price: number | null;
  // Activation checkpointing granularity (optional — old URLs without this decode to 'full')
  acg?: string;  // 's' = selective, omitted = full
  // Selective stored layers (omit for auto)
  sl?: number;
  // CP implementation (optional — old URLs without this decode to 'ring')
  cpi?: string;  // 'a' = all-gather, omitted = ring
  // Fine-tuning (optional — old URLs without these decode to 'full')
  ft?: string;   // FinetuningMethod
  lr?: number;   // loraRank
  ltm?: string;  // LoraTargetModules
}

export interface ShareConfigV1Inference {
  v: 1;
  mode: 'i';
  model: string;
  gpu: string;
  n: number;
  gpn: number;
  bs: number;
  iseq: number;
  oseq: number;
  wpr: string;
  kvpr: string;
  fa: boolean;
  pa: boolean;
  cb: boolean;
  tp: number;
  ep: number;
  sd: boolean;
  dm: string | null;
  nst: number;
  ar: number;
  price: number | null;
}

export type ShareConfig = ShareConfigV1Training | ShareConfigV1Inference;

// ---------------------------------------------------------------------------
// Base64url encoding
// ---------------------------------------------------------------------------

export function base64urlEncode(str: string): string {
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function base64urlDecode(str: string): string {
  let s = str.replace(/-/g, '+').replace(/_/g, '/');
  // Restore padding
  while (s.length % 4 !== 0) s += '=';
  return atob(s);
}

// ---------------------------------------------------------------------------
// Exponent notation for large numbers
// ---------------------------------------------------------------------------

/**
 * Convert a large number to engineering notation string.
 * Uses exponents e6, e9, e12 for readability.
 * Values < 1e6 return plain number string.
 */
export function toExponent(n: number): string {
  if (n === 0) return '0';
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '';

  for (const exp of [12, 9, 6] as const) {
    const divisor = 10 ** exp;
    if (abs >= divisor) {
      const coeff = abs / divisor;
      // Use integer coefficient if it divides evenly, otherwise up to 1 decimal
      const formatted = Number.isInteger(coeff)
        ? coeff.toString()
        : parseFloat(coeff.toPrecision(6)).toString();
      return `${sign}${formatted}e${exp}`;
    }
  }
  return n.toString();
}

// ---------------------------------------------------------------------------
// Build share URL from current config store state
// ---------------------------------------------------------------------------

export function buildShareURL(): string {
  const state = useConfigStore.getState();
  let wire: ShareConfig;

  if (state.mode === 'training') {
    const t = state.training;
    wire = {
      v: 1,
      mode: 't',
      model: state.modelId,
      gpu: state.gpuId,
      n: state.numGPUs,
      gpn: state.gpusPerNode,
      gbs: t.globalBatchSize,
      mbs: t.microBatchSize,
      seq: state.sequenceLength,
      st: t.strategyType,
      tp: t.tpDegree,
      pp: t.ppDegree,
      dp: t.dpDegree,
      ep: t.epDegree,
      cp: t.cpDegree,
      pr: state.precision,
      ckpt: t.activationCheckpointing,
      fa: t.flashAttention,
      sp: t.sequenceParallel,
      ps: t.pipelineSchedule,
      is: t.interleavedStages,
      goal: t.trainingGoal,
      tok: toExponent(t.targetTokens),
      price: state.pricePerGPUHour,
      // Selective checkpointing (only include when selective to keep URLs short)
      ...(t.activationCheckpointing && t.checkpointingGranularity === 'selective' ? { acg: 's' } : {}),
      // Selective stored layers (only include when selective AC + non-auto)
      ...(t.activationCheckpointing && t.checkpointingGranularity === 'selective' && t.selectiveStoredLayers !== 'auto'
        ? { sl: t.selectiveStoredLayers } : {}),
      // CP implementation (only include when all-gather and CP>1 to keep URLs short)
      ...(t.cpDegree > 1 && t.cpImplementation === 'all-gather' ? { cpi: 'a' } : {}),
      // Fine-tuning (only include when not 'full' to keep URLs short)
      ...(t.finetuningMethod !== 'full' ? {
        ft: t.finetuningMethod,
        lr: t.loraRank,
        ltm: t.loraTargetModules,
      } : {}),
    };
  } else {
    const inf = state.inference;
    wire = {
      v: 1,
      mode: 'i',
      model: state.modelId,
      gpu: state.gpuId,
      n: state.numGPUs,
      gpn: state.gpusPerNode,
      bs: inf.batchSize,
      iseq: inf.inputSeqLen,
      oseq: inf.outputSeqLen,
      wpr: inf.weightPrecision,
      kvpr: inf.kvCachePrecision,
      fa: inf.flashAttention,
      pa: inf.pagedAttention,
      cb: inf.continuousBatching,
      tp: inf.tensorParallel,
      ep: inf.expertParallel,
      sd: inf.speculativeDecoding,
      dm: inf.draftModelId,
      nst: inf.numSpeculativeTokens,
      ar: inf.acceptanceRate,
      price: state.pricePerGPUHour,
    };
  }

  const encoded = base64urlEncode(JSON.stringify(wire));
  const base = window.location.origin + window.location.pathname;
  return `${base}?config=${encoded}`;
}

// ---------------------------------------------------------------------------
// Migrations: upgrade old wire formats to current version
// ---------------------------------------------------------------------------

const CURRENT_VERSION = 1;

const MIGRATIONS: Record<number, (config: Record<string, unknown>) => Record<string, unknown>> = {
  // Example future migration:
  // 1: (c) => { c.ac = c.ckpt; delete c.ckpt; c.v = 2; return c; },
};

function applyMigrations(config: Record<string, unknown>): ShareConfig | null {
  let v = config.v;
  while (v as number < CURRENT_VERSION) {
    const migrate = MIGRATIONS[v as number];
    if (!migrate) return null; // Missing migration — can't upgrade
    config = migrate(config);
    v = config.v;
  }
  return config as unknown as ShareConfig;
}

// ---------------------------------------------------------------------------
// Decode share URL
// ---------------------------------------------------------------------------

export function decodeShareURL(encoded: string): ShareConfig | null {
  try {
    const json = base64urlDecode(encoded);
    const config = JSON.parse(json);

    if (!config || typeof config !== 'object') return null;

    // Must have a version field
    if (config.v === undefined) return null;

    if (config.v === CURRENT_VERSION) {
      return config as ShareConfig;
    }

    // Future version we don't understand
    if (config.v > CURRENT_VERSION) {
      return null;
    }

    // Older version — apply migrations
    return applyMigrations(config);
  } catch {
    return null;
  }
}
