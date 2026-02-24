/**
 * Share URL Tests
 *
 * Tests for shareable URL encoding/decoding, round-trip fidelity,
 * legacy format support, and migration framework.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  base64urlEncode,
  base64urlDecode,
  toExponent,
  decodeShareURL,
  type ShareConfigV1Training,
  type ShareConfigV1Inference,
} from '../../src/utils/share.ts';
import { PRECISION_SPECS } from '../../src/types/index.ts';
import { useConfigStore } from '../../src/stores/config.ts';

// ---------------------------------------------------------------------------
// Base64url encoding
// ---------------------------------------------------------------------------

describe('base64urlEncode / base64urlDecode', () => {
  it('round-trips arbitrary strings', () => {
    const inputs = ['hello', '{"v":1}', 'abc+/=', ''];
    for (const input of inputs) {
      expect(base64urlDecode(base64urlEncode(input))).toBe(input);
    }
  });

  it('produces URL-safe output (no +, /, =)', () => {
    const encoded = base64urlEncode('subjects?with+special/chars==');
    expect(encoded).not.toMatch(/[+/=]/);
  });
});

// ---------------------------------------------------------------------------
// toExponent
// ---------------------------------------------------------------------------

describe('toExponent', () => {
  it('formats trillions', () => {
    expect(toExponent(15e12)).toBe('15e12');
    expect(toExponent(1e12)).toBe('1e12');
    expect(toExponent(1.5e12)).toBe('1.5e12');
    expect(toExponent(1.2e12)).toBe('1.2e12');
    expect(toExponent(12.05e12)).toBe('12.05e12');
    expect(toExponent(1.555e12)).toBe('1.555e12');
  });

  it('formats billions with multi-digit coefficients', () => {
    // 1201e9 = 1.201e12, so toExponent picks the largest exponent
    expect(toExponent(1201e9)).toBe('1.201e12');
  });

  it('formats billions', () => {
    expect(toExponent(752e9)).toBe('752e9');
    expect(toExponent(1.5e9)).toBe('1.5e9');
    expect(toExponent(7e9)).toBe('7e9');
  });

  it('formats millions', () => {
    expect(toExponent(100e6)).toBe('100e6');
    expect(toExponent(1e6)).toBe('1e6');
  });

  it('returns plain number for values < 1e6', () => {
    expect(toExponent(500000)).toBe('500000');
    expect(toExponent(0)).toBe('0');
    expect(toExponent(42)).toBe('42');
  });

  it('values can be parsed back with Number()', () => {
    const values = [15e12, 752e9, 1.5e9, 100e6, 1e6, 500000, 0, 42,
      12.05e12, 1201e9, 1.555e12, 1.2e12];
    for (const v of values) {
      expect(Number(toExponent(v))).toBe(v);
    }
  });
});

// ---------------------------------------------------------------------------
// Round-trip: training config
// ---------------------------------------------------------------------------

describe('round-trip training config', () => {
  const trainingWire: ShareConfigV1Training = {
    v: 1,
    mode: 't',
    model: 'llama3-405b',
    gpu: 'h100-sxm',
    n: 16384,
    gpn: 8,
    gbs: 2048,
    mbs: 1,
    seq: 8192,
    st: 'fsdp-tp-pp',
    tp: 8,
    pp: 16,
    dp: 128,
    ep: 1,
    cp: 1,
    pr: 'bf16',
    ckpt: true,
    fa: true,
    sp: true,
    ps: 'interleaved-1f1b',
    is: 4,
    goal: 'custom',
    tok: '15e12',
    price: null,
  };

  it('encodes and decodes to the same config', () => {
    const encoded = base64urlEncode(JSON.stringify(trainingWire));
    const decoded = decodeShareURL(encoded);

    expect(decoded).not.toBeNull();
    expect(decoded!.v).toBe(1);
    expect(decoded!.mode).toBe('t');

    const t = decoded as ShareConfigV1Training;
    expect(t.model).toBe('llama3-405b');
    expect(t.gpu).toBe('h100-sxm');
    expect(t.n).toBe(16384);
    expect(t.gpn).toBe(8);
    expect(t.gbs).toBe(2048);
    expect(t.mbs).toBe(1);
    expect(t.seq).toBe(8192);
    expect(t.st).toBe('fsdp-tp-pp');
    expect(t.tp).toBe(8);
    expect(t.pp).toBe(16);
    expect(t.dp).toBe(128);
    expect(t.ep).toBe(1);
    expect(t.pr).toBe('bf16');
    expect(t.ckpt).toBe(true);
    expect(t.fa).toBe(true);
    expect(t.sp).toBe(true);
    expect(t.ps).toBe('interleaved-1f1b');
    expect(t.is).toBe(4);
    expect(t.goal).toBe('custom');
    expect(Number(t.tok)).toBe(15e12);
    expect(t.price).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Round-trip: inference config
// ---------------------------------------------------------------------------

describe('round-trip inference config', () => {
  const inferenceWire: ShareConfigV1Inference = {
    v: 1,
    mode: 'i',
    model: 'deepseek-v3',
    gpu: 'h200-sxm',
    n: 8,
    gpn: 8,
    bs: 32,
    iseq: 512,
    oseq: 256,
    wpr: 'fp8',
    kvpr: 'bf16',
    fa: true,
    pa: true,
    cb: false,
    tp: 8,
    ep: 1,
    sd: false,
    dm: null,
    nst: 4,
    ar: 0.7,
    price: null,
  };

  it('encodes and decodes to the same config', () => {
    const encoded = base64urlEncode(JSON.stringify(inferenceWire));
    const decoded = decodeShareURL(encoded);

    expect(decoded).not.toBeNull();
    expect(decoded!.mode).toBe('i');

    const inf = decoded as ShareConfigV1Inference;
    expect(inf.model).toBe('deepseek-v3');
    expect(inf.gpu).toBe('h200-sxm');
    expect(inf.n).toBe(8);
    expect(inf.bs).toBe(32);
    expect(inf.iseq).toBe(512);
    expect(inf.oseq).toBe(256);
    expect(inf.wpr).toBe('fp8');
    expect(inf.kvpr).toBe('bf16');
    expect(inf.fa).toBe(true);
    expect(inf.pa).toBe(true);
    expect(inf.tp).toBe(8);
    expect(inf.ep).toBe(1);
    expect(inf.sd).toBe(false);
    expect(inf.dm).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// All strategy types
// ---------------------------------------------------------------------------

describe('all strategy types round-trip', () => {
  const strategies = ['ddp', 'zero-1', 'fsdp', 'fsdp-tp', 'zero1-tp', 'ddp-tp-pp', 'zero1-tp-pp', 'fsdp-tp-pp'];

  for (const st of strategies) {
    it(`preserves strategy '${st}'`, () => {
      const wire: ShareConfigV1Training = {
        v: 1,
        mode: 't',
        model: 'llama2-7b',
        gpu: 'a100-80gb-sxm',
        n: 64,
        gpn: 8,
        gbs: 1024,
        mbs: 4,
        seq: 4096,
        st,
        tp: st.includes('tp') ? 8 : 1,
        pp: st.includes('pp') ? 4 : 1,
        dp: 64 / ((st.includes('tp') ? 8 : 1) * (st.includes('pp') ? 4 : 1)),
        ep: 1,
        cp: 1,
        pr: 'bf16',
        ckpt: true,
        fa: true,
        sp: st.includes('tp'),
        ps: '1f1b',
        is: 2,
        goal: 'chinchilla',
        tok: '140e9',
        price: null,
      };

      const encoded = base64urlEncode(JSON.stringify(wire));
      const decoded = decodeShareURL(encoded) as ShareConfigV1Training;

      expect(decoded).not.toBeNull();
      expect(decoded.st).toBe(st);
      expect(decoded.tp).toBe(wire.tp);
      expect(decoded.pp).toBe(wire.pp);
      expect(decoded.dp).toBe(wire.dp);
    });
  }
});

// ---------------------------------------------------------------------------
// Invalid input
// ---------------------------------------------------------------------------

describe('invalid input handling', () => {
  it('returns null for garbage base64', () => {
    expect(decodeShareURL('!!!not-base64!!!')).toBeNull();
  });

  it('returns null for valid base64 but not JSON', () => {
    expect(decodeShareURL(base64urlEncode('not json'))).toBeNull();
  });

  it('returns null for empty string', () => {
    expect(decodeShareURL('')).toBeNull();
  });

  it('returns null for JSON with unknown future version and no migration', () => {
    const futureConfig = { v: 999, mode: 't', model: 'future-model' };
    const encoded = base64urlEncode(JSON.stringify(futureConfig));
    expect(decodeShareURL(encoded)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// LoRA fine-tuning fields
// ---------------------------------------------------------------------------

describe('LoRA fine-tuning fields', () => {
  const trainingWire: ShareConfigV1Training = {
    v: 1,
    mode: 't',
    model: 'llama2-7b',
    gpu: 'a100-80gb-sxm',
    n: 8,
    gpn: 8,
    gbs: 32,
    mbs: 4,
    seq: 2048,
    st: 'fsdp',
    tp: 1,
    pp: 1,
    dp: 8,
    ep: 1,
    cp: 1,
    pr: 'bf16',
    ckpt: true,
    fa: true,
    sp: false,
    ps: '1f1b',
    is: 2,
    goal: 'chinchilla',
    tok: '140e9',
    price: null,
  };

  it('preserves LoRA fine-tuning fields', () => {
    const loraWire: ShareConfigV1Training = {
      ...trainingWire,
      ft: 'qlora',
      lr: 32,
      ltm: 'all_linear',
    };
    const encoded = base64urlEncode(JSON.stringify(loraWire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Training;

    expect(decoded).not.toBeNull();
    expect(decoded.ft).toBe('qlora');
    expect(decoded.lr).toBe(32);
    expect(decoded.ltm).toBe('all_linear');
  });

  it('omits LoRA fields for full training (keeps URLs short)', () => {
    // trainingWire has no ft/lr/ltm — verify they stay absent
    const encoded = base64urlEncode(JSON.stringify(trainingWire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Training;
    expect(decoded.ft).toBeUndefined();
    expect(decoded.lr).toBeUndefined();
    expect(decoded.ltm).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Custom pricing round-trip
// ---------------------------------------------------------------------------

describe('custom pricing', () => {
  it('preserves custom price', () => {
    const wire: ShareConfigV1Training = {
      v: 1,
      mode: 't',
      model: 'llama2-7b',
      gpu: 'a100-80gb-sxm',
      n: 8,
      gpn: 8,
      gbs: 1024,
      mbs: 4,
      seq: 4096,
      st: 'fsdp',
      tp: 1,
      pp: 1,
      dp: 8,
      ep: 1,
      cp: 1,
      pr: 'bf16',
      ckpt: true,
      fa: true,
      sp: false,
      ps: '1f1b',
      is: 2,
      goal: 'chinchilla',
      tok: '140e9',
      price: 3.50,
    };

    const encoded = base64urlEncode(JSON.stringify(wire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Training;
    expect(decoded.price).toBe(3.50);
  });

  it('preserves null price (default)', () => {
    const wire: ShareConfigV1Training = {
      v: 1,
      mode: 't',
      model: 'llama2-7b',
      gpu: 'a100-80gb-sxm',
      n: 8,
      gpn: 8,
      gbs: 1024,
      mbs: 4,
      seq: 4096,
      st: 'fsdp',
      tp: 1,
      pp: 1,
      dp: 8,
      ep: 1,
      cp: 1,
      pr: 'bf16',
      ckpt: true,
      fa: true,
      sp: false,
      ps: '1f1b',
      is: 2,
      goal: 'chinchilla',
      tok: '140e9',
      price: null,
    };

    const encoded = base64urlEncode(JSON.stringify(wire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Training;
    expect(decoded.price).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Speculative decoding fields
// ---------------------------------------------------------------------------

describe('speculative decoding fields', () => {
  it('preserves draft model and params', () => {
    const wire: ShareConfigV1Inference = {
      v: 1,
      mode: 'i',
      model: 'llama3-405b',
      gpu: 'h100-sxm',
      n: 8,
      gpn: 8,
      bs: 16,
      iseq: 1024,
      oseq: 512,
      wpr: 'bf16',
      kvpr: 'bf16',
      fa: true,
      pa: true,
      cb: false,
      tp: 8,
      ep: 1,
      sd: true,
      dm: 'llama3.1-8b',
      nst: 5,
      ar: 0.8,
      price: null,
    };

    const encoded = base64urlEncode(JSON.stringify(wire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Inference;

    expect(decoded.sd).toBe(true);
    expect(decoded.dm).toBe('llama3.1-8b');
    expect(decoded.nst).toBe(5);
    expect(decoded.ar).toBe(0.8);
  });
});

// ---------------------------------------------------------------------------
// Inference precision wire-level round-trip
// ---------------------------------------------------------------------------

describe('inference precision round-trip', () => {
  const inferenceWire: ShareConfigV1Inference = {
    v: 1,
    mode: 'i',
    model: 'deepseek-v3',
    gpu: 'h200-sxm',
    n: 8,
    gpn: 8,
    bs: 32,
    iseq: 512,
    oseq: 256,
    wpr: 'fp8',
    kvpr: 'bf16',
    fa: true,
    pa: true,
    cb: false,
    tp: 8,
    ep: 1,
    sd: false,
    dm: null,
    nst: 4,
    ar: 0.7,
    price: null,
  };

  it('GGUF weight precision preserved in wire format', () => {
    const wire: ShareConfigV1Inference = { ...inferenceWire, wpr: 'q4_k_m', kvpr: 'bf16' };
    const encoded = base64urlEncode(JSON.stringify(wire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Inference;
    expect(decoded.wpr).toBe('q4_k_m');
    expect(decoded.kvpr).toBe('bf16');
  });

  it('all InferencePrecision values survive wire round-trip', () => {
    for (const prec of Object.keys(PRECISION_SPECS)) {
      const wire: ShareConfigV1Inference = { ...inferenceWire, wpr: prec };
      const encoded = base64urlEncode(JSON.stringify(wire));
      const decoded = decodeShareURL(encoded) as ShareConfigV1Inference;
      expect(decoded.wpr).toBe(prec);
    }
  });

  it('INT8/INT4 preserved in wire format', () => {
    const wire: ShareConfigV1Inference = { ...inferenceWire, wpr: 'int4', kvpr: 'int8' };
    const encoded = base64urlEncode(JSON.stringify(wire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Inference;
    expect(decoded.wpr).toBe('int4');
    expect(decoded.kvpr).toBe('int8');
  });

  it('FP4 KV cache preserved in wire format', () => {
    const wire: ShareConfigV1Inference = { ...inferenceWire, wpr: 'bf16', kvpr: 'fp4' };
    const encoded = base64urlEncode(JSON.stringify(wire));
    const decoded = decodeShareURL(encoded) as ShareConfigV1Inference;
    expect(decoded.wpr).toBe('bf16');
    expect(decoded.kvpr).toBe('fp4');
  });
});

// ---------------------------------------------------------------------------
// loadShareConfig inference precision integration tests
// ---------------------------------------------------------------------------

describe('loadShareConfig inference precision', () => {
  const baseInferenceWire: ShareConfigV1Inference = {
    v: 1,
    mode: 'i',
    model: 'llama3.1-8b',
    gpu: 'h100-sxm',
    n: 1,
    gpn: 8,
    bs: 1,
    iseq: 512,
    oseq: 256,
    wpr: 'fp8',
    kvpr: 'bf16',
    fa: true,
    pa: true,
    cb: false,
    tp: 1,
    ep: 1,
    sd: false,
    dm: null,
    nst: 4,
    ar: 0.7,
    price: null,
  };

  beforeEach(() => {
    // Reset store to defaults before each test
    useConfigStore.getState().loadShareConfig({
      ...baseInferenceWire,
      wpr: 'fp8',
      kvpr: 'bf16',
    });
  });

  it('GGUF precision survives loadShareConfig (bug fix regression)', () => {
    useConfigStore.getState().loadShareConfig({
      ...baseInferenceWire,
      wpr: 'q4_k_m',
      kvpr: 'fp4',
    });
    const state = useConfigStore.getState();
    expect(state.inference.weightPrecision).toBe('q4_k_m');
    expect(state.inference.kvCachePrecision).toBe('fp4');
  });

  it('INT4/INT8 survives loadShareConfig', () => {
    useConfigStore.getState().loadShareConfig({
      ...baseInferenceWire,
      wpr: 'int4',
      kvpr: 'int8',
    });
    const state = useConfigStore.getState();
    expect(state.inference.weightPrecision).toBe('int4');
    expect(state.inference.kvCachePrecision).toBe('int8');
  });

  it('all 13 InferencePrecision values survive loadShareConfig', () => {
    const allPrecisions = Object.keys(PRECISION_SPECS);
    expect(allPrecisions.length).toBe(13);
    for (const prec of allPrecisions) {
      useConfigStore.getState().loadShareConfig({
        ...baseInferenceWire,
        wpr: prec,
      });
      expect(useConfigStore.getState().inference.weightPrecision).toBe(prec);
    }
  });

  it('invalid precision falls back to default in loadShareConfig', () => {
    useConfigStore.getState().loadShareConfig({
      ...baseInferenceWire,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      wpr: 'not_a_precision' as any,
    });
    // Should fall back to default (fp8)
    expect(useConfigStore.getState().inference.weightPrecision).toBe('fp8');
  });

  it('training precision validation unchanged', () => {
    const trainingWire: ShareConfigV1Training = {
      v: 1,
      mode: 't',
      model: 'llama2-7b',
      gpu: 'a100-80gb-sxm',
      n: 8,
      gpn: 8,
      gbs: 1024,
      mbs: 4,
      seq: 4096,
      st: 'fsdp',
      tp: 1,
      pp: 1,
      dp: 8,
      ep: 1,
      cp: 1,
      pr: 'bf16',
      ckpt: true,
      fa: true,
      sp: false,
      ps: '1f1b',
      is: 2,
      goal: 'chinchilla',
      tok: '140e9',
      price: null,
    };

    // Valid training precision should be preserved
    useConfigStore.getState().loadShareConfig(trainingWire);
    expect(useConfigStore.getState().precision).toBe('bf16');

    // GGUF precision should fall back for training (not a valid training precision)
    useConfigStore.getState().loadShareConfig({
      ...trainingWire,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      pr: 'q4_k_m' as any,
    });
    // Should fall back to default training precision (bf16)
    expect(['fp32', 'tf32', 'fp16', 'bf16', 'fp8', 'fp4']).toContain(
      useConfigStore.getState().precision
    );
  });
});
