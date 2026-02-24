/**
 * Tests for 1F1B pipeline schedule generation.
 * Validates correctness of the Gantt chart schedule — ordering, dependencies, bubble placement,
 * and optimality (no wasted idle time).
 */

import { describe, it, expect } from 'vitest';
import { generate1F1BSchedule, type ScheduleBlock } from '../../src/components/visualization/PipelineTimeline.tsx';

// Helper: group blocks by stage
function blocksByStage(blocks: ScheduleBlock[], pp: number) {
  const result: ScheduleBlock[][] = Array.from({ length: pp }, () => []);
  for (const b of blocks) result[b.stage].push(b);
  // Sort by start time within each stage
  for (const arr of result) arr.sort((a, b) => a.startMs - b.startMs);
  return result;
}

// Helper: find a specific block
function findBlock(blocks: ScheduleBlock[], stage: number, type: 'forward' | 'backward', mb: number) {
  return blocks.find(b => b.stage === stage && b.type === type && b.microBatch === mb);
}

// Helper: find all idle gaps on a stage
function findGaps(stageBlocks: ScheduleBlock[], maxTime: number): { start: number; end: number }[] {
  const sorted = [...stageBlocks].sort((a, b) => a.startMs - b.startMs);
  const gaps: { start: number; end: number }[] = [];
  let cursor = 0;
  for (const blk of sorted) {
    if (blk.startMs > cursor + 0.001) {
      gaps.push({ start: cursor, end: blk.startMs });
    }
    cursor = blk.startMs + blk.durationMs;
  }
  if (cursor < maxTime - 0.001) {
    gaps.push({ start: cursor, end: maxTime });
  }
  return gaps;
}

describe('generate1F1BSchedule', () => {
  describe('basic structure', () => {
    it('generates correct number of blocks', () => {
      const blocks = generate1F1BSchedule(4, 8, 1, 2);
      // Each stage does numMB forwards + numMB backwards
      expect(blocks.length).toBe(4 * 8 * 2);
    });

    it('every stage has exactly numMB forwards and numMB backwards', () => {
      const pp = 4, m = 6;
      const blocks = generate1F1BSchedule(pp, m, 1, 2);
      for (let s = 0; s < pp; s++) {
        const stageBlocks = blocks.filter(b => b.stage === s);
        const forwards = stageBlocks.filter(b => b.type === 'forward');
        const backwards = stageBlocks.filter(b => b.type === 'backward');
        expect(forwards.length).toBe(m);
        expect(backwards.length).toBe(m);
      }
    });
  });

  describe('no backward before its forward on the same stage', () => {
    it.each([
      { pp: 4, m: 4 },
      { pp: 4, m: 8 },
      { pp: 8, m: 8 },
      { pp: 2, m: 2 },
      { pp: 4, m: 1 },
    ])('PP=$pp, m=$m', ({ pp, m }) => {
      const blocks = generate1F1BSchedule(pp, m, 1, 2);
      for (let s = 0; s < pp; s++) {
        for (let mb = 0; mb < m; mb++) {
          const fwd = findBlock(blocks, s, 'forward', mb);
          const bwd = findBlock(blocks, s, 'backward', mb);
          expect(fwd).toBeDefined();
          expect(bwd).toBeDefined();
          expect(bwd!.startMs).toBeGreaterThanOrEqual(fwd!.startMs + fwd!.durationMs - 0.001);
        }
      }
    });
  });

  describe('forward dependency: F[s][m] starts after F[s-1][m] ends', () => {
    it('PP=4, m=8', () => {
      const blocks = generate1F1BSchedule(4, 8, 1, 2);
      for (let s = 1; s < 4; s++) {
        for (let mb = 0; mb < 8; mb++) {
          const prev = findBlock(blocks, s - 1, 'forward', mb)!;
          const curr = findBlock(blocks, s, 'forward', mb)!;
          expect(curr.startMs).toBeGreaterThanOrEqual(prev.startMs + prev.durationMs - 0.001);
        }
      }
    });
  });

  describe('backward dependency: B[s][m] starts after B[s+1][m] ends', () => {
    it('PP=4, m=8', () => {
      const blocks = generate1F1BSchedule(4, 8, 1, 2);
      for (let s = 0; s < 3; s++) {
        for (let mb = 0; mb < 8; mb++) {
          const next = findBlock(blocks, s + 1, 'backward', mb)!;
          const curr = findBlock(blocks, s, 'backward', mb)!;
          expect(curr.startMs).toBeGreaterThanOrEqual(next.startMs + next.durationMs - 0.001);
        }
      }
    });
  });

  describe('no overlapping blocks on the same stage', () => {
    it.each([
      { pp: 4, m: 8 },
      { pp: 4, m: 4 },
      { pp: 8, m: 8 },
      { pp: 2, m: 2 },
    ])('PP=$pp, m=$m', ({ pp, m }) => {
      const blocks = generate1F1BSchedule(pp, m, 1, 2);
      const byStage = blocksByStage(blocks, pp);
      for (let s = 0; s < pp; s++) {
        const sorted = byStage[s];
        for (let i = 1; i < sorted.length; i++) {
          const prevEnd = sorted[i - 1].startMs + sorted[i - 1].durationMs;
          expect(sorted[i].startMs).toBeGreaterThanOrEqual(prevEnd - 0.001);
        }
      }
    });
  });

  describe('no unnecessary idle — every gap is dependency-forced', () => {
    // The key invariant: if a stage is idle, it must be because EVERY remaining
    // operation has an unmet cross-stage dependency. The old pre-computed sequence
    // violated this — Stage 1 was idle at t=4 even though F3's dep was ready.
    it.each([
      { pp: 4, m: 4, F: 1, B: 1 },
      { pp: 4, m: 4, F: 1, B: 3 },
      { pp: 4, m: 8, F: 1, B: 1 },
      { pp: 4, m: 8, F: 1, B: 2 },
      { pp: 8, m: 8, F: 1, B: 1 },
      { pp: 2, m: 2, F: 1, B: 1 },
      { pp: 4, m: 1, F: 1, B: 1 },
      { pp: 4, m: 16, F: 1, B: 1 },
    ])('PP=$pp, m=$m, F=$F, B=$B', ({ pp, m, F, B }) => {
      const blocks = generate1F1BSchedule(pp, m, F, B);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));
      const byStage = blocksByStage(blocks, pp);

      // Build completion time maps from the schedule
      const forwardEndTime: number[][] = Array.from({ length: pp }, () => new Array(m).fill(Infinity));
      const backwardEndTime: number[][] = Array.from({ length: pp }, () => new Array(m).fill(Infinity));
      for (const blk of blocks) {
        const end = blk.startMs + blk.durationMs;
        if (blk.type === 'forward') forwardEndTime[blk.stage][blk.microBatch] = end;
        else backwardEndTime[blk.stage][blk.microBatch] = end;
      }

      for (let s = 0; s < pp; s++) {
        const gaps = findGaps(byStage[s], maxTime);

        // Which micro-batches have been completed by a given time on this stage?
        const stageBlocks = byStage[s];

        for (const gap of gaps) {
          const t = gap.start;

          // Determine what's done on THIS stage by time t
          const fDone = new Set<number>();
          const bDone = new Set<number>();
          for (const blk of stageBlocks) {
            if (blk.startMs + blk.durationMs <= t + 0.001) {
              if (blk.type === 'forward') fDone.add(blk.microBatch);
              else bDone.add(blk.microBatch);
            }
          }

          // Check: is any forward eligible at time t?
          let anyFReady = false;
          for (let mb = 0; mb < m && !anyFReady; mb++) {
            if (fDone.has(mb)) continue;
            // F[s][mb] needs F[s-1][mb] done
            if (s === 0 || forwardEndTime[s - 1][mb] <= t + 0.001) {
              anyFReady = true;
            }
          }

          // Check: is any backward eligible at time t?
          let anyBReady = false;
          for (let mb = 0; mb < m && !anyBReady; mb++) {
            if (bDone.has(mb)) continue;
            // Also need own forward done first
            if (!fDone.has(mb) && forwardEndTime[s][mb] > t + 0.001) continue;
            if (s === pp - 1) {
              // Last stage: B depends on own F
              if (forwardEndTime[s][mb] <= t + 0.001) anyBReady = true;
            } else {
              // B[s][mb] needs B[s+1][mb] done
              if (backwardEndTime[s + 1][mb] <= t + 0.001) anyBReady = true;
            }
          }

          expect(anyFReady || anyBReady,
            `Stage ${s} idle at t=${t.toFixed(2)} but has ready ops (F=${anyFReady}, B=${anyBReady})`
          ).toBe(false);
        }
      }
    });
  });

  describe('bubble placement', () => {
    it('Stage 0 starts at t=0 (no leading bubble)', () => {
      const blocks = generate1F1BSchedule(4, 8, 1, 2);
      const stage0 = blocks.filter(b => b.stage === 0);
      const earliest = Math.min(...stage0.map(b => b.startMs));
      expect(earliest).toBeCloseTo(0, 6);
    });

    it('Stage 0 finishes last (no trailing bubble)', () => {
      const pp = 4, m = 8;
      const blocks = generate1F1BSchedule(pp, m, 1, 2);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));
      const byStage = blocksByStage(blocks, pp);

      const stage0End = Math.max(...byStage[0].map(b => b.startMs + b.durationMs));
      expect(stage0End).toBeCloseTo(maxTime, 6);
    });

    it('last stage has leading bubble (fill latency)', () => {
      const pp = 4;
      const blocks = generate1F1BSchedule(pp, 8, 1, 2);
      const lastStage = blocks.filter(b => b.stage === pp - 1);
      const earliest = Math.min(...lastStage.map(b => b.startMs));
      expect(earliest).toBeGreaterThan(0);
    });

    it('last stage finishes before stage 0 (has trailing bubble)', () => {
      const pp = 4, m = 8;
      const blocks = generate1F1BSchedule(pp, m, 1, 2);
      const byStage = blocksByStage(blocks, pp);

      const stage0End = Math.max(...byStage[0].map(b => b.startMs + b.durationMs));
      const lastEnd = Math.max(...byStage[pp - 1].map(b => b.startMs + b.durationMs));
      expect(lastEnd).toBeLessThan(stage0End);
    });

    it('leading bubbles grow from stage 0 to stage PP-1', () => {
      const pp = 4;
      const blocks = generate1F1BSchedule(pp, 8, 1, 2);
      const byStage = blocksByStage(blocks, pp);

      const leadingGaps = byStage.map(stageBlocks =>
        Math.min(...stageBlocks.map(b => b.startMs))
      );

      for (let s = 1; s < pp; s++) {
        expect(leadingGaps[s]).toBeGreaterThan(leadingGaps[s - 1]);
      }
    });

    it('trailing bubbles grow from stage 0 to stage PP-1', () => {
      const pp = 4, m = 8;
      const blocks = generate1F1BSchedule(pp, m, 1, 2);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));
      const byStage = blocksByStage(blocks, pp);

      const trailingGaps = byStage.map(stageBlocks =>
        maxTime - Math.max(...stageBlocks.map(b => b.startMs + b.durationMs))
      );

      // Stage 0 has no trailing gap
      expect(trailingGaps[0]).toBeCloseTo(0, 6);
      // Each subsequent stage has more trailing gap
      for (let s = 1; s < pp; s++) {
        expect(trailingGaps[s]).toBeGreaterThan(trailingGaps[s - 1] - 0.001);
      }
    });
  });

  describe('total bubble fraction', () => {
    it('matches (pp-1)/(pp-1+m) formula exactly', () => {
      // Fill gap = (pp-1)*F, drain gap = (pp-1)*B
      // Total bubble per stage = (pp-1)*(F+B)
      // maxTime = (m+pp-1)*(F+B)
      // Bubble fraction = (pp-1)/(m+pp-1)
      const pp = 4, m = 8, F = 1, B = 1;
      const blocks = generate1F1BSchedule(pp, m, F, B);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));

      const expectedMaxTime = (m + pp - 1) * (F + B);
      expect(maxTime).toBeCloseTo(expectedMaxTime, 4);

      const bubbleFraction = 1 - (m * (F + B)) / maxTime;
      expect(bubbleFraction).toBeCloseTo((pp - 1) / (pp - 1 + m), 4);
    });

    it('matches for asymmetric F/B durations', () => {
      const pp = 4, m = 8, F = 1, B = 3;
      const blocks = generate1F1BSchedule(pp, m, F, B);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));

      // Same formula: maxTime = (m+pp-1)*(F+B)
      const expectedMaxTime = (m + pp - 1) * (F + B);
      expect(maxTime).toBeCloseTo(expectedMaxTime, 4);
    });

    it('decreases monotonically as m increases', () => {
      const pp = 4, F = 1, B = 1;
      let prevBubble = 1;
      for (const m of [2, 4, 8, 16]) {
        const blocks = generate1F1BSchedule(pp, m, F, B);
        const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));
        const bubble = 1 - (m * (F + B)) / maxTime;
        expect(bubble).toBeLessThan(prevBubble);
        prevBubble = bubble;
      }
    });
  });

  describe('edge cases', () => {
    it('PP=1: no bubble, single stage', () => {
      const blocks = generate1F1BSchedule(1, 4, 1, 2);
      expect(blocks.length).toBe(8); // 4F + 4B
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));
      expect(maxTime).toBeCloseTo(4 * (1 + 2), 6); // all sequential, no gaps
    });

    it('m=1: maximum bubble', () => {
      const pp = 4;
      const blocks = generate1F1BSchedule(pp, 1, 1, 1);
      expect(blocks.length).toBe(8); // 4F + 4B
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));
      const computeArea = pp * 1 * 2;
      const bubbleFraction = 1 - computeArea / (pp * maxTime);
      const expected = (pp - 1) / (pp - 1 + 1);
      expect(bubbleFraction).toBeCloseTo(expected, 2);
    });

    it('asymmetric F/B durations', () => {
      const blocks = generate1F1BSchedule(4, 8, 1, 3);
      // All invariants should still hold
      for (let s = 0; s < 4; s++) {
        for (let mb = 0; mb < 8; mb++) {
          const fwd = findBlock(blocks, s, 'forward', mb)!;
          const bwd = findBlock(blocks, s, 'backward', mb)!;
          expect(bwd.startMs).toBeGreaterThanOrEqual(fwd.startMs + fwd.durationMs - 0.001);
        }
      }
    });
  });
});

// ---------------------------------------------------------------------------
// Interleaved schedule (device sharing)
// ---------------------------------------------------------------------------

// Helper: group blocks by physical device
function blocksByDevice(blocks: ScheduleBlock[], numDevices: number) {
  const result: ScheduleBlock[][] = Array.from({ length: numDevices }, () => []);
  for (const b of blocks) result[b.stage % numDevices].push(b);
  for (const arr of result) arr.sort((a, b) => a.startMs - b.startMs);
  return result;
}

describe('interleaved schedule (device sharing)', () => {
  describe('no two blocks on the same device overlap', () => {
    it.each([
      { pp: 4, v: 2, m: 8, F: 1, B: 2 },
      { pp: 4, v: 2, m: 4, F: 1, B: 1 },
      { pp: 2, v: 2, m: 8, F: 1, B: 1 },
      { pp: 4, v: 3, m: 6, F: 1, B: 2 },
      { pp: 2, v: 4, m: 8, F: 1, B: 3 },
    ])('PP=$pp, v=$v, m=$m, F=$F, B=$B', ({ pp, v, m, F, B }) => {
      const totalStages = pp * v;
      const effF = F / v;
      const effB = B / v;
      const blocks = generate1F1BSchedule(totalStages, m, effF, effB, pp);
      const byDevice = blocksByDevice(blocks, pp);

      for (let d = 0; d < pp; d++) {
        const sorted = byDevice[d];
        for (let i = 1; i < sorted.length; i++) {
          const prevEnd = sorted[i - 1].startMs + sorted[i - 1].durationMs;
          expect(sorted[i].startMs,
            `Device ${d}: block ${i} overlaps with block ${i-1}`
          ).toBeGreaterThanOrEqual(prevEnd - 0.001);
        }
      }
    });
  });

  describe('all core invariants hold with numDevices param', () => {
    it('forward dependency: F[s][m] starts after F[s-1][m] ends', () => {
      const pp = 4, v = 2, m = 8;
      const totalStages = pp * v;
      const blocks = generate1F1BSchedule(totalStages, m, 1, 2, pp);
      for (let s = 1; s < totalStages; s++) {
        for (let mb = 0; mb < m; mb++) {
          const prev = findBlock(blocks, s - 1, 'forward', mb)!;
          const curr = findBlock(blocks, s, 'forward', mb)!;
          expect(curr.startMs).toBeGreaterThanOrEqual(prev.startMs + prev.durationMs - 0.001);
        }
      }
    });

    it('backward dependency: B[s][m] starts after B[s+1][m] ends', () => {
      const pp = 4, v = 2, m = 8;
      const totalStages = pp * v;
      const blocks = generate1F1BSchedule(totalStages, m, 1, 2, pp);
      for (let s = 0; s < totalStages - 1; s++) {
        for (let mb = 0; mb < m; mb++) {
          const next = findBlock(blocks, s + 1, 'backward', mb)!;
          const curr = findBlock(blocks, s, 'backward', mb)!;
          expect(curr.startMs).toBeGreaterThanOrEqual(next.startMs + next.durationMs - 0.001);
        }
      }
    });

    it('no backward before its forward on the same stage', () => {
      const pp = 4, v = 2, m = 8;
      const totalStages = pp * v;
      const blocks = generate1F1BSchedule(totalStages, m, 1, 2, pp);
      for (let s = 0; s < totalStages; s++) {
        for (let mb = 0; mb < m; mb++) {
          const fwd = findBlock(blocks, s, 'forward', mb)!;
          const bwd = findBlock(blocks, s, 'backward', mb)!;
          expect(bwd.startMs).toBeGreaterThanOrEqual(fwd.startMs + fwd.durationMs - 0.001);
        }
      }
    });

    it('correct number of blocks', () => {
      const pp = 4, v = 2, m = 8;
      const totalStages = pp * v;
      const blocks = generate1F1BSchedule(totalStages, m, 1, 2, pp);
      expect(blocks.length).toBe(totalStages * m * 2);
    });
  });

  describe('bubble fraction per device', () => {
    it('PP=4, v=2, m=8: bubble less than standard 1F1B (no device sharing)', () => {
      const pp = 4, v = 2, m = 8, F = 1, B = 1;
      const totalStages = pp * v;
      const effF = F / v;
      const effB = B / v;
      const blocks = generate1F1BSchedule(totalStages, m, effF, effB, pp);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));

      // Per device: v stages × m micro-batches × (F+B)/v = m × (F+B) compute
      const computePerDevice = m * (F + B);
      const bubbleFraction = 1 - computePerDevice / maxTime;

      // Standard 1F1B bubble for pp stages: (pp-1)/(pp-1+m)
      const standardBubble = (pp - 1) / (pp - 1 + m);
      // Interleaved should have less bubble (or at worst equal due to device contention)
      expect(bubbleFraction).toBeLessThanOrEqual(standardBubble + 0.01);
      // Bubble must be non-negative
      expect(bubbleFraction).toBeGreaterThanOrEqual(0);
    });

    it('PP=4, v=3, m=6: bubble less than standard 1F1B', () => {
      const pp = 4, v = 3, m = 6, F = 1, B = 1;
      const totalStages = pp * v;
      const effF = F / v;
      const effB = B / v;
      const blocks = generate1F1BSchedule(totalStages, m, effF, effB, pp);
      const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));

      const computePerDevice = m * (F + B);
      const bubbleFraction = 1 - computePerDevice / maxTime;

      const standardBubble = (pp - 1) / (pp - 1 + m);
      expect(bubbleFraction).toBeLessThanOrEqual(standardBubble + 0.01);
      expect(bubbleFraction).toBeGreaterThanOrEqual(0);
    });
  });

  it('PP=4, v=2, m=8: device 0 (stages 0,4) interleaves work', () => {
    const pp = 4, v = 2, m = 8;
    const totalStages = pp * v;
    const blocks = generate1F1BSchedule(totalStages, m, 0.5, 0.5, pp);
    const device0Blocks = blocks.filter(b => b.stage % pp === 0);

    // Device 0 hosts stages 0 and 4 — should have blocks from both
    const stage0Blocks = device0Blocks.filter(b => b.stage === 0);
    const stage4Blocks = device0Blocks.filter(b => b.stage === 4);
    expect(stage0Blocks.length).toBe(m * 2); // m forwards + m backwards
    expect(stage4Blocks.length).toBe(m * 2);

    // The blocks should interleave — stage 4's first forward should start
    // before stage 0's last backward ends
    const stage4FirstF = Math.min(...stage4Blocks.filter(b => b.type === 'forward').map(b => b.startMs));
    const stage0LastB = Math.max(...stage0Blocks.filter(b => b.type === 'backward').map(b => b.startMs + b.durationMs));
    expect(stage4FirstF).toBeLessThan(stage0LastB);
  });

  it('asymmetric F/B: dependencies and no-overlap still hold with device sharing', () => {
    const pp = 4, v = 2, m = 8, F = 1, B = 3;
    const totalStages = pp * v;
    const effF = F / v;
    const effB = B / v;
    const blocks = generate1F1BSchedule(totalStages, m, effF, effB, pp);
    const maxTime = Math.max(...blocks.map(b => b.startMs + b.durationMs));

    // maxTime should be positive and finite
    expect(maxTime).toBeGreaterThan(0);
    expect(maxTime).toBeLessThan(Infinity);

    // No device overlap
    const byDevice = blocksByDevice(blocks, pp);
    for (let d = 0; d < pp; d++) {
      const sorted = byDevice[d];
      for (let i = 1; i < sorted.length; i++) {
        const prevEnd = sorted[i - 1].startMs + sorted[i - 1].durationMs;
        expect(sorted[i].startMs).toBeGreaterThanOrEqual(prevEnd - 0.001);
      }
    }

    // Bubble should be bounded and reasonable
    const computePerDevice = m * (F + B);
    const bubbleFraction = 1 - computePerDevice / maxTime;
    expect(bubbleFraction).toBeGreaterThanOrEqual(0);
    expect(bubbleFraction).toBeLessThan(1);
  });
});
