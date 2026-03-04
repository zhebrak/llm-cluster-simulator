/**
 * RPG game mode types.
 * Reuses TaskSetup, WinningCriterion, ExpectedChange from the shared game types.
 *
 * RPG Text Style Guide
 *
 * BRIEFINGS — Pure narrative. No ML acronyms (BF16, FSDP, TP, MFU, etc.),
 * no config notation (TP=4, batch=1), no simulator jargon (activation memory,
 * AllGather, optimizer state, weight matrices). Hardware names (A100, H100)
 * and size descriptions (70B, 140GB) are fine — they describe the world.
 * Characters can use terms they'd plausibly know (Okafor: "optimizer state").
 *
 * HINTS — Progressive reveal: concept → principle → directive.
 * First hint explains the concept. Middle hints narrow the approach.
 * Final hint can reveal the answer. ML terms OK in hints.
 *
 * SUCCESS NARRATIVES — Educational. ML terms welcome. This is where
 * the player learns the technical vocabulary and principles.
 */

import type { TaskSetup, WinningCriterion, ExpectedChange } from '../game/types.ts';

/**
 * Mission Design Principle: Physics over Rules
 *
 * All mission requirements must be justified by physics or narrative — never
 * by artificial rule enforcement. Three categories (in priority order):
 *
 * 1. PHYSICS: The simulator makes alternatives impossible (OOM, threshold
 *    unreachable). The player *cannot* bypass the intended lesson.
 *    Example: seqLen=16384 makes BF16+AC=off OOM at all micro-batch sizes.
 *
 * 2. NARRATIVE: The story explains why a constraint exists ("the colony needs
 *    this specific model deployed"). The player *would not* bypass it.
 *    Example: "the probe feed arrives as a stream" motivates continuous batching.
 *
 * 3. LEARNING OBJECTIVE: The mission explicitly teaches a specific skill.
 *    The expectedChanges check is upfront — the player knows they must
 *    demonstrate it. Used only when physics can't enforce and narrative alone
 *    isn't sufficient.
 *    Example: capstone requires continuousBatching: enabled as a demonstrated skill.
 *
 * Never use `unchanged` checks to silently block valid approaches. If a player
 * finds a physics-valid path that skips the intended lesson, either redesign
 * the setup so physics closes that path, or accept their solution.
 */

export interface RPGArc {
  id: string;           // 'arc1-survival'
  name: string;         // 'Survival'
  subtitle: string;     // 'The ship is old. Systems are failing.'
  description: string;
  order: number;
  closureText?: string; // Shown in ArcComplete modal when all missions done
  briefing?: string;       // Arc intro briefing shown in [BRIEFING] modal
  heroImage?: { dark: string; light: string }; // Hero image override
}

export interface MissionObjective {
  id: string;                         // 'obj-train', 'obj-infer'
  label: string;                      // "Training: Protein model fine-tune"
  primaryMode: 'training' | 'inference';
  setup: TaskSetup;
  winningCriteria: WinningCriterion[];
  expectedChanges?: ExpectedChange[];
}

export interface RPGMission {
  id: string;            // 'mission-1-1'
  arcId: string;
  order: number;         // display order within arc
  title: string;
  subtitle: string;      // one-line teaser for mission select
  briefing: string;      // narrative + task framing (GlossaryText syntax)
  successNarrative: string; // post-success story + explanation

  type?: 'mission' | 'pivot';  // default: 'mission'. Pivots have no sim task.
  learningObjectives: string[];  // 2-4 action-oriented objectives

  primaryMode: 'training' | 'inference';
  setup: TaskSetup;
  winningCriteria: WinningCriterion[];
  expectedChanges?: ExpectedChange[];
  hints: string[];

  /** Multi-objective missions: each objective has its own mode/setup/criteria. */
  objectives?: MissionObjective[];

  prerequisites: string[];   // mission IDs (DAG edges)
  skillsAwarded: string[];   // skill IDs earned on completion
}

export interface RPGSkill {
  id: string;
  name: string;
  description: string;
  starLabels: string[];  // one per star, ordered by mission progression
}
