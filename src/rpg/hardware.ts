/**
 * Hardware progression — tracks available GPUs as the story progresses.
 * Each tier is unlocked by completing a specific mission.
 */

export interface HardwareSlot {
  gpuId: string;
  count: number;
  label: string;         // "T4 16GB" or "A100 80GB"
}

export interface HardwareTier {
  id: string;
  name: string;          // "Original Compute Bay"
  description: string;   // "The ship's 80-year-old hardware"
  gpus: HardwareSlot[];
  unlockedBy: string | null;  // Mission ID whose COMPLETION makes this available (null = starting)
}

/**
 * Arc 1 hardware tiers, in progression order.
 */
export const HARDWARE_PROGRESSION: HardwareTier[] = [
  {
    id: 'starting',
    name: 'Original Compute Bay',
    description: "The ship's 80-year-old hardware, still running",
    gpus: [
      { gpuId: 't4', count: 2, label: 'T4 16GB' },
      { gpuId: 'rtx-4090', count: 4, label: 'RTX 4090 24GB' },
    ],
    unlockedBy: null,
  },
  {
    id: 'archive-vault',
    name: 'Archive Vault Module',
    description: 'Sealed module found near the reactor core',
    gpus: [
      { gpuId: 'a100-80gb', count: 4, label: 'A100 80GB' },
    ],
    unlockedBy: 'mission-1-6',
  },

  // ── Arc 2: Discovery ──────────────────────────────────────────────
  {
    id: 'derelict-station',
    name: 'Derelict Relay Station',
    description: 'An abandoned station drifting near the ship — 4 more A100 GPUs still functional',
    gpus: [
      { gpuId: 'a100-80gb', count: 4, label: 'A100 80GB' },
    ],
    unlockedBy: 'mission-2-2',
  },
  {
    id: 'resupply-drone',
    name: 'Earth Resupply Drone',
    description: 'A cargo drone from Earth — two sealed compute modules with H100 GPUs',
    gpus: [
      { gpuId: 'h100-sxm', count: 16, label: 'H100 SXM 80GB' },
    ],
    unlockedBy: 'mission-2-7',
  },

  // ── Arc 3: Wonder ───────────────────────────────────────────────────
  {
    id: 'asteroid-forge',
    name: 'Asteroid Fabrication Forge',
    description: 'The inner asteroid belt is rich in semiconductor-grade silicates. Chief Engineer Okafor\'s fabrication line has been forging compute modules — 62 full nodes, assembled and tested.',
    gpus: [
      { gpuId: 'h100-sxm', count: 496, label: 'H100 SXM 80GB' },
    ],
    unlockedBy: 'mission-3-1',
  },
];

function getUnlockedTiers(completedMissions: string[]): HardwareTier[] {
  return HARDWARE_PROGRESSION.filter(
    tier => tier.unlockedBy === null || completedMissions.includes(tier.unlockedBy),
  );
}

/**
 * Get all available hardware tiers based on mission progression.
 * A tier is available when its `unlockedBy` mission is COMPLETED
 * or is null (starting hardware).
 */
export function getAvailableHardware(completedMissions: string[]): HardwareSlot[] {
  return getUnlockedTiers(completedMissions).flatMap(tier => tier.gpus);
}

/**
 * Get available tier IDs for hardware-seen tracking.
 */
export function getAvailableTierIds(completedMissions: string[]): string[] {
  return getUnlockedTiers(completedMissions).map(tier => tier.id);
}
