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
      { gpuId: 't4', count: 4, label: 'T4 16GB' },
      { gpuId: 'rtx-4090', count: 2, label: 'RTX 4090 24GB' },
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
];

/**
 * Get all available hardware tiers based on mission progression.
 * A tier is available when its `unlockedBy` mission is COMPLETED
 * or is null (starting hardware).
 */
export function getAvailableHardware(completedMissions: string[]): HardwareSlot[] {
  const slots: HardwareSlot[] = [];

  for (const tier of HARDWARE_PROGRESSION) {
    if (tier.unlockedBy === null) {
      slots.push(...tier.gpus);
      continue;
    }

    if (completedMissions.includes(tier.unlockedBy)) {
      slots.push(...tier.gpus);
    }
  }

  return slots;
}

/**
 * Get available tier IDs for hardware-seen tracking.
 */
export function getAvailableTierIds(completedMissions: string[]): string[] {
  const ids: string[] = [];

  for (const tier of HARDWARE_PROGRESSION) {
    if (tier.unlockedBy === null) {
      ids.push(tier.id);
      continue;
    }

    if (completedMissions.includes(tier.unlockedBy)) {
      ids.push(tier.id);
    }
  }

  return ids;
}
