/**
 * RPG overlay root — routes based on store state.
 * Uses deriveOverlayState() pure function for testable state machine.
 */

import { useRPGStore } from '../../stores/rpg.ts';
import { getMissionById } from '../../rpg/missions/index.ts';
import { MissionSelect } from './MissionSelect.tsx';
import { MissionHUD } from './MissionHUD.tsx';
import { MissionSuccess } from './MissionSuccess.tsx';
import { ArcComplete } from './ArcComplete.tsx';
import { IntroBriefing } from './IntroBriefing.tsx';
import { PivotCutscene } from './PivotCutscene.tsx';

export type OverlayKind =
  | 'none'
  | 'intro-briefing'
  | 'arc-complete'
  | 'mission-select'
  | 'pivot-cutscene'
  | 'mission-hud-success'
  | 'mission-hud-menu'
  | 'mission-hud';

export interface OverlayInput {
  active: boolean;
  introSeen: boolean;
  showingBriefing: boolean;
  showArcComplete: string | null;
  activeMissionId: string | null;
  activeMissionType: 'mission' | 'pivot' | undefined;
  missionSelectDismissed: boolean;
  passed: boolean;
  approachValid: boolean;
  successDismissed: boolean;
  menuOpen: boolean;
}

/**
 * Pure state machine — testable without rendering.
 */
export function deriveOverlayState(input: OverlayInput): OverlayKind {
  if (!input.active) return 'none';
  if (!input.introSeen || input.showingBriefing) return 'intro-briefing';
  if (input.showArcComplete) return 'arc-complete';
  if (!input.activeMissionId && !input.missionSelectDismissed) return 'mission-select';
  if (!input.activeMissionId && input.missionSelectDismissed) return 'none';
  if (input.activeMissionType === 'pivot') return 'pivot-cutscene';
  if (input.passed && input.approachValid && !input.successDismissed) return 'mission-hud-success';
  if (input.menuOpen) return 'mission-hud-menu';
  if (input.activeMissionId) return 'mission-hud';
  return 'none';
}

export function RPGOverlay() {
  const active = useRPGStore(s => s.active);
  const introSeen = useRPGStore(s => s.introSeen);
  const showingBriefing = useRPGStore(s => s.showingBriefing);
  const activeMissionId = useRPGStore(s => s.activeMissionId);
  const lastValidation = useRPGStore(s => s.lastValidation);
  const approachValid = useRPGStore(s => s.approachValid);
  const successDismissed = useRPGStore(s => s.successDismissed);
  const showArcComplete = useRPGStore(s => s.showArcComplete);
  const menuOpen = useRPGStore(s => s.menuOpen);
  const missionSelectDismissed = useRPGStore(s => s.missionSelectDismissed);

  const mission = activeMissionId ? getMissionById(activeMissionId) : null;

  const overlayKind = deriveOverlayState({
    active,
    introSeen,
    showingBriefing,
    showArcComplete,
    activeMissionId,
    activeMissionType: mission?.type ?? (activeMissionId ? 'mission' : undefined),
    missionSelectDismissed,
    passed: lastValidation?.passed ?? false,
    approachValid,
    successDismissed,
    menuOpen,
  });

  switch (overlayKind) {
    case 'none':
      return null;
    case 'intro-briefing':
      return <IntroBriefing />;
    case 'arc-complete':
      return <ArcComplete />;
    case 'mission-select':
      return <MissionSelect />;
    case 'pivot-cutscene':
      return <PivotCutscene />;
    case 'mission-hud-success':
      return (
        <>
          <MissionHUD />
          <MissionSuccess />
        </>
      );
    case 'mission-hud-menu':
      return (
        <>
          <MissionHUD />
          <MissionSelect />
        </>
      );
    case 'mission-hud':
      return <MissionHUD />;
  }
}
