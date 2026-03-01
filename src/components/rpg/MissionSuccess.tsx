/**
 * MissionSuccess — post-success modal with narrative and skill badge.
 * Amber (awards/CTA), teal (success icon), gray (narrative).
 */

import { Check, ArrowRight, Award } from 'lucide-react';
import { useRPGStore } from '../../stores/rpg.ts';
import { getMissionById } from '../../rpg/missions/index.ts';
import { ALL_SKILLS } from '../../rpg/skills.ts';
import { CriteriaChecklist } from '../game/CriteriaChecklist.tsx';
import { GlossaryText } from '../game/GlossaryText.tsx';

export function MissionSuccess() {
  const activeMissionId = useRPGStore(s => s.activeMissionId);
  const lastValidation = useRPGStore(s => s.lastValidation);
  const acknowledgeMissionSuccess = useRPGStore(s => s.acknowledgeMissionSuccess);
  const dismissSuccess = useRPGStore(s => s.dismissSuccess);

  if (!activeMissionId) return null;
  const mission = getMissionById(activeMissionId);
  if (!mission) return null;

  const awardedSkills = mission.skillsAwarded
    .map(id => ALL_SKILLS[id])
    .filter(Boolean);

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div
        className="bg-gray-950 border border-amber-500/30 rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-teal-500/20 flex items-center justify-center">
            <Check className="w-6 h-6 text-teal-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-amber-300 font-mono uppercase tracking-wider">
              Systems Restored
            </h3>
            <p className="text-sm text-gray-400 font-mono">{mission.title}</p>
          </div>
        </div>

        {/* Criteria recap */}
        <div className="mb-4 p-3 bg-gray-900/50 rounded-lg border border-gray-800 font-mono">
          <CriteriaChecklist criteria={mission.winningCriteria} validation={lastValidation} />
        </div>

        {/* Skills earned */}
        {awardedSkills.length > 0 && (
          <div className="mb-4">
            <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">Skill Acquired</div>
            <div className="flex flex-wrap gap-2">
              {awardedSkills.map(skill => (
                <div
                  key={skill.id}
                  className="flex items-center gap-1.5 px-3 py-1.5 border border-amber-500/40 rounded-full text-sm text-amber-300 bg-amber-500/5 font-mono"
                >
                  <Award className="w-3.5 h-3.5 text-amber-400" />
                  {skill.name}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Success narrative */}
        <div className="mb-6">
          <div className="space-y-2">
            {mission.successNarrative.trim().split('\n\n').map((block, i) => {
              const lines = block.trim().split('\n');
              const isList = lines.every(l => l.trimStart().startsWith('- '));
              if (isList) {
                return (
                  <ul key={i} className="space-y-1 ml-1 border-l-2 border-gray-700 pl-3">
                    {lines.map((line, j) => (
                      <li key={j} className="text-sm text-gray-300/70 leading-relaxed font-mono">
                        <GlossaryText text={line.trimStart().slice(2)} />
                      </li>
                    ))}
                  </ul>
                );
              }
              return (
                <p key={i} className="text-sm text-gray-300/70 leading-relaxed font-mono">
                  <GlossaryText text={block.trim()} />
                </p>
              );
            })}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={dismissSuccess}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 text-gray-400 hover:text-white border border-gray-700 hover:border-gray-600 rounded-lg font-medium font-mono transition-colors cursor-pointer"
          >
            Explore Results
          </button>
          <button
            onClick={acknowledgeMissionSuccess}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-teal-600 hover:bg-teal-500 text-white rounded-lg font-medium font-mono transition-colors cursor-pointer"
          >
            Continue
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
