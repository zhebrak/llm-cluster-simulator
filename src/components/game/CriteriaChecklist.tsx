/**
 * Criteria checklist — displays winning criteria with check/x/circle indicators
 */

import { Check, X, Circle } from 'lucide-react';
import type { WinningCriterion, ValidationResult } from '../../game/types.ts';

interface CriteriaChecklistProps {
  criteria: WinningCriterion[];
  validation: ValidationResult | null;  // null = not yet evaluated
  cleared?: boolean;  // all criteria passed (completed objective)
}

export function CriteriaChecklist({ criteria, validation, cleared }: CriteriaChecklistProps) {
  return (
    <ul className="space-y-1.5">
      {criteria.map((criterion, i) => {
        const result = validation?.results[i];
        const met = result?.met;
        const state = cleared ? 'passed' : validation === null ? 'pending' : met ? 'passed' : 'failed';

        return (
          <li key={i} className="flex items-center gap-2 text-sm">
            {state === 'pending' && (
              <Circle className="w-4 h-4 text-accent flex-shrink-0" />
            )}
            {state === 'passed' && (
              <Check className="w-4 h-4 text-accent flex-shrink-0" />
            )}
            {state === 'failed' && (
              <X className="w-4 h-4 text-red-400 flex-shrink-0" />
            )}
            <span className={
              state === 'passed' ? 'text-accent' :
              state === 'failed' ? 'text-red-300' :
              'text-accent'
            }>
              {criterion.label}
            </span>
          </li>
        );
      })}
    </ul>
  );
}
