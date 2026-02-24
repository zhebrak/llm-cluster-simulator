/**
 * Custom model JSON editor panel.
 * Renders below the model selector when the user selects "Define Custom Model...".
 */

import { useState, useRef, useEffect } from 'react';
import { useConfigStore } from '../../stores/config.ts';
import { modelRegistry } from '../../core/models/registry.ts';
import { validateModelJSON, type ValidationResult } from '../../utils/model-validator.ts';
import { formatNumber } from '../../types/base.ts';

interface CustomModelEditorProps {
  onClose: () => void;
}

export function CustomModelEditor({ onClose }: CustomModelEditorProps) {
  const config = useConfigStore();

  // Get initial config from current model (or existing custom config)
  const getInitialJSON = (): string => {
    let cfg: Record<string, unknown> | undefined;
    if (config.modelId.startsWith('custom-') && config.customModels[config.modelId]) {
      cfg = { ...config.customModels[config.modelId] } as Record<string, unknown>;
    } else {
      const modelCfg = modelRegistry.getConfig(config.modelId);
      if (modelCfg) {
        cfg = { ...modelCfg } as Record<string, unknown>;
      }
    }
    if (cfg) {
      // Remove name from JSON (shown in separate input)
      const { name: _name, ...rest } = cfg;
      return JSON.stringify(rest, null, 2);
    }
    return JSON.stringify({
      numLayers: 32,
      hiddenSize: 4096,
      intermediateSize: 11008,
      numAttentionHeads: 32,
      vocabSize: 32000,
      maxSeqLength: 4096,
    }, null, 2);
  };

  const getInitialName = (): string => {
    if (config.modelId.startsWith('custom-') && config.customModels[config.modelId]) {
      return config.customModels[config.modelId].name;
    }
    const modelCfg = modelRegistry.getConfig(config.modelId);
    return modelCfg ? `${modelCfg.name} (Custom)` : 'Custom Model';
  };

  const [nameText, setNameText] = useState(getInitialName);
  const [jsonText, setJsonText] = useState(getInitialJSON);
  const [validation, setValidation] = useState<ValidationResult>(() =>
    validateModelJSON(jsonText, nameText)
  );
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const handleJsonChange = (text: string) => {
    setJsonText(text);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      const result = validateModelJSON(text, nameText);
      setValidation(result);
      // Sync: if JSON contains a name field, update the name input to match
      try {
        const parsed = JSON.parse(text);
        if (typeof parsed?.name === 'string' && parsed.name !== nameText) {
          setNameText(parsed.name);
        }
      } catch { /* invalid JSON, skip sync */ }
    }, 500);
  };

  const handleNameChange = (newName: string) => {
    setNameText(newName);
    // If JSON currently has a name field, update it to stay in sync
    try {
      const parsed = JSON.parse(jsonText);
      if (typeof parsed === 'object' && parsed !== null && 'name' in parsed) {
        parsed.name = newName;
        setJsonText(JSON.stringify(parsed, null, 2));
      }
    } catch { /* JSON invalid, don't clobber it */ }
    // Re-validate with new name
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setValidation(validateModelJSON(jsonText, newName));
    }, 200);
  };

  useEffect(() => () => clearTimeout(timerRef.current), []);

  const handleSave = () => {
    if (!validation.valid || !validation.config) return;
    const finalConfig = { ...validation.config, name: nameText };
    config.addCustomModel(finalConfig);
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Ctrl/Cmd+Enter to save
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && validation.valid) {
      e.preventDefault();
      handleSave();
    }
    // Escape to cancel
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  return (
    <div className="space-y-2" onKeyDown={handleKeyDown}>
      {/* Name input */}
      <div>
        <label className="block text-xs text-gray-400 mb-0.5">Name</label>
        <input
          type="text"
          value={nameText}
          onChange={(e) => handleNameChange(e.target.value)}
          className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 ring-accent focus:border-transparent"
          placeholder="Custom Model"
        />
      </div>

      {/* JSON editor */}
      <div>
        <label className="block text-xs text-gray-400 mb-0.5">Architecture (JSON)</label>
        <textarea
          value={jsonText}
          onChange={(e) => handleJsonChange(e.target.value)}
          spellCheck={false}
          rows={15}
          className="w-full px-2 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-md text-white font-mono focus:outline-none focus:ring-2 ring-accent focus:border-transparent resize-y"
          style={{ tabSize: 2 }}
        />
      </div>

      {/* Validation messages */}
      {(validation.errors.length > 0 || validation.warnings.length > 0) && (
        <div className="space-y-1 text-xs">
          {validation.errors.map((err, i) => (
            <div key={`e-${i}`} className="text-red-400">✗ {err}</div>
          ))}
          {validation.warnings.map((warn, i) => (
            <div key={`w-${i}`} className="text-yellow-400">⚠ {warn}</div>
          ))}
        </div>
      )}

      {/* Param estimate */}
      {validation.modelSpec && (
        <div className="text-xs text-gray-500">
          ~{formatNumber(validation.modelSpec.totalParams)} parameters
          {validation.modelSpec.isMoE && validation.modelSpec.activeParams && (
            <span> ({formatNumber(validation.modelSpec.activeParams)} active)</span>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 justify-end">
        <button
          onClick={onClose}
          className="px-3 py-1.5 text-sm text-gray-400 hover:text-white rounded-md hover:bg-gray-800 transition-colors cursor-pointer"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={!validation.valid}
          className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
            validation.valid
              ? 'bg-accent text-white hover:bg-accent/80 cursor-pointer'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
        >
          Save
        </button>
      </div>
    </div>
  );
}
