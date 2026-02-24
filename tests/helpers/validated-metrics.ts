/**
 * Test utility: validated simulation metrics.
 *
 * Wraps getSimulationMetrics with a validation step that throws on config
 * errors (pipeline can't fill, etc.).  Use this in tests where the config
 * represents a valid, runnable training setup.  Tests that intentionally
 * exercise invalid/OOM configs should continue using the raw
 * getSimulationMetrics (which skips validation).
 */

import {
  SimulationEngine,
  type SimulationConfig,
  type SimulationMetrics,
} from '../../src/core/simulation/engine.ts';

export function getValidatedSimulationMetrics(config: SimulationConfig): SimulationMetrics {
  const engine = new SimulationEngine();
  engine.configure(config);
  const validation = engine.validate();
  if (validation.errors.length > 0) {
    throw new Error(
      `Simulation config has validation errors:\n  ${validation.errors.join('\n  ')}`
    );
  }
  return engine.simulate();
}

/**
 * Validate an already-configured engine, throwing on errors.
 * Use before engine.simulate() in tests that instantiate SimulationEngine directly.
 */
export function assertValidEngine(engine: SimulationEngine): void {
  const validation = engine.validate();
  if (validation.errors.length > 0) {
    throw new Error(
      `Simulation config has validation errors:\n  ${validation.errors.join('\n  ')}`
    );
  }
}
