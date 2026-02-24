/**
 * Simulation module exports
 */

export type { SimulationConfig, SimulationMetrics } from './engine.ts';
export {
  SimulationEngine,
  simulationEngine,
  runSimulation,
  getSimulationMetrics,
} from './engine.ts';

export type { OptimizationResult, ChangelogEntry } from './optimizer.ts';
export { optimizeTraining } from './optimizer.ts';
