/**
 * Roofline model module exports
 */

export type {
  RooflinePoint,
  RooflineCeiling,
  RooflineData,
  TrainingRooflineConfig,
  InferenceRooflineConfig,
} from './compute.ts';

export {
  computeCeiling,
  computeTrainingRoofline,
  computeInferenceRoofline,
} from './compute.ts';
