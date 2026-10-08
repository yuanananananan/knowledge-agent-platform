import type { Modality } from '@/types/dataset';
import { MODALITY_LABELS, MODALITY_DESCRIPTIONS } from '@/constants/modality';

/**
 * Get the Chinese label for a modality
 */
export function getModalityLabel(modality: Modality): string {
  return MODALITY_LABELS[modality];
}

/**
 * Get the description for a modality
 */
export function getModalityDescription(modality: Modality): string {
  return MODALITY_DESCRIPTIONS[modality];
}

/**
 * Get modality options for select components
 */
export function getModalityOptions() {
  return Object.entries(MODALITY_LABELS).map(([value, label]) => ({
    value: value as Modality,
    label,
  }));
}
