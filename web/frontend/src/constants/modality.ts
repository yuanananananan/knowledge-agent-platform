import type { Modality } from '@/types/dataset';

/**
 * All available modality values
 */
export const MODALITIES = ['SAR', 'RD', 'MON'] as const;

/**
 * Chinese labels for each modality
 */
export const MODALITY_LABELS: Record<Modality, string> = {
  SAR: 'SAR图像',
  RD: 'RD图像',
  MON: 'MON图像',
} as const;

/**
 * Detailed descriptions for each modality
 */
export const MODALITY_DESCRIPTIONS: Record<Modality, string> = {
  SAR: '合成孔径雷达图像',
  RD: '距离-多普勒图像',
  MON: '单脉冲时序数据',
} as const;
