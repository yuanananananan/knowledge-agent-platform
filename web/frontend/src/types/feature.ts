import type { Modality } from './dataset';
import type { ParamsSchema } from './params-schema';

export type Feature = {
  id: number;
  name: string;
  modality: Modality;
  description: string;
  paramsSchema: ParamsSchema;
};

export type FeatureGroup = {
  id: number;
  name: string;
  description: string;
  featureCount: number;
  createTime: string;
};

export type FeatureGroupDetail = {
  groupId: number;
  groupName: string;
  features: Feature[];
};

export type FeatureGroupWithDetails = FeatureGroup & {
  features: Feature[];
};
