import { httpService } from '@/services/http';
import type { Modality } from '@/types/dataset';
import type { Feature, FeatureGroup, FeatureGroupDetail } from '@/types/feature';
import type { ParamsSchema } from '@/types/params-schema';

type FeatureDTO = {
  id: number;
  name: string;
  modality: Modality;
  description?: string; // Backend may not provide description
  params_schema: ParamsSchema;
};

type FeatureGroupDTO = {
  id: number;
  name: string;
  description?: string; // Backend may not provide description
  feature_count: number;
  createTime: string;
};

type FeatureGroupDetailsDTO = {
  group_id: number;
  group_name: string;
  features: FeatureDTO[];
};

type UploadFeatureInput = {
  name: string;
  description: string;
  modality: Modality;
  code: string;
};

export const featureApi = {
  async getFeatureList(): Promise<Feature[]> {
    const data = await httpService.get<FeatureDTO[]>('/api/feature_extractor/');
    return data.map((item) => ({
      id: item.id,
      name: item.name,
      modality: item.modality,
      description: item.description ?? '', // Provide default empty string if description is missing
      paramsSchema: item.params_schema,
    }));
  },

  async createFeature(data: {
    name: string;
    modality: Modality;
    description: string;
    paramsSchema: string;
  }) {
    return await httpService.post('/api/feature_extractor/', {
      name: data.name,
      modality: data.modality,
      description: data.description,
      params_schema: data.paramsSchema,
    });
  },

  async deleteFeature(id: number) {
    return await httpService.delete(`/api/feature_extractor/${id}`);
  },

  async getFeatureGroups(): Promise<FeatureGroup[]> {
    const data = await httpService.get<FeatureGroupDTO[]>('/api/feature_extractor/group/');
    return data.map((item) => ({
      id: item.id,
      name: item.name,
      description: item.description ?? '', // Provide default empty string if description is missing
      featureCount: item.feature_count,
      createTime: item.createTime,
    }));
  },

  async getFeatureGroupDetails(groupId: number): Promise<FeatureGroupDetail> {
    const data = await httpService.get<FeatureGroupDetailsDTO>(
      `/api/feature_extractor/group/${groupId}`
    );
    return {
      groupId: data.group_id,
      groupName: data.group_name,
      features: data.features.map((item) => ({
        id: item.id,
        name: item.name,
        modality: item.modality,
        description: item.description ?? '', // Provide default empty string if description is missing
        paramsSchema: item.params_schema,
      })),
    };
  },

  async uploadFeatureGroup(data: {
    name: string;
    description: string;
    features: UploadFeatureInput[];
  }) {
    return await httpService.post('/api/feature_extractor/upload_json', {
      name: data.name,
      description: data.description,
      features: data.features,
    });
  },

  async createFeatureGroupByIds(data: { name: string; features: number[] }) {
    return await httpService.post('/api/feature_extractor/group/', {
      name: data.name,
      features: data.features,
    });
  },
};
