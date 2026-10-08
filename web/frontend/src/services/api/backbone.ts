import { httpService } from '@/services/http';
import type { Backbone } from '@/types/backbone';
import type { ParamsSchema } from '@/types/params-schema';

type BackboneDTO = {
  id: number;
  name: string;
  description: string;
  params_schema: ParamsSchema;
  artifact_path?: string | null;
  modality?: string;
  backbone_type?: string;
};

export const backboneApi = {
  async getBackboneList(): Promise<Backbone[]> {
    const data = await httpService.get<BackboneDTO[]>('/api/backbone/');
    return data.map((item) => {
      return {
        id: item.id,
        name: item.name,
        modality: item.modality,
        backboneType: item.backbone_type,
        paramsSchema: item.params_schema,
        artifactPath: item.artifact_path,
        description: item.description ?? '',
      };
    });
  },
};
