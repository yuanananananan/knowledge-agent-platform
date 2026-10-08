import { httpService } from '@/services/http';
import type { Modality } from '@/types/dataset';

export type ExtraConfigDTO = {
  name: string;
  type: 'string' | 'number' | 'boolean';
  value: string | number | boolean;
  description: string;
};

export const taskApi = {
  async getExctraConfigByModality(modality: Modality) {
    const configFileMap: Record<Modality, string> = {
      SAR: 'sar_config.yaml',
      RD: 'rd_config.yaml',
      MON: 'mon_config.yaml',
    };

    return await httpService.get<{ extra: ExtraConfigDTO[] }>(
      '/api/config/schema',
      {
        params: {
          name: configFileMap[modality],
        },
      },
    );
  },
};