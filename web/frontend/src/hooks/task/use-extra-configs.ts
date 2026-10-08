import { useQuery } from '@tanstack/react-query';
import { taskKeys } from './query-keys';
import { taskApi, type ExtraConfigDTO } from '@/services/api/task';
import type { Modality } from '@/types/dataset';
import { MODALITIES } from '@/constants/modality';

export function useExtraConfigs() {
  return useQuery({
    queryKey: taskKeys.extraConfigs(),
    queryFn: async () => {
      const results = await Promise.all(
        MODALITIES.map((modality) => taskApi.getExctraConfigByModality(modality))
      );
      return MODALITIES.reduce(
        (acc, modality, index) => {
          acc[modality] = results[index].extra;
          return acc;
        },
        {} as Record<Modality, ExtraConfigDTO[]>
      );
    },
  });
}
